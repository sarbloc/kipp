"""Split an oversized entity into topic entities (#52).

``plan_split`` groups an entity's current facts by topic and suggests a name
for each group. The plan is a JSON file an operator (or agent) edits: rename
targets, move facts between groups, drop groups. ``apply_plan`` then moves
each group's facts with ``move_facts``, which is also the MCP ``memory_move``
tool. Grouping is deterministic: no LLM, same input, same plan.

Moves never lose a fact. The target is written before the facts leave the
source, so a crash in between leaves a fact on both and re-running finishes
the move without doubling it. Moves serialise on a per-domain lock; a
``memory_store`` racing a move is last-writer-wins, as for any two KIPP writes
to one entity.
"""

from __future__ import annotations

import fcntl
import logging
import re
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import astuple, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Protocol

import numpy as np
from qdrant_client import QdrantClient

from kipp.client import get_entity
from kipp.facts import sync_until_stable, write_entity
from kipp.merge import longer_lifetime
from kipp.models import Entity, Fact

FACTS_PER_GROUP = 12   # default group count is current facts / this
MIN_GROUPS = 2
MAX_GROUPS = 12
MIN_GROUP_SIZE = 3     # smaller groups stay on the source, in ``keep``
KEYWORDS_PER_GROUP = 3
MOVE_ATTEMPTS = 5      # target rewrites while the source keeps changing under a move

_WORD = re.compile(r"[a-z0-9][a-z0-9'+.-]*[a-z0-9]|[a-z0-9]")
_SLUG = re.compile(r"[^a-z0-9]+")
_KEYWORD = re.compile(r"[a-z][a-z-]*[a-z]{2}")
_STOPWORDS = frozenset(
    "the and for with that this from into onto over under about after before "
    "are was were has have had not but his her its their they them him she he "
    "you your our can will would should could may might also than then when "
    "where which who what how why all any each every some via per uses used use "
    "using one two three more most less very just only been being does did doing "
    "out off yes now new old get got set like well".split()
)


log = logging.getLogger(__name__)


class EmbedderLike(Protocol):
    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


def fact_key(fact: Fact) -> tuple[str, str | None, str | None]:
    """A fact's identity for moves: the same fields its search point id covers."""
    return (fact.text, fact.valid_from, fact.superseded_at)


# --- moving facts ----------------------------------------------------------


@dataclass
class MoveResult:
    source: str
    target: str
    moved: list[str] = field(default_factory=list)      # requested texts moved now
    already: list[str] = field(default_factory=list)    # requested texts already on the target only
    unmatched: list[str] = field(default_factory=list)  # requested texts on neither
    carried: int = 0         # superseded facts that followed their replacement
    created: bool = False    # target didn't exist before
    source_facts: int = 0    # source's current facts afterwards
    target_facts: int = 0    # target's current facts afterwards


def validate_entity_id(entity_id: str) -> str:
    """Return the type of a full ``type:name`` id, or raise ``ValueError``."""
    etype, sep, name = entity_id.partition(":")
    if not sep or not etype or not name.strip():
        raise ValueError(f"{entity_id!r} is not a full entity id like 'person:alice-health'")
    return etype


def check_parent(client: QdrantClient, entity_id: str, parent: str, *, domain: str) -> None:
    """Raise ``ValueError`` unless ``parent`` is another existing entity in ``domain``."""
    validate_entity_id(parent)
    if parent == entity_id:
        raise ValueError(f"{entity_id} can't be its own parent")
    if get_entity(client, parent, domain=domain) is None:
        raise ValueError(f"parent {parent} not found in domain {domain!r}")


def select_facts(source: Entity, texts: list[str]) -> tuple[list[Fact], set[str], int]:
    """Facts of ``source`` that a move of ``texts`` takes along.

    A requested text selects the current facts with that text, never its
    history directly, so repeating a move takes nothing new. Then, until
    nothing changes, each superseded fact follows its replacement (see
    ``_replacement``) if that was selected: history goes where its
    replacement goes.

    Known gap (#26): ``superseded_by`` records the replacement text as
    stored, and a later merge can lengthen the surviving fact's wording;
    history linked by the old wording stays on the source.

    Returns the facts in source order, the requested texts that matched, and
    how many facts came via supersession only.
    """
    requested = set(texts)
    picked = {i for i, f in enumerate(source.facts) if f.is_current and f.text in requested}
    matched = {source.facts[i].text for i in picked}
    direct = len(picked)
    replaced_by = {i: _replacement(source.facts, i) for i, f in enumerate(source.facts)
                   if not f.is_current}
    while True:
        more = {i for i, r in replaced_by.items() if i not in picked and r in picked}
        if not more:
            break
        picked |= more
    facts = [f for i, f in enumerate(source.facts) if i in picked]
    return facts, matched, len(picked) - direct


def _replacement(facts: list[Fact], i: int) -> int | None:
    """Index of the fact that superseded ``facts[i]``, if it can be told.

    ``superseded_by`` is a text, and a text can recur (Linux, then macOS,
    then Linux again). The replacement is the latest occurrence of that text
    in effect (``valid_from``, else ``added``) on or before the supersession
    date; a later reassertion isn't it. None when no occurrence qualifies or
    two tie on that date: that history stays put rather than guess.
    """
    fact = facts[i]
    day = (fact.superseded_at or "")[:10]
    dated = [
        ((f.valid_from or f.added)[:10], j) for j, f in enumerate(facts)
        if j != i and f.text == fact.superseded_by and (f.valid_from or f.added)[:10] <= day
    ]
    if not dated:
        return None
    latest = max(d for d, _ in dated)
    best = [j for d, j in dated if d == latest]
    return best[0] if len(best) == 1 else None


@contextmanager
def move_lock(domain: str):
    """Serialise moves in ``domain`` across processes on this host.

    Two moves into one target would otherwise read the same snapshot of it
    and the later write would drop the earlier move's facts. Ordinary writes
    (``memory_store``) don't take this lock: like every KIPP write, a store
    racing a move is last-writer-wins on that entity.
    """
    # One fixed place, so a CLI and an MCP server started with different
    # environments still share it.
    base = Path.home() / ".cache" / "kipp"
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"move-{domain}.lock"
    with open(path, "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def move_facts(
    client: QdrantClient,
    embedder: EmbedderLike,
    source_id: str,
    target_id: str,
    texts: list[str],
    *,
    domain: str,
    now: datetime,
    apply: bool = True,
) -> MoveResult:
    """Move the facts of ``source_id`` whose text is in ``texts`` to ``target_id``.

    Texts match exactly; see ``select_facts`` for which facts a text takes.
    Each fact moves unchanged: same timestamps, provenance and hit count, no
    similarity merge on landing. A target that doesn't exist is created with
    the type from its id prefix and ``parent=source_id``; an existing target
    keeps its parent.

    Order of writes: the target first, then the source re-read and written
    without exactly the facts the target received. A crash between them
    leaves the facts on both. Running the move again finds the leftover
    copies on the target (same text, dates, source and supersession), merges
    each pair (higher hit count, longer lifetime) and finishes removing them
    from the source. If a copy was changed on either side in between, it no
    longer matches and the re-run leaves a duplicate rather than guess.

    If the re-read shows the selection changed meanwhile (a concurrent store
    superseded a selected fact, say), the copies this call wrote are taken
    back off the target and the new selection written, before the source is
    touched. Before the source write, the target is re-read: a fact a
    concurrent write knocked off it stays on the source.

    With ``apply=False`` nothing is written; the result says what would move.
    """
    validate_entity_id(source_id)
    target_type = validate_entity_id(target_id)
    if source_id == target_id:
        raise ValueError("source and target are the same entity")
    if not apply:
        return _move(client, embedder, source_id, target_id, target_type, texts,
                     domain=domain, now=now, apply=False)
    with move_lock(domain):
        return _move(client, embedder, source_id, target_id, target_type, texts,
                     domain=domain, now=now, apply=True)


def _move(
    client: QdrantClient, embedder: EmbedderLike, source_id: str, target_id: str,
    target_type: str, texts: list[str], *, domain: str, now: datetime, apply: bool,
) -> MoveResult:
    source = get_entity(client, source_id, domain=domain)
    if source is None:
        raise ValueError(f"{source_id} not found in domain {domain!r}")
    target = get_entity(client, target_id, domain=domain)
    result = MoveResult(source=source_id, target=target_id, created=target is None)
    if target is None:
        target = Entity(id=target_id, type=target_type, parent=source_id)

    facts, matched, carried = select_facts(source, texts)
    on_target = {f.text for f in target.facts}
    wanted = list(dict.fromkeys(texts))
    result.moved = [t for t in wanted if t in matched]
    result.already = [t for t in wanted if t not in matched and t in on_target]
    result.unmatched = [t for t in wanted if t not in matched and t not in on_target]
    result.carried = carried

    if not apply or not facts:
        if apply:
            # Nothing left to move, but a crash inside the last source write
            # can have left its search points stale: repair both indexes.
            sync_until_stable(client, source, embedder, domain=domain)
            if not result.created:
                sync_until_stable(client, target, embedder, domain=domain)
        _land(target, facts)
        result.source_facts = _current(source, without={fact_key(f) for f in facts})
        result.target_facts = _current(target)
        return result

    fresh: Entity | None = source
    landed: list[Fact] = []    # the selection the target was last written with
    appended: list[Fact] = []  # copies this call added to the target in that write
    for attempt in range(MOVE_ATTEMPTS):
        if attempt:
            # Re-read, so a write to the target since our last one survives,
            # and take back exactly the copies we added (a copy someone has
            # since changed stays: a duplicate, not a loss).
            target = get_entity(client, target_id, domain=domain) or Entity(
                id=target_id, type=target_type, parent=source_id,
            )
            target.facts = _without(target.facts, appended)
        appended = _land(target, facts)
        target.last_updated = now.isoformat()
        write_entity(client, target, embedder, domain=domain, now=now)
        landed = facts
        # Re-read so a write that landed since the first read isn't clobbered.
        fresh = get_entity(client, source_id, domain=domain)
        if fresh is None:
            break  # deleted meanwhile: the facts are safe on the target; don't resurrect it
        again, _, _ = select_facts(fresh, texts)
        if again == facts:
            break
        facts = again
    else:
        log.warning(
            "move %s -> %s (%s): the source changed on each of %d attempts; facts "
            "that changed last stay on the source", source_id, target_id, domain, MOVE_ATTEMPTS,
        )

    if fresh is not None:
        # Remove only facts exactly as the target received them, and only
        # those still on it. One changed since stays on both.
        safe = _still_on(get_entity(client, target_id, domain=domain), landed)
        if len(safe) < len(landed):
            log.warning(
                "move %s -> %s (%s): %d of %d facts were no longer on the target "
                "after writing it; they stay on the source", source_id, target_id,
                domain, len(landed) - len(safe), len(landed),
            )
        if safe:
            fresh.facts = _without(fresh.facts, safe)
            fresh.last_updated = now.isoformat()
            write_entity(client, fresh, embedder, domain=domain, now=now)

    result.source_facts = _current(fresh) if fresh is not None else 0
    result.target_facts = _current(target)
    return result


def _state(fact: Fact) -> tuple:
    """Every stored field of a fact, hashable."""
    return astuple(replace(fact, embedding=None))


def _without(facts: list[Fact], remove: list[Fact]) -> list[Fact]:
    """``facts`` minus one exact match per fact in ``remove``."""
    pending = Counter(_state(f) for f in remove)
    kept = []
    for f in facts:
        if pending[_state(f)] > 0:
            pending[_state(f)] -= 1
        else:
            kept.append(f)
    return kept


def _copy_key(fact: Fact) -> tuple:
    """Fields a moved copy shares with its original until either is changed."""
    return (fact.text, fact.valid_from, fact.superseded_at, fact.superseded_by,
            fact.added, fact.source)


def _still_on(target: Entity | None, facts: list[Fact]) -> list[Fact]:
    """The ``facts`` that ``target`` holds an intact copy of, one copy each.

    Intact: same identity, and at least the original's hit count, last-seen
    and lifetime, so a stale write to the target can't cost metadata.
    """
    free = list(target.facts) if target else []
    kept = []
    for f in facts:
        i = next((i for i, t in enumerate(free) if _covers(t, f)), None)
        if i is not None:
            free.pop(i)
            kept.append(f)
    return kept


def _covers(copy: Fact, original: Fact) -> bool:
    return (
        _copy_key(copy) == _copy_key(original)
        and copy.hit_count >= original.hit_count
        and (copy.last_seen or "") >= (original.last_seen or "")
        and longer_lifetime(copy.expires, original.expires) == copy.expires
    )


def _land(target: Entity, facts: list[Fact]) -> list[Fact]:
    """Add ``facts`` to ``target``, merging copies an interrupted move left.

    A target fact with the same text, dates, source and supersession is such
    a copy: the pair becomes one fact keeping the higher hit count, later
    last-seen and longer lifetime. Matching is one-to-one. Anything else is
    appended: the same text filed separately stays a separate fact (moves
    never similarity-merge). Returns the appended copies.
    """
    used: set[int] = set()
    by_copy: dict[tuple, list[int]] = defaultdict(list)
    for i, f in enumerate(target.facts):
        by_copy[_copy_key(f)].append(i)
    appended = []
    for fact in facts:
        i = next((j for j in by_copy[_copy_key(fact)] if j not in used), None)
        if i is None:
            copy = replace(fact, embedding=None)
            target.facts.append(copy)
            appended.append(copy)
            continue
        used.add(i)
        twin = target.facts[i]
        target.facts[i] = replace(
            twin,
            last_seen=max(filter(None, [twin.last_seen, fact.last_seen]), default=None),
            hit_count=max(twin.hit_count, fact.hit_count),
            expires=longer_lifetime(twin.expires, fact.expires),
            embedding=None,
        )
    return appended


def _current(entity: Entity, without: set | frozenset = frozenset()) -> int:
    return sum(1 for f in entity.facts if f.is_current and fact_key(f) not in without)


# --- planning --------------------------------------------------------------


def default_group_count(n_facts: int) -> int:
    return max(MIN_GROUPS, min(MAX_GROUPS, round(n_facts / FACTS_PER_GROUP)))


def cluster(vectors: np.ndarray, k: int) -> list[list[int]]:
    """Ward agglomerative clustering of unit-normalised ``vectors`` into ``k`` groups.

    Ward merges the pair of clusters that least increases within-cluster
    variance, which keeps groups of comparable size. Average linkage, tried
    first, chained most of an entity's facts into one group: facts about one
    entity are all fairly similar. Ties go to the lowest indices, so the
    result is deterministic. Returns sorted index lists, ordered by first member.
    """
    n = len(vectors)
    if n == 0:
        return []
    k = max(1, min(k, n))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = vectors / np.where(norms == 0, 1.0, norms)
    sq = np.sum(unit**2, axis=1)
    dist = np.maximum(sq[:, None] + sq[None, :] - 2 * unit @ unit.T, 0.0)
    np.fill_diagonal(dist, np.inf)
    members: dict[int, list[int]] = {i: [i] for i in range(n)}
    size = np.ones(n)
    while len(members) > k:
        a, b = divmod(int(np.argmin(dist)), n)
        a, b = min(a, b), max(a, b)
        na, nb = size[a], size[b]
        # Lance-Williams update for Ward on squared Euclidean distances.
        merged = ((na + size) * dist[a] + (nb + size) * dist[b] - size * dist[a, b]) / (na + nb + size)
        dist[a, :] = merged
        dist[:, a] = merged
        dist[a, a] = np.inf
        dist[b, :] = np.inf
        dist[:, b] = np.inf
        size[a] = na + nb
        members[a] = sorted(members[a] + members.pop(b))
    return sorted(members.values(), key=lambda m: m[0])


def _tokens(text: str, exclude: frozenset[str] = frozenset()) -> set[str]:
    """Candidate keywords: plain words of 3+ letters (no numbers, dates or
    versions), possessives folded, stopwords and ``exclude`` dropped."""
    words = (w.removesuffix("'s") for w in _WORD.findall(text.lower()))
    return {w for w in words if _KEYWORD.fullmatch(w) and w not in _STOPWORDS and w not in exclude}


def keywords(
    groups: list[list[str]], top: int = KEYWORDS_PER_GROUP, exclude: frozenset[str] = frozenset(),
) -> list[list[str]]:
    """Words that mark each group: share of its facts using the word, minus
    the share of all other facts using it. Ties break alphabetically."""
    token_sets = [[_tokens(t, exclude) for t in g] for g in groups]
    total: dict[str, int] = {}
    for sets in token_sets:
        for s in sets:
            for w in s:
                total[w] = total.get(w, 0) + 1
    n_all = sum(len(g) for g in groups)
    out = []
    for sets in token_sets:
        inside: dict[str, int] = {}
        for s in sets:
            for w in s:
                inside[w] = inside.get(w, 0) + 1
        n_in, n_out = len(sets), n_all - len(sets)
        scored = sorted(
            inside,
            key=lambda w: (
                -(inside[w] / n_in - ((total[w] - inside[w]) / n_out if n_out else 0.0)),
                w,
            ),
        )
        out.append([w for w in scored if inside[w] > 1][:top] or scored[:1])
    return out


def _slug(word: str) -> str:
    return _SLUG.sub("-", word.lower()).strip("-")


def suggest_targets(source_id: str, group_keywords: list[list[str]]) -> list[str]:
    """``<source>-<keyword>`` per group, unique within the plan."""
    used: set[str] = set()
    names = []
    for i, kws in enumerate(group_keywords):
        name = None
        for kw in kws:
            candidate = f"{source_id}-{_slug(kw)}"
            if _slug(kw) and candidate not in used:
                name = candidate
                break
        if name is None:
            name = f"{source_id}-topic{i + 1}"
        used.add(name)
        names.append(name)
    return names


def plan_split(
    entity: Entity, embedder: EmbedderLike, *, domain: str, now: datetime,
    groups: int | None = None,
) -> dict:
    """An editable split plan for ``entity``'s current facts.

    Superseded facts aren't listed: they follow their replacement when the
    plan is applied, or stay on the source. Groups under ``MIN_GROUP_SIZE``
    facts go to ``keep`` (they stay on the source, alongside whatever the
    operator decides is core identity).
    """
    texts = list(dict.fromkeys(f.text for f in entity.facts if f.is_current))
    k = groups if groups is not None else default_group_count(len(texts))
    vectors = np.array(embedder.embed_batch(texts)) if texts else np.zeros((0, 1))
    clusters = [[texts[i] for i in c] for c in cluster(vectors, k)]
    big = sorted((c for c in clusters if len(c) >= MIN_GROUP_SIZE), key=lambda c: -len(c))
    keep = [t for t in texts if not any(t in c for c in big)]
    # The entity's own name says nothing about a group of its facts.
    etype, _, name = entity.id.lower().partition(":")
    own = frozenset({etype, name, *_WORD.findall(name.replace("-", " "))})
    kws = keywords(big + ([keep] if keep else []), exclude=own)[: len(big)]
    targets = suggest_targets(entity.id, kws)
    return {
        "source": entity.id,
        "domain": domain,
        "created": now.isoformat(timespec="seconds"),
        "groups": [
            {"target": t, "keywords": kw, "facts": c}
            for t, kw, c in zip(targets, kws, big)
        ],
        "keep": keep,
    }


# --- applying a plan -------------------------------------------------------


def validate_plan(plan: dict) -> None:
    """Raise ``ValueError`` on a plan that can't be applied as written."""
    for key in ("source", "domain", "groups"):
        if key not in plan:
            raise ValueError(f"plan is missing {key!r}")
    validate_entity_id(plan["source"])
    seen_targets: set[str] = set()
    seen_texts: dict[str, str] = {}
    for g in plan["groups"]:
        target = g.get("target", "")
        validate_entity_id(target)
        if target == plan["source"]:
            raise ValueError(f"group target {target} is the source itself")
        if target in seen_targets:
            raise ValueError(f"target {target} appears in two groups; merge them")
        seen_targets.add(target)
        for text in g.get("facts", []):
            if text in seen_texts:
                raise ValueError(f"fact is in both {seen_texts[text]} and {target}: {text[:80]!r}")
            seen_texts[text] = target
    for text in plan.get("keep", []):
        if text in seen_texts:
            raise ValueError(f"fact is in keep and in {seen_texts[text]}: {text[:80]!r}")


def apply_plan(
    client: QdrantClient, embedder: EmbedderLike, plan: dict, *, now: datetime,
    apply: bool = False,
) -> list[MoveResult]:
    """Move each group of ``plan``, in order. Dry run unless ``apply``.

    Groups are independent moves, so an interrupted run can be re-run as is:
    finished groups report their facts as ``already`` moved.
    """
    validate_plan(plan)
    source = get_entity(client, plan["source"], domain=plan["domain"])
    if source is not None:
        targets = {g["target"]: get_entity(client, g["target"], domain=plan["domain"])
                   for g in plan["groups"]}
        check_carried(plan, source, targets)
    return [
        move_facts(
            client, embedder, plan["source"], g["target"], g.get("facts", []),
            domain=plan["domain"], now=now, apply=apply,
        )
        for g in plan["groups"]
    ]


def check_carried(plan: dict, source: Entity, targets: dict[str, Entity | None]) -> None:
    """Reject a plan that files history somewhere other than where it would go.

    History moves with its replacement. If a text with only history left on
    the source is also listed in ``keep``, or in another group whose target
    doesn't already hold it as current (a group finished on an earlier run
    does), the plan says two things: raise rather than pick one.
    """
    current = {f.text for f in source.facts if f.is_current}
    done = {
        (t, f.text) for t, e in targets.items() if e is not None
        for f in e.facts if f.is_current
    }
    listed = {t: "keep" for t in plan.get("keep", [])}
    for g in plan["groups"]:
        listed.update({t: g["target"] for t in g.get("facts", [])})
    for g in plan["groups"]:
        facts, _, _ = select_facts(source, g.get("facts", []))
        for f in facts:
            where = listed.get(f.text, g["target"])
            if f.text in current or where == g["target"] or (where, f.text) in done:
                continue
            raise ValueError(
                f"history {f.text[:80]!r} is listed in {where} but moves with "
                f"its replacement to {g['target']}; list it there or drop it",
            )
