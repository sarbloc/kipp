"""Split an oversized entity into topic entities (#52).

``plan_split`` groups an entity's current facts by topic and suggests a name
for each group. The plan is a JSON file an operator (or agent) edits: rename
targets, move facts between groups, drop groups. ``apply_plan`` then moves
each group's facts with ``move_facts``, which is also the MCP ``memory_move``
tool. Grouping is deterministic: no LLM, same input, same plan.

Moves never lose a fact. The target is written before the facts leave the
source, so a crash in between leaves a fact on both and re-running finishes
the move without doubling it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Protocol

import numpy as np
from qdrant_client import QdrantClient

from kipp.client import get_entity
from kipp.facts import write_entity
from kipp.models import Entity, Fact

FACTS_PER_GROUP = 12   # default group count is current facts / this
MIN_GROUPS = 2
MAX_GROUPS = 12
MIN_GROUP_SIZE = 3     # smaller groups stay on the source, in ``keep``
KEYWORDS_PER_GROUP = 3

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

    Every fact whose text is requested, then, until nothing changes,
    superseded facts whose ``superseded_by`` names a selected fact's text, so
    history follows the fact that replaced it. Returns the facts in source
    order, the requested texts that matched, and how many facts came via
    supersession only.
    """
    requested = set(texts)
    picked = {i for i, f in enumerate(source.facts) if f.text in requested}
    matched = {source.facts[i].text for i in picked}
    direct = len(picked)
    while True:
        names = {source.facts[i].text for i in picked}
        more = {
            i for i, f in enumerate(source.facts)
            if i not in picked and not f.is_current and f.superseded_by in names
        }
        if not more:
            break
        picked |= more
    facts = [f for i, f in enumerate(source.facts) if i in picked]
    return facts, matched, len(picked) - direct


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

    Texts match exactly. Superseded facts follow the fact that replaced them
    (see ``select_facts``). Each fact moves unchanged: same timestamps,
    provenance and hit count, no similarity merge on landing. A target that
    doesn't exist is created with the type from its id prefix and
    ``parent=source_id``; an existing target keeps its parent.

    Order of writes: the target first, then the source re-read and written
    without the moved facts. A crash between them leaves the facts on both.
    Running the move again finds them already on the target, skips the copy
    (keeping the higher hit count) and finishes removing them from the source.

    With ``apply=False`` nothing is written; the result says what would move.
    """
    validate_entity_id(source_id)
    target_type = validate_entity_id(target_id)
    if source_id == target_id:
        raise ValueError("source and target are the same entity")
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

    gone = {fact_key(f) for f in facts}
    _land(target, facts)
    if not apply or not facts:
        result.source_facts = _current(source, without=gone)
        result.target_facts = _current(target)
        return result

    target.last_updated = now.isoformat()
    write_entity(client, target, embedder, domain=domain, now=now)

    # Re-read so a write that landed since the first read isn't clobbered.
    # Deleted meanwhile: the facts are safe on the target; don't resurrect it.
    fresh = get_entity(client, source_id, domain=domain)
    if fresh is not None:
        fresh.facts = [f for f in fresh.facts if fact_key(f) not in gone]
        fresh.last_updated = now.isoformat()
        write_entity(client, fresh, embedder, domain=domain, now=now)

    result.source_facts = _current(fresh) if fresh is not None else 0
    result.target_facts = _current(target)
    return result


def _land(target: Entity, facts: list[Fact]) -> None:
    """Append ``facts`` to ``target``, skipping ones it already holds."""
    held = {fact_key(f): f for f in target.facts}
    for fact in facts:
        twin = held.get(fact_key(fact))
        if twin is None:
            copy = replace(fact, embedding=None)
            target.facts.append(copy)
            held[fact_key(copy)] = copy
            continue
        # Left by an interrupted move: keep the stronger signal of the two.
        twin.hit_count = max(twin.hit_count, fact.hit_count)
        twin.last_seen = max(filter(None, [twin.last_seen, fact.last_seen]), default=None)


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
    return [
        move_facts(
            client, embedder, plan["source"], g["target"], g.get("facts", []),
            domain=plan["domain"], now=now, apply=apply,
        )
        for g in plan["groups"]
    ]
