"""Repair facts misattributed by the old untagged cosine extraction (issue #34).

Extracted facts that still carry a leading ``[type:id]`` tag name their own
subject. When that tag disagrees with the entity holding the fact, the fact
was misrouted (issue #33). ``plan_repair`` finds those; ``apply_repair`` moves
each one to its tagged entity (tag stripped, merged with dedup) when the target
exists in the same domain. Targets that don't exist are reported, not created:
entity creation stays the LLM agent's call.

Untagged extracted facts can't be repaired mechanically — there is no ground
truth for where they belong — so the plan only counts them per entity.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Protocol

from qdrant_client import QdrantClient

from entity_memory.client import upsert_entity
from entity_memory.extract import parse_lead_tag
from entity_memory.merge import build_search_text, drop_expired, find_duplicate
from entity_memory.models import Entity, Fact

EXTRACT_SOURCE = "extract"


class EmbedderLike(Protocol):
    def embed(self, text: str) -> list[float]: ...


@dataclass
class Move:
    source_id: str
    target_id: str
    fact: Fact  # the fact object as it sits on the source entity
    body: str  # fact text with the tag stripped


@dataclass
class RepairPlan:
    moves: list[Move] = field(default_factory=list)
    missing_targets: Counter = field(default_factory=Counter)  # tag -> facts
    untagged_extracted: Counter = field(default_factory=Counter)  # entity -> facts


def plan_repair(entities: list[Entity]) -> RepairPlan:
    """Classify every current extracted fact in one domain's entities."""
    plan = RepairPlan()
    by_id = {e.id: e for e in entities}
    for entity in entities:
        for fact in entity.facts:
            if fact.source != EXTRACT_SOURCE or not fact.is_current:
                continue
            tag, body = parse_lead_tag(fact.text)
            body = body.strip()
            if tag is None:
                plan.untagged_extracted[entity.id] += 1
            elif tag == entity.id or not body:
                continue
            elif tag in by_id:
                plan.moves.append(Move(entity.id, tag, fact, body))
            else:
                plan.missing_targets[tag] += 1
    return plan


def apply_repair(
    client: QdrantClient,
    embedder: EmbedderLike,
    domain: str,
    entities: list[Entity],
    plan: RepairPlan,
    now: datetime,
) -> int:
    """Execute ``plan``'s moves against ``domain``. Returns entities rewritten.

    Each moved fact keeps its provenance and timestamps; only the text loses
    its tag. In memory, every outgoing fact is detached before any lands, so
    an incoming fact can never dedup into a fact that is itself about to leave
    (A→B plus B→C would otherwise drop A's fact).

    Writes are ordered so an interruption can duplicate a fact but never lose
    one: first each target is written with its incoming facts AND its own
    still-outgoing ones, then every touched entity gets its final state. A
    crash between the phases leaves facts on both ends; re-running repair
    moves the leftover copy again and it dedups on landing. Known limitation:
    that recovery re-adds the copy's hit_count (scoring only — no fact is
    lost). Restore the pre-repair snapshot if exact counts matter.
    """
    by_id = {e.id: e for e in entities}
    leaving = {id(m.fact) for m in plan.moves}
    outgoing: dict[str, list[Fact]] = {}
    for move in plan.moves:
        outgoing.setdefault(move.source_id, []).append(move.fact)
    for eid in outgoing:
        by_id[eid].facts = [f for f in by_id[eid].facts if id(f) not in leaving]

    targets: dict[str, Entity] = {}
    for move in plan.moves:
        target = by_id[move.target_id]
        _land(target, replace(move.fact, text=move.body, embedding=None), embedder, now)
        targets[target.id] = target

    def write(entity: Entity) -> None:
        entity.last_updated = now.isoformat()
        upsert_entity(
            client, entity, embedder.embed(build_search_text(entity, now=now)),
            domain=domain,
        )

    for target in targets.values():
        write(replace(target, facts=target.facts + outgoing.get(target.id, [])))
    touched = {**targets, **{eid: by_id[eid] for eid in outgoing}}
    for entity in touched.values():
        write(entity)
    return len(touched)


def _land(target: Entity, fact: Fact, embedder: EmbedderLike, now: datetime) -> None:
    """Add a moved fact to ``target``, folding it into a current duplicate.

    Unlike ``merge()``, a move is not a re-observation: a duplicate absorbs the
    moved fact's hit_count and keeps the later effective observation date
    (``last_seen or added``) and the earlier effective validity start
    (``valid_from or added``), so repair neither inflates recency nor narrows
    as-of history; it also keeps the longer lifetime (permanent beats any
    ``expires``). Expired facts and facts tagged for a different entity
    (left in place as misattributed) are never dedup candidates.
    """
    candidates = [
        f for f in drop_expired(target.facts, now)
        if f.is_current and parse_lead_tag(f.text)[0] in (None, target.id)
    ]
    dupe = find_duplicate(fact, candidates, embedder)
    if dupe is None:
        target.facts.append(fact)
        return
    dupe.hit_count += fact.hit_count
    # The merged fact lives as long as the longer-lived copy: permanent wins.
    dupe.expires = None if None in (dupe.expires, fact.expires) else max(dupe.expires, fact.expires)
    dupe.last_seen = max(fact.last_seen or fact.added, dupe.last_seen or dupe.added)
    start = min(fact.valid_from or fact.added, dupe.valid_from or dupe.added)
    if start < (dupe.valid_from or dupe.added):
        dupe.valid_from = start
    if len(fact.text) > len(dupe.text):
        dupe.text, dupe.embedding = fact.text, fact.embedding
