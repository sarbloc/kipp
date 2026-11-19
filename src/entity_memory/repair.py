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
from entity_memory.merge import build_search_text, find_duplicate
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
    its tag. Two phases: every outgoing fact is detached first, so an incoming
    fact can never dedup into a fact that is itself about to leave (A→B plus
    B→C would otherwise drop A's fact). Then each fact lands on its target via
    ``_land``, which dedups without treating the move as a fresh observation.
    """
    by_id = {e.id: e for e in entities}
    touched: dict[str, Entity] = {}

    leaving = {id(m.fact) for m in plan.moves}
    for move in plan.moves:
        source = by_id[move.source_id]
        source.facts = [f for f in source.facts if id(f) not in leaving]
        touched[source.id] = source

    for move in plan.moves:
        target = by_id[move.target_id]
        _land(target, replace(move.fact, text=move.body, embedding=None), embedder)
        touched[target.id] = target

    for entity in touched.values():
        entity.last_updated = now.isoformat()
        upsert_entity(
            client, entity, embedder.embed(build_search_text(entity, now=now)),
            domain=domain,
        )
    return len(touched)


def _land(target: Entity, fact: Fact, embedder: EmbedderLike) -> None:
    """Add a moved fact to ``target``, folding it into a current duplicate.

    Unlike ``merge()``, a move is not a re-observation: a duplicate absorbs the
    moved fact's hit_count and keeps the LATER of the two last_seen dates
    rather than being stamped today, so repair doesn't inflate recency for
    search and compaction.
    """
    current = [f for f in target.facts if f.is_current]
    dupe = find_duplicate(fact, current, embedder)
    if dupe is None:
        target.facts.append(fact)
        return
    dupe.hit_count += fact.hit_count
    dupe.last_seen = max(filter(None, (dupe.last_seen, fact.last_seen)), default=None)
    if fact.valid_from is not None:
        dupe.valid_from = min(fact.valid_from, dupe.valid_from or dupe.added)
    if len(fact.text) > len(dupe.text):
        dupe.text, dupe.embedding = fact.text, fact.embedding
