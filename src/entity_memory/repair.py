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
from entity_memory.merge import build_search_text, merge
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
    its tag. The target merge dedups against existing facts, so a fact the
    target already holds bumps that fact's hit_count instead of duplicating.
    """
    by_id = {e.id: e for e in entities}
    touched: dict[str, Entity] = {}

    for move in plan.moves:
        source, target = by_id[move.source_id], by_id[move.target_id]
        source.facts = [f for f in source.facts if f is not move.fact]
        moved = replace(move.fact, text=move.body, embedding=None)
        merge(target, [moved], embedder, now=now)
        # merge() stamps a newly appended fact last_seen=today; a move is not a
        # re-observation, so restore the original recency for compaction.
        moved.last_seen = move.fact.last_seen
        touched[source.id] = source
        touched[target.id] = target

    for entity in touched.values():
        entity.last_updated = now.isoformat()
        upsert_entity(
            client, entity, embedder.embed(build_search_text(entity, now=now)),
            domain=domain,
        )
    return len(touched)
