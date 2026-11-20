"""Garbage collection and compaction, per domain.

Shared by the ``memory expire`` / ``memory compact`` CLI commands. Each helper
works on exactly one domain so callers choose the blast radius explicitly; the
CLI loops over ``DOMAINS`` for ``--domain all`` (issue #32).
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from qdrant_client import QdrantClient
from qdrant_client.models import PointVectors

from kipp.client import collection_name, point_to_entity, scroll_entities, upsert_entity
from kipp.merge import MAX_FACTS, build_search_text, build_vector_text, compact, drop_expired


class EmbedderLike(Protocol):
    def embed(self, text: str) -> list[float]: ...


def expire_events(client: QdrantClient, domain: str, now: datetime) -> int:
    """Delete events in ``domain`` whose ``expires`` timestamp has passed.

    Returns the number of events deleted. Events with no ``expires`` are kept.
    """
    coll = collection_name(domain, "events")
    if not client.collection_exists(coll):
        return 0

    expired_ids: list = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=coll, limit=256, offset=offset, with_payload=["expires"],
        )
        for p in points:
            expires = (p.payload or {}).get("expires")
            if expires and datetime.fromisoformat(expires) < now:
                expired_ids.append(p.id)
        if offset is None:
            break

    # Delete after the scroll completes so deletions can't shift pagination.
    for i in range(0, len(expired_ids), 500):
        client.delete(collection_name=coll, points_selector=expired_ids[i : i + 500])
    return len(expired_ids)


def expire_facts(
    client: QdrantClient, embedder: EmbedderLike, domain: str, now: datetime,
) -> int:
    """Drop expired facts from every entity in ``domain``; re-embed changed ones.

    Returns the number of entities rewritten.
    """
    updated = 0
    for entity in scroll_entities(client, domain=domain):
        before = len(entity.facts)
        entity.facts = drop_expired(entity.facts, now)
        if len(entity.facts) < before:
            entity.last_updated = now.isoformat()
            upsert_entity(
                client, entity, embedder.embed(build_vector_text(entity, now=now)),
                domain=domain,
            )
            updated += 1
    return updated


def compact_domain(
    client: QdrantClient,
    embedder: EmbedderLike,
    domain: str,
    now: datetime,
    max_facts: int = MAX_FACTS,
) -> int:
    """Compact every entity in ``domain`` holding more than ``max_facts`` facts.

    Returns the number of entities compacted.
    """
    compacted = 0
    for entity in scroll_entities(client, domain=domain):
        if len(entity.facts) <= max_facts:
            continue
        entity = compact(entity, max_facts=max_facts, now=now)
        upsert_entity(
            client, entity, embedder.embed(build_vector_text(entity, now=now)),
            domain=domain,
        )
        compacted += 1
    return compacted


def reindex_domain(
    client: QdrantClient, embedder: EmbedderLike, domain: str, now: datetime,
) -> int:
    """Rebuild the vector and ``search_text`` payload of every entity in ``domain``.

    Vectors are only rebuilt when an entity is written, so a change to how they
    are built (``build_vector_text``, issue #49) reaches stored entities only
    through this pass. Facts are never rewritten: only the vector and the
    ``search_text`` payload are set, so a store that lands mid-pass keeps its
    facts. At worst that one entity keeps a vector built from its facts just
    before the store, until its next write.

    Returns the number of entities re-embedded.
    """
    count = 0
    for kind in ("entities", "decisions"):
        coll = collection_name(domain, kind)
        if not client.collection_exists(coll):
            continue
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=coll, limit=100, offset=offset, with_payload=True,
            )
            for p in points:
                entity = point_to_entity(p)
                client.update_vectors(
                    collection_name=coll,
                    points=[PointVectors(id=p.id, vector=embedder.embed(build_vector_text(entity, now=now)))],
                )
                client.set_payload(
                    collection_name=coll,
                    payload={"search_text": build_search_text(entity, now=now)},
                    points=[p.id],
                )
                count += 1
            if offset is None:
                break
    return count
