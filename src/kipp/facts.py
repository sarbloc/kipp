"""Per-fact search points (Phase 2).

Every fact gets its own point in its domain's ``facts`` collection, with a
vector built from ``"[type] id: fact"``. The entity point stays the record of
truth: fact points are an index derived from it, rebuilt on every write
(``write_entity``) and by ``kipp reindex``.

Point ids are derived from the fact's identity, so a write embeds only facts
the index hasn't seen and deletes points for facts the entity no longer holds.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from typing import Protocol

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointIdsList,
    PointStruct,
)

from kipp.client import (
    collection_name,
    ensure_fact_collection,
    entity_point_id,
    get_entity,
    upsert_entity,
)
from kipp.merge import build_vector_text
from kipp.models import Entity, Fact

FACT_FETCH = 50      # entities fetched per fact search (grouped, so many-fact entities can't crowd others out)
FACTS_PER_ENTITY = 10  # matched facts kept per entity, for ordering its facts
SYNC_ATTEMPTS = 5
_UPSERT_CHUNK = 100

log = logging.getLogger(__name__)


class EmbedderLike(Protocol):
    def embed(self, text: str) -> list[float]: ...


def fact_point_id(entity_id: str, fact: Fact) -> str:
    """Deterministic point id for one fact of one entity.

    Covers every field the point's vector or payload is built from, so an
    unchanged id means an unchanged point. Superseding a fact changes its id:
    the old point is deleted and a flagged one written.
    """
    key = "\x00".join([entity_id, fact.text, fact.valid_from or "", fact.superseded_at or ""])
    h = hashlib.sha256(key.encode()).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def fact_vector_text(entity: Entity, fact: Fact) -> str:
    """Text embedded for one fact. The entity prefix gives short facts context."""
    return f"[{entity.type}] {entity.id}: {fact.text}"


def _payload(entity: Entity, fact: Fact) -> dict:
    return {
        "entity_id": entity.id,
        "type": entity.type,
        "text": fact.text,
        "valid_from": fact.valid_from,
        "superseded_at": fact.superseded_at,
        "current": fact.is_current,
    }


def _entity_filter(entity_id: str) -> Filter:
    return Filter(must=[FieldCondition(key="entity_id", match=MatchValue(value=entity_id))])


def _indexed_ids(client: QdrantClient, coll: str, entity_id: str) -> set[str]:
    ids: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=coll, scroll_filter=_entity_filter(entity_id),
            limit=1000, offset=offset, with_payload=False,
        )
        ids.update(str(p.id) for p in points)
        if offset is None:
            return ids


def sync_fact_points(
    client: QdrantClient, entity: Entity, embedder: EmbedderLike, *, domain: str,
    rebuild: bool = False,
) -> None:
    """Make the domain's fact points for ``entity`` match its facts exactly.

    Only facts without a point are embedded, unless ``rebuild`` re-embeds them
    all (reindex: the embedder or vector text may have changed). New points
    are written before stale ones are deleted, so a concurrent search never
    sees the entity with no fact points mid-sync.
    """
    coll = collection_name(domain, "facts")
    exists = client.collection_exists(coll)
    wanted = {fact_point_id(entity.id, f): f for f in entity.facts}
    indexed = _indexed_ids(client, coll, entity.id) if exists else set()

    points = [
        PointStruct(id=pid, vector=embedder.embed(fact_vector_text(entity, f)), payload=_payload(entity, f))
        for pid, f in wanted.items()
        if rebuild or pid not in indexed
    ]
    if points and not exists:
        # Created on first write, sized to the embedder actually in use.
        ensure_fact_collection(client, domain, size=len(points[0].vector))
    for i in range(0, len(points), _UPSERT_CHUNK):
        client.upsert(collection_name=coll, points=points[i : i + _UPSERT_CHUNK])

    stale = sorted(indexed - wanted.keys())
    if stale:
        client.delete(collection_name=coll, points_selector=PointIdsList(points=stale))


def write_entity(
    client: QdrantClient,
    entity: Entity,
    embedder: EmbedderLike,
    *,
    domain: str = "shared",
    now: datetime | None = None,
) -> None:
    """Persist ``entity``: its point (vector from ``build_vector_text``) and its fact points.

    Every write path goes through here so the fact index can't drift from the
    entity. ``upsert_entity`` is the low-level primitive underneath.

    The entity point is written before its fact points. Two writers racing on
    one entity could leave the index following the loser's facts, so after
    syncing, the stored entity is re-read and the index re-synced to it until
    they agree. A writer that lands after our last re-read syncs after it, so
    the index converges unless this loop runs out first, which is logged:
    the entity itself is stored, and ``kipp reindex`` repairs the index.
    """
    upsert_entity(client, entity, embedder.embed(build_vector_text(entity, now=now)), domain=domain)
    _sync_until_stable(client, entity, embedder, domain=domain)


def _sync_until_stable(
    client: QdrantClient, entity: Entity, embedder: EmbedderLike, *, domain: str,
) -> None:
    """Sync ``entity``'s fact points, then re-read and re-sync until the stored
    entity matches what was synced. Never writes the entity itself."""
    for _ in range(SYNC_ATTEMPTS):
        sync_fact_points(client, entity, embedder, domain=domain)
        stored = get_entity(client, entity.id, domain=domain)
        if stored is None or _fact_ids(stored) == _fact_ids(entity):
            return
        entity = stored
    log.warning(
        "fact sync: %s/%s changed on each of %d attempts; its fact points may be "
        "stale until its next write or kipp reindex --domain %s",
        domain, entity.id, SYNC_ATTEMPTS, domain,
    )


def _fact_ids(entity: Entity) -> set[str]:
    return {fact_point_id(entity.id, f) for f in entity.facts}


def prune_orphan_fact_points(
    client: QdrantClient, domain: str, embedder: EmbedderLike,
) -> int:
    """Delete fact points whose entity no longer exists. Returns entities pruned.

    Covers deletes that bypassed ``delete_entity`` (raw Qdrant edits, restores
    of an older entities snapshot) and delete races. An entity re-created
    between the existence check and the delete would lose its points, so it
    is checked again afterwards and, if present, re-synced.
    """
    coll = collection_name(domain, "facts")
    if not client.collection_exists(coll):
        return 0
    entity_ids: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=coll, limit=1000, offset=offset, with_payload=["entity_id"],
        )
        entity_ids.update(p.payload["entity_id"] for p in points)
        if offset is None:
            break
    pruned = 0
    for eid in sorted(entity_ids):
        if _entity_exists(client, eid, domain):
            continue
        client.delete(collection_name=coll, points_selector=FilterSelector(filter=_entity_filter(eid)))
        pruned += 1
        revived = get_entity(client, eid, domain=domain)
        if revived is not None:
            _sync_until_stable(client, revived, embedder, domain=domain)
    return pruned


def _entity_exists(client: QdrantClient, entity_id: str, domain: str) -> bool:
    pid = entity_point_id(entity_id)
    for kind in ("entities", "decisions"):
        coll = collection_name(domain, kind)
        if client.collection_exists(coll) and client.retrieve(coll, ids=[pid], with_payload=False):
            return True
    return False


def search_facts(
    client: QdrantClient,
    query_vector: list[float],
    *,
    domain: str,
    entity_type: str | None = None,
    current_only: bool = True,
    limit: int = FACT_FETCH,
) -> dict[str, list[tuple[float, str]]]:
    """Nearest facts to ``query_vector``, grouped by entity id.

    Returns up to ``limit`` entities as ``{entity_id: [(score, fact_text), ...]}``,
    in order of their best fact, each list best first (at most
    ``FACTS_PER_ENTITY``). Grouping happens in Qdrant, so a few entities with
    many matching facts can't use up the candidate window. Empty if the domain
    has no facts collection yet (pre-reindex installs fall back to entity
    vectors alone).
    """
    coll = collection_name(domain, "facts")
    if not client.collection_exists(coll):
        return {}
    must = []
    if current_only:
        must.append(FieldCondition(key="current", match=MatchValue(value=True)))
    if entity_type:
        must.append(FieldCondition(key="type", match=MatchValue(value=entity_type)))
    result = client.query_points_groups(
        collection_name=coll,
        group_by="entity_id",
        query=query_vector,
        query_filter=Filter(must=must) if must else None,
        limit=limit,
        group_size=FACTS_PER_ENTITY,
        with_payload=["text"],
    )
    return {
        str(g.id): [(h.score, h.payload["text"]) for h in g.hits]
        for g in result.groups
    }
