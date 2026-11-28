"""Qdrant client wrapper: connection, collection setup, and CRUD helpers.

Collections are organised along two axes:
- **Domain** — `shared`, `dev`, or `personal`. Endurance writes per-domain;
  OpenClaw and existing callers default to `shared`.
- **Kind** — `entities`, `events`, or `decisions`. The original storage shape.

`shared` is the legacy default and uses the original collection names
(`entities`, `events`, `decisions`) for backward compatibility. Other
domains use `<domain>_<kind>` (e.g. `dev_entities`, `personal_events`).

All public CRUD helpers accept an optional `domain="shared"` kwarg, so
existing call sites continue to work unchanged.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    TextIndexParams,
    TokenizerType,
    VectorParams,
)

# Config resolution and the domain list live in kipp.config, which imports
# only the standard library so the capture hook can load it without Qdrant.
# Re-exported here for existing importers.
from kipp.config import (  # noqa: F401
    CONFIG_ENV_VARS,
    DOMAINS,
    _config_search_paths,
    _xdg_config_home,
    load_config,
)
from kipp.models import Entity, Event, Fact

COLLECTION_KINDS = ["entities", "events", "decisions", "facts"]
VECTOR_SIZE = 384


def collection_name(domain: str, kind: str) -> str:
    """Resolve a (domain, kind) pair to a Qdrant collection name.

    `shared` keeps the legacy bare names so OpenClaw data isn't moved.
    Other domains are namespaced with `<domain>_<kind>`.
    """
    if domain not in DOMAINS:
        raise ValueError(f"Unknown domain {domain!r}; expected one of {DOMAINS}")
    if kind not in COLLECTION_KINDS:
        raise ValueError(f"Unknown kind {kind!r}; expected one of {COLLECTION_KINDS}")
    if domain == "shared":
        return kind
    return f"{domain}_{kind}"


ALL_COLLECTIONS = [collection_name(d, k) for d in DOMAINS for k in COLLECTION_KINDS]

# Back-compat alias for callers that imported the legacy 3-collection list.
COLLECTIONS = [collection_name("shared", k) for k in COLLECTION_KINDS]


def qdrant_connection(config: dict | None = None) -> tuple[str, str | None]:
    """Resolve the Qdrant (url, api_key) every client in this package uses."""
    config = config or load_config()
    qdrant_cfg = config.get("qdrant", {})
    url = qdrant_cfg.get("url", "http://127.0.0.1:6333")
    api_key_env = qdrant_cfg.get("api_key_env", "QDRANT_API_KEY")
    return url, os.environ.get(api_key_env)


def get_client(config: dict | None = None, timeout: int | None = None) -> QdrantClient:
    """Create a QdrantClient from config. ``timeout=None`` keeps the 5s default."""
    url, api_key = qdrant_connection(config)
    return QdrantClient(url=url, api_key=api_key, timeout=timeout)


def ensure_collections(client: QdrantClient) -> list[str]:
    """Create all (domain, kind) collections + indexes if missing.

    Returns the list of collection names that were freshly created.
    """
    created = []
    for domain in DOMAINS:
        for kind in COLLECTION_KINDS:
            name = collection_name(domain, kind)
            if client.collection_exists(name):
                continue
            if kind == "facts":
                ensure_fact_collection(client, domain)
            else:
                _create_record_collection(client, name)
            created.append(name)
    return created


def _create_record_collection(client: QdrantClient, name: str) -> None:
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    client.create_payload_index(
        collection_name=name,
        field_name="search_text",
        field_schema=TextIndexParams(
            type="text",
            tokenizer=TokenizerType.WORD,
            min_token_len=2,
        ),
    )
    client.create_payload_index(name, "type", field_schema=PayloadSchemaType.KEYWORD)
    client.create_payload_index(name, "entity_id", field_schema=PayloadSchemaType.KEYWORD)


def ensure_fact_collection(client: QdrantClient, domain: str, size: int = VECTOR_SIZE) -> str:
    """Create the domain's per-fact search collection and its indexes if missing."""
    coll = collection_name(domain, "facts")
    if not client.collection_exists(coll):
        try:
            client.create_collection(
                collection_name=coll,
                vectors_config=VectorParams(size=size, distance=Distance.COSINE),
            )
        except Exception:
            # Another process created it first (both saw it missing): fine.
            if not client.collection_exists(coll):
                raise
        # Index creation is idempotent, so a racing creator repeating it is harmless.
        client.create_payload_index(coll, "entity_id", field_schema=PayloadSchemaType.KEYWORD)
        client.create_payload_index(coll, "type", field_schema=PayloadSchemaType.KEYWORD)
        client.create_payload_index(coll, "current", field_schema=PayloadSchemaType.BOOL)
    return coll


def entity_point_id(entity_id: str) -> str:
    """Deterministic UUID from entity_id for Qdrant point ID."""
    h = hashlib.sha256(entity_id.encode()).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def entity_to_point(entity: Entity, vector: list[float]) -> PointStruct:
    """Convert an Entity to a Qdrant PointStruct."""
    from kipp.merge import build_search_text

    payload = {
        "entity_id": entity.id,
        "type": entity.type,
        "search_text": build_search_text(entity),
        "facts": [
            {
                "text": f.text,
                "added": f.added,
                "source": f.source,
                "expires": f.expires,
                "last_seen": f.last_seen,
                "hit_count": f.hit_count,
                "valid_from": f.valid_from,
                "superseded_at": f.superseded_at,
                "superseded_by": f.superseded_by,
            }
            for f in entity.facts
        ],
        "last_updated": entity.last_updated,
        "parent": entity.parent,
    }
    return PointStruct(
        id=entity_point_id(entity.id),
        vector=vector,
        payload=payload,
    )


def point_to_entity(point) -> Entity:
    """Convert a Qdrant point (with payload) back to an Entity."""
    p = point.payload
    facts = [
        Fact(
            text=f["text"],
            added=f["added"],
            source=f["source"],
            expires=f.get("expires"),
            last_seen=f.get("last_seen"),
            hit_count=f.get("hit_count", 1),
            # Bi-temporal fields (issue #21); .get() so pre-existing payloads
            # without these keys load as None — no migration needed.
            valid_from=f.get("valid_from"),
            superseded_at=f.get("superseded_at"),
            superseded_by=f.get("superseded_by"),
        )
        for f in p.get("facts", [])
    ]
    return Entity(
        id=p["entity_id"],
        type=p["type"],
        facts=facts,
        last_updated=p.get("last_updated", ""),
        parent=p.get("parent"),
    )


def get_entity(
    client: QdrantClient, entity_id: str, *, domain: str = "shared",
) -> Optional[Entity]:
    """Retrieve a single entity by ID from the given domain. Returns None if missing.

    Decisions and entities live in different collections; this checks both within
    the same domain.
    """
    point_id = entity_point_id(entity_id)
    primary = collection_name(domain, "decisions" if entity_id.startswith("decision:") else "entities")
    fallback = collection_name(domain, "entities" if primary.endswith("decisions") else "decisions")

    if client.collection_exists(primary):
        points = client.retrieve(collection_name=primary, ids=[point_id], with_payload=True)
        if points:
            return point_to_entity(points[0])
    if client.collection_exists(fallback):
        points = client.retrieve(collection_name=fallback, ids=[point_id], with_payload=True)
        if points:
            return point_to_entity(points[0])
    return None


def upsert_entity(
    client: QdrantClient, entity: Entity, vector: list[float], *, domain: str = "shared",
) -> None:
    """Upsert an entity into the appropriate (domain, kind) collection.

    Low-level primitive: it trusts the entity's facts to be already valid-time
    consistent (no future ``valid_from`` or ``superseded_at``). That invariant is
    enforced at the public write boundaries (``memory_store`` / CLI ``store`` /
    ``import``), not here, so the default ``is_current`` view stays equal to
    "valid now" (#24).
    """
    kind = "decisions" if entity.type == "decision" else "entities"
    point = entity_to_point(entity, vector)
    client.upsert(collection_name=collection_name(domain, kind), points=[point])


def delete_entity(
    client: QdrantClient, entity_id: str, *, domain: str = "shared",
) -> bool:
    """Delete an entity by ID from the given domain. Returns True if found and deleted."""
    point_id = entity_point_id(entity_id)
    for kind in ("entities", "decisions"):
        coll = collection_name(domain, kind)
        if not client.collection_exists(coll):
            continue
        points = client.retrieve(collection_name=coll, ids=[point_id], with_payload=False)
        if points:
            # Fact points go first. If a store re-creates the entity mid-delete,
            # the worst case is orphaned fact points (skipped by search, pruned
            # by reindex), never a live entity stripped of its index.
            _delete_fact_points(client, entity_id, domain)
            client.delete(collection_name=coll, points_selector=[point_id])
            return True
    return False


def _delete_fact_points(client: QdrantClient, entity_id: str, domain: str) -> None:
    coll = collection_name(domain, "facts")
    if client.collection_exists(coll):
        client.delete(
            collection_name=coll,
            points_selector=FilterSelector(filter=Filter(
                must=[FieldCondition(key="entity_id", match=MatchValue(value=entity_id))],
            )),
        )


def scroll_entities(
    client: QdrantClient, entity_type: str | None = None, *, domain: str = "shared",
) -> list[Entity]:
    """Scroll through entities + decisions in the given domain, optionally filtered by type."""
    results: list[Entity] = []
    for kind in ("entities", "decisions"):
        coll = collection_name(domain, kind)
        if not client.collection_exists(coll):
            continue
        scroll_filter = None
        if entity_type:
            scroll_filter = Filter(
                must=[FieldCondition(key="type", match=MatchValue(value=entity_type))]
            )
        offset = None
        while True:
            points, next_offset = client.scroll(
                collection_name=coll,
                scroll_filter=scroll_filter,
                limit=100,
                offset=offset,
                with_payload=True,
            )
            for p in points:
                results.append(point_to_entity(p))
            if next_offset is None:
                break
            offset = next_offset
    return results


def list_children(
    client: QdrantClient, parent_id: str, *, domain: str = "shared",
) -> list[str]:
    """Ids of the entities in ``domain`` whose ``parent`` is ``parent_id``, sorted.

    The filter runs without a payload index; at hundreds of entities per
    domain that's a cheap full scan.
    """
    children: list[str] = []
    parent_filter = Filter(must=[FieldCondition(key="parent", match=MatchValue(value=parent_id))])
    for kind in ("entities", "decisions"):
        coll = collection_name(domain, kind)
        if not client.collection_exists(coll):
            continue
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=coll, scroll_filter=parent_filter, limit=100,
                offset=offset, with_payload=["entity_id"],
            )
            children.extend(p.payload["entity_id"] for p in points)
            if offset is None:
                break
    return sorted(children)


def store_event(
    client: QdrantClient, event_id: str, text: str, vector: list[float],
    timestamp: str, source: str = "conversation", agent: str = "main",
    expires: str | None = None,
    *,
    domain: str = "shared",
    run_id: str | None = None,
    session_id: str | None = None,
    profile: str | None = None,
    trigger_source: str | None = None,
) -> None:
    """Store a raw event in the events collection of the given domain.

    Provenance fields (run_id/session_id/profile/trigger_source) are
    optional; legacy CLI callers leave them None and the payload omits
    them on read for older rows.
    """
    payload = {
        "text": text,
        "timestamp": timestamp,
        "source": source,
        "agent": agent,
        "extracted": False,
        "expires": expires,
    }
    if run_id is not None:
        payload["run_id"] = run_id
    if session_id is not None:
        payload["session_id"] = session_id
    if profile is not None:
        payload["profile"] = profile
    if trigger_source is not None:
        payload["trigger_source"] = trigger_source

    point = PointStruct(id=event_id, vector=vector, payload=payload)
    client.upsert(collection_name=collection_name(domain, "events"), points=[point])


def get_unextracted_events(
    client: QdrantClient, since: datetime | None = None, *, domain: str = "shared",
) -> list[dict]:
    """Get events where extracted=false, optionally filtered by time, in a domain."""
    coll = collection_name(domain, "events")
    if not client.collection_exists(coll):
        return []
    conditions = [FieldCondition(key="extracted", match=MatchValue(value=False))]
    scroll_filter = Filter(must=conditions)
    results: list[dict] = []
    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=coll,
            scroll_filter=scroll_filter,
            limit=100,
            offset=offset,
            with_payload=True,
        )
        for p in points:
            payload = p.payload
            if since is not None:
                ts = datetime.fromisoformat(payload["timestamp"])
                if ts < since:
                    continue
            results.append({"id": p.id, **payload})
        if next_offset is None:
            break
        offset = next_offset
    return results


def mark_event_extracted(
    client: QdrantClient, event_id: str, *, domain: str = "shared",
    resolved_into: str | None = None,
) -> None:
    """Mark an event as extracted within its domain.

    If ``resolved_into`` is given, also records which entity the event fed
    (provenance) on the event payload, so a resolved event points at the
    entity created/updated from it.
    """
    payload: dict = {"extracted": True}
    if resolved_into is not None:
        payload["resolved_into"] = resolved_into
    client.set_payload(
        collection_name=collection_name(domain, "events"),
        payload=payload,
        points=[event_id],
    )


def collection_stats(client: QdrantClient) -> dict[str, dict]:
    """Get point counts and metadata for every (domain, kind) collection."""
    stats: dict[str, dict] = {}
    for name in ALL_COLLECTIONS:
        if client.collection_exists(name):
            info = client.get_collection(name)
            stats[name] = {"points": info.points_count}
        else:
            stats[name] = {"points": 0, "exists": False}
    return stats
