"""MCP stdio server exposing KIPP operations as tools.

Started by `memory mcp`. Talks the Model Context Protocol over stdio so an
MCP client (e.g. Endurance, Claude Code, etc.) can invoke memory operations
without shelling out to the CLI.

Tools mirror the existing CLI verbs and add an explicit `domain` parameter
for routing between the per-domain Qdrant collections defined in client.py.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Any

from mcp.server.fastmcp import FastMCP

from kipp.client import (
    DOMAINS,
    collection_stats,
    delete_entity,
    ensure_collections,
    get_client,
    get_entity,
    get_unextracted_events,
    list_children,
    mark_event_extracted,
    scroll_entities,
    store_event,
)
from kipp.extract import MATCH_THRESHOLD
from kipp.facts import write_entity
from kipp.merge import (
    compact,
    current_fact_count,
    long_fact_warning,
    mark_superseded,
    merge,
    oversized_warning,
)
from kipp.models import Entity, Fact
from kipp.pipeline import DEFAULT_BATCH_SIZE, run_extraction
from kipp.search import format_results, search_entities
from kipp.split import check_parent, move_facts

EVENT_TTL_DAYS = 30
# Facts per search hit: always the top SEARCH_FACTS_PER_HIT, then more while the
# hit's facts stay within SEARCH_FACT_WORDS. Short atomic facts get a deeper
# window; entities of long facts return the same 10 as before.
SEARCH_FACTS_PER_HIT = 10
SEARCH_FACT_WORDS = 400

mcp = FastMCP("kipp")

_embedder = None


def _embedder_instance():
    """Lazy-load the embedder so import time stays cheap."""
    global _embedder
    if _embedder is None:
        from kipp.embedder import Embedder

        _embedder = Embedder()
    return _embedder


def _entity_to_dict(entity: Entity) -> dict[str, Any]:
    return {
        "id": entity.id,
        "type": entity.type,
        "last_updated": entity.last_updated,
        "parent": entity.parent,
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
    }


def _validate_domain(domain: str) -> None:
    if domain not in DOMAINS:
        raise ValueError(f"Unknown domain {domain!r}; expected one of {DOMAINS}")


@mcp.tool()
def memory_store(
    entity_type: str,
    entity_id: str,
    content: str,
    domain: str = "shared",
    supersedes: str | None = None,
    valid_from: str | None = None,
    parent: str | None = None,
) -> dict[str, Any]:
    """Upsert an entity with a fact in the given domain.

    Args:
        entity_type: person | project | tool | preference | decision
        entity_id: short id (e.g. 'alice') or full id ('person:alice')
        content: the fact: one short, self-contained claim, key words first.
            Each fact is searched on its own; split longer or mixed-topic
            text into separate facts.
        domain: shared | dev | personal (default shared)
        supersedes: text of an existing CURRENT fact that this
            new fact replaces. When set, that fact is marked superseded (today)
            and dropped from default search/recall, but kept for as-of queries.
            Use this when reality changed (e.g. "lives in Berlin" supersedes
            "lives in London") rather than just adding a new independent fact.
        valid_from: ISO date the new fact became true in the world, if it
            differs from today (when it was recorded). Optional.
        parent: full id of the existing entity this one is a topic of, e.g.
            'person:alice' for 'person:alice-health'. Set it when creating a
            topic entity; ``memory_get`` on the parent then lists it.

    The superseded fact records ``superseded_by=content`` — the replacement text
    as the caller asserted it. That's a descriptive audit trail of intent, not a
    lookup key: if merge dedups ``content`` into an existing current fact with
    longer wording, the surviving fact's text can differ from the recorded
    ``superseded_by`` string. Nothing reads ``superseded_by`` back to resolve a
    fact, so the divergence is cosmetic; tightening it to the post-merge wording
    is deferred (issue #26).

    Returns the entity id, its current fact count, (when supersession fired)
    the superseded fact's text, a ``warning`` when the fact is over 40 words
    (re-store it as shorter facts), and an ``oversized`` note when the entity
    holds over 30 current facts (file later facts on a topic entity named
    ``<id>-<topic>``; nothing is trimmed).
    """
    _validate_domain(domain)
    client = get_client()
    ensure_collections(client)
    embedder = _embedder_instance()

    full_id = f"{entity_type}:{entity_id}" if ":" not in entity_id else entity_id
    now = datetime.utcnow()
    today = now.date().isoformat()

    # Future-effective dating (a valid_from past today) is a scheduling feature
    # beyond this system's scope: the live view assumes is_current == valid-now,
    # which only holds for backdated/same-day facts. Reject rather than half-apply
    # it (tracked in issue #24).
    if valid_from is not None and valid_from[:10] > today:
        raise ValueError(
            f"future valid_from {valid_from!r} is not supported yet (today is "
            f"{today}); future-effective dating is tracked in issue #24"
        )

    if parent is not None:
        check_parent(client, full_id, parent, domain=domain)

    existing = get_entity(client, full_id, domain=domain)
    if existing is None:
        existing = Entity(id=full_id, type=entity_type)
    if parent is not None:
        existing.parent = parent

    superseded = None
    if supersedes:
        # Record supersession before merging the replacement, so the old fact is
        # already historical and the new one merges as a current fact. The agent
        # decides WHAT supersedes WHAT; the core just records it (issue #21).
        # Close the old fact exactly when the replacement starts (its valid_from,
        # falling back to today): a backdated replacement must end the old fact at
        # the same date, or an as-of query in the gap sees both as valid.
        marked = mark_superseded(
            existing, supersedes, by=content, on_date=valid_from or today,
        )
        superseded = marked.text if marked is not None else None

    new_fact = Fact(text=content, added=today, source="mcp:store", valid_from=valid_from)
    merged = merge(existing, [new_fact], embedder, now=now)
    write_entity(client, merged, embedder, domain=domain)

    result: dict[str, Any] = {"id": full_id, "facts": current_fact_count(merged), "domain": domain}
    if supersedes:
        result["superseded"] = superseded
    warning = long_fact_warning(content)
    if warning:
        result["warning"] = warning
    # Separate key from ``warning``: agents told to re-store shorter facts on a
    # ``warning`` must not do that for an oversized entity.
    oversized = oversized_warning(merged)
    if oversized:
        result["oversized"] = oversized
    return result


@mcp.tool()
def memory_get(entity_id: str, domain: str = "shared") -> dict[str, Any] | None:
    """Retrieve a single entity by id from a domain. Returns None if missing.

    ``parent`` names the entity this one is a topic of; ``children`` lists
    its own topic entities. Facts about a topic may live on a child.
    """
    _validate_domain(domain)
    client = get_client()
    entity = get_entity(client, entity_id, domain=domain)
    if entity is None:
        return None
    return {**_entity_to_dict(entity), "children": list_children(client, entity.id, domain=domain)}


@mcp.tool()
def memory_list(
    domain: str = "shared",
    entity_type: str | None = None,
) -> list[dict[str, Any]]:
    """List entities in a domain, optionally filtered by type."""
    _validate_domain(domain)
    client = get_client()
    entities = scroll_entities(client, entity_type=entity_type, domain=domain)
    return [
        {
            "id": e.id, "type": e.type, "facts": len(e.facts),
            "last_updated": e.last_updated, "parent": e.parent,
        }
        for e in entities
    ]


@mcp.tool()
def memory_search(
    query: str,
    domain: str = "shared",
    entity_type: str | None = None,
    limit: int = 5,
    as_of: str | None = None,
) -> list[dict[str, Any]]:
    """Semantic search across entities and decisions in a domain.

    Returns hits sorted by score, each with the entity payload and score.
    Each hit carries its best-matching facts first: at least 10, then more
    while they total under about 400 words, so entities of short facts return
    more of them. When more exist, ``facts_omitted`` says how many were left
    out. Use ``memory_get`` for every fact.

    Superseded facts are hidden by default (issue #21). Pass ``as_of`` (an ISO
    date, e.g. "2025-06-01") to see each entity as it was on that date — facts
    valid then are returned, later/replaced ones are not. Use ``memory_get`` for
    the full, unfiltered record including history.

    Each fact is searched on its own as well as the entity as a whole, and an
    entity scores by whichever matches better, so a single matching fact
    surfaces its entity. With ``as_of`` set, superseded facts are searched too,
    so an entity can be found by a fact that has since been replaced.
    """
    _validate_domain(domain)
    client = get_client()
    embedder = _embedder_instance()
    results = search_entities(
        client, query, embedder, entity_type=entity_type, limit=limit,
        domain=domain, as_of=as_of,
    )
    return [_search_hit(r.score, r.entity) for r in results]


def _search_hit(score: float, entity: Entity) -> dict[str, Any]:
    """One search result, trimmed to the query's top facts.

    Entities are uncapped (#52), so a hit can carry 100+ facts; returning them
    all made search responses too large for MCP clients. ``search_entities``
    already ranks facts best-first for the query, so the head is what matters.
    """
    hit = {"score": score, **_entity_to_dict(entity)}
    keep = _facts_within_budget([f["text"] for f in hit["facts"]])
    omitted = len(hit["facts"]) - keep
    if omitted > 0:
        hit["facts"] = hit["facts"][:keep]
        hit["facts_omitted"] = omitted
    return hit


def _facts_within_budget(texts: list[str]) -> int:
    """How many leading facts a hit returns: the first SEARCH_FACTS_PER_HIT,
    then each further fact while the running total stays within
    SEARCH_FACT_WORDS."""
    words = 0
    for i, text in enumerate(texts):
        words += len(text.split())
        if i >= SEARCH_FACTS_PER_HIT and words > SEARCH_FACT_WORDS:
            return i
    return len(texts)


@mcp.tool()
def memory_move(
    source_id: str,
    target_id: str,
    facts: list[str],
    domain: str = "shared",
) -> dict[str, Any]:
    """Move facts from one entity to another in the same domain.

    Use it to split an oversized entity into topic entities, or to fix a fact
    filed on the wrong entity.

    Args:
        source_id: full id of the entity holding the facts, e.g. 'person:alice'.
        target_id: full id to move them to, e.g. 'person:alice-health'. Created
            if missing, with the type from its prefix and ``parent=source_id``.
        facts: exact texts of the facts to move, as ``memory_get`` returns them.
            Superseded facts follow the fact that replaced them.
        domain: shared | dev | personal (default shared)

    Facts move unchanged (dates, source, hit count). The target is written
    first, so a failure part-way can leave a fact on both entities but never
    on neither; calling again with the same arguments finishes the move.

    Returns ``moved`` (texts moved now), ``already`` (texts only on the target
    already), ``unmatched`` (texts on neither entity), ``left`` (texts still
    on the source because a concurrent write changed them: call again),
    ``carried`` (superseded facts moved with them), ``created`` and each
    side's current fact count.
    """
    _validate_domain(domain)
    client = get_client()
    ensure_collections(client)
    result = move_facts(
        client, _embedder_instance(), source_id, target_id, facts,
        domain=domain, now=datetime.utcnow(),
    )
    return {
        "source": result.source,
        "target": result.target,
        "domain": domain,
        "moved": result.moved,
        "already": result.already,
        "unmatched": result.unmatched,
        "left": result.left,
        "carried": result.carried,
        "created": result.created,
        "source_facts": result.source_facts,
        "target_facts": result.target_facts,
    }


@mcp.tool()
def memory_event(
    text: str,
    domain: str = "shared",
    source: str = "conversation",
    agent: str = "main",
    run_id: str | None = None,
    session_id: str | None = None,
    profile: str | None = None,
    trigger_source: str | None = None,
) -> dict[str, Any]:
    """Log a raw observation. No entity extraction happens immediately.

    Provenance fields (run_id/session_id/profile/trigger_source) are
    persisted on the event payload for later filtering.
    """
    _validate_domain(domain)
    client = get_client()
    ensure_collections(client)
    embedder = _embedder_instance()

    now = datetime.utcnow()
    event_id = str(uuid.uuid4())
    vector = embedder.embed(text)
    expires = (now + timedelta(days=EVENT_TTL_DAYS)).isoformat()

    store_event(
        client,
        event_id=event_id,
        text=text,
        vector=vector,
        timestamp=now.isoformat(),
        source=source,
        agent=agent,
        expires=expires,
        domain=domain,
        run_id=run_id,
        session_id=session_id,
        profile=profile,
        trigger_source=trigger_source,
    )
    return {"id": event_id, "domain": domain}


@mcp.tool()
def memory_extract(
    domain: str = "shared",
    since_minutes: int | None = None,
    process_all: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    threshold: float = MATCH_THRESHOLD,
    match_untagged: bool = False,
) -> dict[str, Any]:
    """Process unextracted events into entity upserts within a domain.

    Routing: an event opening with a ``[type:id]`` tag attaches to that entity
    when it exists; if it doesn't, the event stays unextracted for creation via
    memory_store. Untagged events stay unextracted too unless
    ``match_untagged=True`` opts in to cosine matching — off by default because
    it misattributed most facts in practice (issue #33). Tag your memory_event
    text with its subject so it routes deterministically.

    Events are processed in batches (batch_size); each batch upserts its
    enriched entities and marks ONLY its matched events extracted before the
    next batch, so a large backlog makes incremental, kill-safe progress.
    Fully-unmatched events stay unextracted so an external LLM agent can turn
    them into new entities via memory_store, then call memory_event_resolve.
    (Tradeoff: an event with a MIX of matched and unmatched sentences is marked
    extracted because >=1 matched, so its unmatched sentences are not separately
    surfaced. Acceptable for now.)

    ``threshold`` is the cosine gate for a sentence→fact match, used only with
    ``match_untagged`` (default: the tuned MATCH_THRESHOLD).
    """
    _validate_domain(domain)
    client = get_client()
    embedder = _embedder_instance()

    since_dt = None
    if since_minutes is not None and not process_all:
        since_dt = datetime.utcnow() - timedelta(minutes=since_minutes)

    return run_extraction(
        client, embedder, domain=domain, since=since_dt,
        batch_size=batch_size, threshold=threshold, match_untagged=match_untagged,
    )


@mcp.tool()
def memory_events_unextracted(
    domain: str = "shared", limit: int = 50, since_minutes: int | None = None,
) -> dict[str, Any]:
    """List raw events not yet extracted into entities, oldest first, within a domain.

    Use this to find events that need turning into entities (via memory_store).
    When since_minutes is set, only events newer than that many minutes ago are
    returned (e.g. since_minutes=1440 for the last 24h); otherwise the whole
    unextracted backlog is considered. Results are oldest-first either way.
    Returns {"events": [{"id": str, "text": str, "timestamp": str}], "returned": int}.
    """
    _validate_domain(domain)
    client = get_client()
    since = (
        datetime.utcnow() - timedelta(minutes=since_minutes)
        if since_minutes is not None
        else None
    )
    events = get_unextracted_events(client, since=since, domain=domain)
    events.sort(key=lambda e: e.get("timestamp", ""))
    sliced = events[:limit]
    projected = [
        {"id": e["id"], "text": e["text"], "timestamp": e.get("timestamp", "")}
        for e in sliced
    ]
    return {"events": projected, "returned": len(projected)}


@mcp.tool()
def memory_event_resolve(
    event_id: str, domain: str = "shared", created_entity_id: str | None = None,
) -> dict[str, Any]:
    """Mark a raw event as extracted/resolved, optionally recording which entity it fed.

    Call after creating/updating an entity from the event (pass created_entity_id),
    or to dismiss a non-memory-worthy event (omit created_entity_id).
    Returns {"resolved": event_id, "created_entity_id": created_entity_id}.
    """
    _validate_domain(domain)
    client = get_client()
    mark_event_extracted(
        client, event_id, domain=domain, resolved_into=created_entity_id,
    )
    return {"resolved": event_id, "created_entity_id": created_entity_id}


@mcp.tool()
def memory_embed(text: str) -> list[float]:
    """Return the MiniLM embedding (384 dims) for a string.

    Exposed so consumers (e.g. Endurance's per-turn auto-injection) can
    use the same embedder rather than spinning up their own model.
    """
    return _embedder_instance().embed(text)


@mcp.tool()
def memory_stats() -> dict[str, dict[str, Any]]:
    """Per-collection point counts for all (domain, kind) collections."""
    client = get_client()
    return collection_stats(client)


def run() -> None:
    """Entry point used by `memory mcp`."""
    mcp.run()


if __name__ == "__main__":
    run()
