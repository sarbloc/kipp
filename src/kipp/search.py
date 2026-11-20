"""Semantic search across entity collections with vector + text + filter fusion."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchText,
    MatchValue,
    SearchParams,
)

from kipp.client import collection_name, entity_point_id, point_to_entity
from kipp.facts import FACT_FETCH, search_facts
from kipp.merge import fact_score
from kipp.models import Entity


class EmbedderLike(Protocol):
    def embed(self, text: str) -> list[float]: ...


@dataclass
class SearchResult:
    entity: Entity
    score: float


def _project_temporal(entity: Entity, as_of: str | None) -> Entity:
    """Return ``entity`` with its facts projected to a temporal view (issue #21).

    Default (``as_of is None``): drop superseded facts so recall reflects current
    state. With ``as_of`` set: keep only facts in effect on that ISO date, so the
    caller sees the entity as it was then. Mutates the (throwaway) entity built
    by ``point_to_entity`` — safe because each search rebuilds entities freshly.

    Note: ranking still comes from the stored "now" vector; only the *returned
    facts* are time-shifted. Reconstructing historical ranking would need a
    historical vector, which we deliberately don't keep.
    """
    if as_of is None:
        entity.facts = [f for f in entity.facts if f.is_current]
    else:
        entity.facts = [f for f in entity.facts if f.valid_at(as_of)]
    return entity


_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "the and for with that this from what when where which who whom whose why how "
    "are was were has have had his her its our their your you not but any all can "
    "does did about into over than then them they there these those will would "
    "should could been being just also".split()
)


# Leading bracket groups are routing/provenance ("[person:x]",
# "[run_id:... profile:personal-act]"), not content: they must not score.
_LEAD_BRACKETS = re.compile(r"^\s*(?:\[[^\]]*\]\s*)+")


def _terms(text: str) -> set[str]:
    text = _LEAD_BRACKETS.sub("", text).lower()
    return {w for w in _WORD.findall(text) if len(w) >= 3 and w not in _STOPWORDS}


def _rank_facts(
    entity: Entity, query: str, now: datetime, matched: dict[str, float] | None = None,
) -> None:
    """Order ``entity.facts`` best-first for this query, in place.

    Facts are stored oldest-first, so callers that show the first fact (e.g.
    Endurance's per-turn injection) surfaced the entity's OLDEST fact rather
    than the one the query hit. Facts the per-fact vector search matched
    (``matched``: fact text -> score) come first, best score first. The rest
    follow by query-term overlap, then frequency × recency (``fact_score``):
    that lexical order is all an entity gets before its fact points exist.
    """
    q = _terms(query)
    entity.facts.sort(
        key=lambda f: (len(q & _terms(f.text)), fact_score(f, now)), reverse=True,
    )
    if matched:
        # Stable sort: unmatched facts keep the lexical order above.
        entity.facts.sort(key=lambda f: -matched.get(f.text, float("-inf")))


def search_entities(
    client: QdrantClient,
    query: str,
    embedder: EmbedderLike,
    entity_type: str | None = None,
    limit: int = 5,
    *,
    domain: str = "shared",
    as_of: str | None = None,
) -> list[SearchResult]:
    """Search across entities and decisions collections within a domain.

    Uses dense vector search as primary, with optional type filter.
    Deduplicates results across collections by entity_id, keeping the higher score.

    Each fact also has its own point (``kipp.facts``). The nearest facts are
    searched alongside the entity vectors and each entity scores the higher of
    its entity hit and its best fact hit, so one fact buried among many still
    surfaces its entity. Without a facts collection (installs not yet
    reindexed) this is entity-vector search alone.

    Each result's facts are ordered best-first for the query (``_rank_facts``)
    and projected to a temporal view: superseded facts are
    hidden by default; pass ``as_of`` (ISO date) to see each entity as it was on
    that date (issue #21). Entities with no facts valid at that date are dropped,
    and the limit is applied AFTER projection so they don't squat the top slots.
    Use ``memory_get`` for the full, unfiltered record.

    Entity vectors are current-state only (see ``_project_temporal``). With
    ``as_of`` set, the fact search includes superseded facts, so an entity can
    be found by a fact that has since been replaced, but its ranking is still
    not a full historical view (issue #25).
    """
    query_vector = embedder.embed(query)

    search_filter = None
    if entity_type:
        search_filter = Filter(
            must=[FieldCondition(key="type", match=MatchValue(value=entity_type))]
        )

    # Temporal projection can empty an entity's fact list, so we project and drop
    # empties BEFORE applying the caller's limit (issue #21). Historical (as_of)
    # queries empty more often — a fact may not be valid at a past date — so we
    # over-fetch more generously than the default "now" view. The fetch stays
    # bounded: an earlier unbounded "whole collection" fetch turned every as_of
    # search into an O(N) scan (Codex review of PR #23). as_of ranking is
    # best-effort regardless (entity vectors are current-state only — issue #25),
    # so a capped candidate window is the right tradeoff over scanning everything.
    fetch_limit = max(limit * (10 if as_of is not None else 2), limit)

    results_map: dict[str, SearchResult] = {}

    for kind in ("entities", "decisions"):
        coll = collection_name(domain, kind)
        if not client.collection_exists(coll):
            continue

        hits = client.query_points(
            collection_name=coll,
            query=query_vector,
            query_filter=search_filter,
            limit=fetch_limit,
            with_payload=True,
        )

        for hit in hits.points:
            entity = point_to_entity(hit)
            eid = entity.id
            if eid not in results_map or hit.score > results_map[eid].score:
                results_map[eid] = SearchResult(entity=entity, score=hit.score)

    fact_hits = search_facts(
        client, query_vector, domain=domain, entity_type=entity_type,
        current_only=as_of is None, limit=max(FACT_FETCH, fetch_limit * 5),
    )
    _merge_fact_hits(client, results_map, fact_hits, domain=domain)

    text_results = _text_search(client, query, search_filter, fetch_limit, domain=domain)
    for sr in text_results:
        eid = sr.entity.id
        if eid not in results_map:
            results_map[eid] = sr

    ranked = sorted(results_map.values(), key=lambda r: r.score, reverse=True)

    now = datetime.utcnow()
    projected: list[SearchResult] = []
    for r in ranked:
        _project_temporal(r.entity, as_of)
        if r.entity.facts:
            matched: dict[str, float] = {}
            for score, text in fact_hits.get(r.entity.id, []):
                matched[text] = max(score, matched.get(text, float("-inf")))
            _rank_facts(r.entity, query, now, matched)
            projected.append(r)
        if len(projected) >= limit:
            break
    return projected


def _merge_fact_hits(
    client: QdrantClient,
    results_map: dict[str, SearchResult],
    fact_hits: dict[str, list[tuple[float, str]]],
    *,
    domain: str,
) -> None:
    """Fold fact hits into ``results_map``: each entity keeps its best score.

    Entities found only through a fact are fetched from their collection. A
    fact point whose entity is gone (deleted mid-search) is skipped.
    """
    missing = {}
    for eid, hits in fact_hits.items():
        best = hits[0][0]
        if eid in results_map:
            results_map[eid].score = max(results_map[eid].score, best)
        else:
            missing[entity_point_id(eid)] = best
    if not missing:
        return
    for kind in ("entities", "decisions"):
        coll = collection_name(domain, kind)
        if not client.collection_exists(coll):
            continue
        for p in client.retrieve(coll, ids=list(missing), with_payload=True):
            entity = point_to_entity(p)
            score = missing.get(str(p.id))
            if score is not None and entity.id not in results_map:
                results_map[entity.id] = SearchResult(entity=entity, score=score)


def _text_search(
    client: QdrantClient,
    query: str,
    extra_filter: Filter | None,
    limit: int,
    *,
    domain: str = "shared",
) -> list[SearchResult]:
    """Keyword fallback: search the text index on search_text field within a domain."""
    results = []
    text_condition = FieldCondition(key="search_text", match=MatchText(text=query))

    conditions = [text_condition]
    if extra_filter and extra_filter.must:
        conditions.extend(extra_filter.must)

    scroll_filter = Filter(must=conditions)

    for kind in ("entities", "decisions"):
        coll = collection_name(domain, kind)
        if not client.collection_exists(coll):
            continue
        points, _ = client.scroll(
            collection_name=coll,
            scroll_filter=scroll_filter,
            limit=limit,
            with_payload=True,
        )
        for p in points:
            entity = point_to_entity(p)
            results.append(SearchResult(entity=entity, score=0.5))

    return results


def format_results(results: list[SearchResult]) -> str:
    """Format search results for CLI output."""
    if not results:
        return "No results found."

    lines = []
    for r in results:
        # Build a summary from top facts
        facts_summary = ". ".join(f.text for f in r.entity.facts[:3])
        lines.append(f"[{r.score:.2f}] {r.entity.id} — {facts_summary}")
    return "\n".join(lines)
