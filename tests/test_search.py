"""Query-aware fact ordering in search results."""

import warnings
from datetime import datetime

from qdrant_client import QdrantClient

from kipp.client import ensure_collections, upsert_entity
from kipp.embedder import MockEmbedder
from kipp.models import Entity, Fact
from kipp.search import _rank_facts, search_entities

NOW = datetime(2026, 11, 19, 12, 0, 0)


def _fact(text, last_seen="2026-11-01", hit_count=1):
    return Fact(text=text, added="2026-10-01", source="t", last_seen=last_seen, hit_count=hit_count)


def test_query_overlap_ranks_first():
    e = Entity(id="person:sarbloc", type="person", facts=[
        _fact("Owns an espresso machine."),
        _fact("Notice period at current employer is one month."),
        _fact("Lives in London."),
    ])

    _rank_facts(e, "what is my notice period?", NOW)

    assert e.facts[0].text == "Notice period at current employer is one month."


def test_no_overlap_falls_back_to_frequency_and_recency():
    e = Entity(id="person:sarbloc", type="person", facts=[
        _fact("Old fact.", last_seen="2026-01-01"),
        _fact("Frequent recent fact.", last_seen="2026-11-18", hit_count=5),
        _fact("Recent fact.", last_seen="2026-11-18"),
    ])

    _rank_facts(e, "zzz unrelated", NOW)

    assert [f.text for f in e.facts] == ["Frequent recent fact.", "Recent fact.", "Old fact."]


def test_stopwords_do_not_count_as_overlap():
    e = Entity(id="person:a", type="person", facts=[
        _fact("The cat and the dog.", last_seen="2026-01-01"),
        _fact("Recent unrelated fact.", last_seen="2026-11-18"),
    ])

    _rank_facts(e, "what about the weather and the time", NOW)

    assert e.facts[0].text == "Recent unrelated fact."


def test_leading_provenance_brackets_do_not_count_as_overlap():
    e = Entity(id="person:a", type="person", facts=[
        _fact("[run_id:abc profile:personal-act] Root cause found.", last_seen="2026-11-18"),
        _fact("Prefers the personal profile for errands.", last_seen="2026-01-01"),
    ])

    _rank_facts(e, "which profile for personal errands", NOW)

    assert e.facts[0].text == "Prefers the personal profile for errands."

    _rank_facts(e, "run_id act", NOW)  # prefix-only terms: no overlap, recency wins
    assert e.facts[0].text.startswith("[run_id")


class _PaddedEmbedder(MockEmbedder):
    def embed(self, text: str) -> list[float]:
        v = super().embed(text)
        return v + [0.0] * (384 - len(v))


def test_search_returns_facts_best_first():
    client = QdrantClient(":memory:")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensure_collections(client)
    emb = _PaddedEmbedder()
    e = Entity(id="person:sarbloc", type="person", facts=[
        _fact("Owns an espresso machine."),
        _fact("Commute backup route goes via Victoria."),
    ])
    upsert_entity(client, e, emb.embed("x"))

    (hit,) = search_entities(client, "commute route", emb, limit=1)

    assert hit.entity.facts[0].text == "Commute backup route goes via Victoria."
