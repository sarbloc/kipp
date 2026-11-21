"""Per-fact search points (Phase 2), against Qdrant's in-process local mode."""

import math
import warnings
from datetime import datetime

import pytest
from qdrant_client import QdrantClient

import kipp.facts as facts_mod
from kipp.client import (
    collection_name,
    delete_entity,
    ensure_collections,
    get_entity,
    upsert_entity,
)
from kipp.embedder import MockEmbedder
from kipp.facts import (
    FACTS_PER_ENTITY,
    fact_point_id,
    fact_vector_text,
    prune_orphan_fact_points,
    search_facts,
    sync_fact_points,
    write_entity,
)
from kipp.maintenance import reindex_domain
from kipp.models import Entity, Fact
from kipp.search import search_entities

NOW = datetime(2026, 11, 19, 12, 0, 0)
DIMS = 384


def _unit(i: int) -> list[float]:
    v = [0.0] * DIMS
    v[i] = 1.0
    return v


class _Embedder(MockEmbedder):
    """Hash vectors padded to 384 dims, with exact vectors for chosen texts.

    Counts embed calls so tests can assert unchanged facts aren't re-embedded.
    """

    def __init__(self, table: dict[str, list[float]] | None = None):
        super().__init__()
        self.table = table or {}
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        if text in self.table:
            return self.table[text]
        v = super().embed(text)
        return v + [0.0] * (DIMS - len(v))


@pytest.fixture
def client():
    c = QdrantClient(":memory:")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # local mode ignores payload indexes
        ensure_collections(c)
    return c


def _fact(text, **kw):
    return Fact(text=text, added="2026-10-01", source="t", last_seen="2026-11-01", **kw)


def _points(client, domain="shared"):
    pts, _ = client.scroll(collection_name(domain, "facts"), limit=1000, with_payload=True)
    return {str(p.id): p.payload for p in pts}


def _entity(eid="person:alice", facts=("Lives in London.", "Owns a cat.")):
    return Entity(id=eid, type=eid.split(":")[0], facts=[_fact(t) for t in facts])


# --- sync ---------------------------------------------------------------------


def test_sync_writes_one_point_per_fact(client):
    emb = _Embedder()
    e = _entity()

    sync_fact_points(client, e, emb, domain="shared")

    pts = _points(client)
    assert set(pts) == {fact_point_id(e.id, f) for f in e.facts}
    p = pts[fact_point_id(e.id, e.facts[0])]
    assert p == {
        "entity_id": "person:alice", "type": "person", "text": "Lives in London.",
        "valid_from": None, "superseded_at": None, "current": True,
    }
    assert emb.calls == [fact_vector_text(e, f) for f in e.facts]
    assert emb.calls[0] == "[person] person:alice: Lives in London."


def test_resync_embeds_only_new_facts_and_deletes_dropped_ones(client):
    emb = _Embedder()
    e = _entity()
    sync_fact_points(client, e, emb, domain="shared")
    emb.calls.clear()

    sync_fact_points(client, e, emb, domain="shared")
    assert emb.calls == []  # unchanged: nothing embedded

    e.facts = [e.facts[0], _fact("Works at Acme.")]
    sync_fact_points(client, e, emb, domain="shared")

    assert emb.calls == ["[person] person:alice: Works at Acme."]
    assert {p["text"] for p in _points(client).values()} == {"Lives in London.", "Works at Acme."}


def test_superseding_a_fact_reflags_its_point(client):
    emb = _Embedder()
    e = _entity()
    sync_fact_points(client, e, emb, domain="shared")
    old_id = fact_point_id(e.id, e.facts[0])

    e.facts[0].superseded_at = "2026-11-10"
    sync_fact_points(client, e, emb, domain="shared")

    pts = _points(client)
    assert old_id not in pts
    flagged = pts[fact_point_id(e.id, e.facts[0])]
    assert flagged["current"] is False and flagged["superseded_at"] == "2026-11-10"


def test_sync_is_scoped_to_one_entity_and_domain(client):
    emb = _Embedder()
    alice, bob = _entity(), _entity("person:bob", ("Plays chess.",))
    sync_fact_points(client, alice, emb, domain="shared")
    sync_fact_points(client, bob, emb, domain="shared")
    sync_fact_points(client, alice, emb, domain="dev")

    alice.facts = []
    sync_fact_points(client, alice, emb, domain="shared")

    assert {p["entity_id"] for p in _points(client).values()} == {"person:bob"}
    assert len(_points(client, "dev")) == 2


def test_sync_creates_missing_collection_at_embedder_size():
    c = QdrantClient(":memory:")  # no ensure_collections: pre-Phase-2 install
    emb = MockEmbedder()  # unpadded: not the 384-dim default
    dim = len(emb.embed("probe"))

    sync_fact_points(c, _entity(), emb, domain="dev")

    info = c.get_collection(collection_name("dev", "facts"))
    assert info.config.params.vectors.size == dim


def test_sync_with_no_facts_and_no_collection_creates_nothing():
    c = QdrantClient(":memory:")
    sync_fact_points(c, _entity(facts=()), _Embedder(), domain="shared")
    assert not c.collection_exists(collection_name("shared", "facts"))


# --- write_entity / delete / prune --------------------------------------------


def test_write_entity_writes_entity_and_fact_points(client):
    emb = _Embedder()
    e = _entity()

    write_entity(client, e, emb, domain="personal")

    assert get_entity(client, e.id, domain="personal").facts[0].text == "Lives in London."
    assert len(_points(client, "personal")) == 2
    assert _points(client, "shared") == {}


def test_write_entity_resyncs_when_another_writer_lands(client, monkeypatch):
    """A writer racing between our entity upsert and our sync must not leave
    the index following our (now stale) facts."""
    emb = _Embedder()
    ours = _entity(facts=("Lives in London.",))
    theirs = _entity(facts=("Lives in London.", "Moved to Leeds."))
    real_sync, calls = facts_mod.sync_fact_points, {"n": 0}

    def racing_sync(client, entity, embedder, *, domain):
        calls["n"] += 1
        if calls["n"] == 1:  # the other writer lands first, entity and points
            upsert_entity(client, theirs, emb.embed("x"), domain=domain)
            real_sync(client, theirs, embedder, domain=domain)
        real_sync(client, entity, embedder, domain=domain)

    monkeypatch.setattr(facts_mod, "sync_fact_points", racing_sync)
    write_entity(client, ours, emb)

    assert {p["text"] for p in _points(client).values()} == {"Lives in London.", "Moved to Leeds."}


def test_delete_entity_removes_its_fact_points(client):
    emb = _Embedder()
    write_entity(client, _entity(), emb)
    write_entity(client, _entity("person:bob", ("Plays chess.",)), emb)

    assert delete_entity(client, "person:alice")

    assert {p["entity_id"] for p in _points(client).values()} == {"person:bob"}


def test_prune_removes_only_orphaned_fact_points(client):
    emb = _Embedder()
    write_entity(client, _entity(), emb)
    write_entity(client, _entity("decision:use-qdrant", ("Qdrant over pgvector.",)), emb)
    sync_fact_points(client, _entity("person:ghost", ("Gone.",)), emb, domain="shared")

    assert prune_orphan_fact_points(client, "shared", emb) == 1

    assert {p["entity_id"] for p in _points(client).values()} == {"person:alice", "decision:use-qdrant"}


# --- search_facts -------------------------------------------------------------


def test_search_facts_skips_superseded_unless_asked(client):
    emb = _Embedder()
    e = _entity()
    e.facts[0].superseded_at = "2026-11-10"
    write_entity(client, e, emb)
    q = emb.embed(fact_vector_text(e, e.facts[0]))

    current = search_facts(client, q, domain="shared")
    everything = search_facts(client, q, domain="shared", current_only=False)

    assert [t for _, t in current["person:alice"]] == ["Owns a cat."]
    assert everything["person:alice"][0][1] == "Lives in London."


def test_search_facts_filters_by_type_and_groups_best_first(client):
    emb = _Embedder()
    write_entity(client, _entity(), emb)
    write_entity(client, _entity("project:hub", ("Hub ships Friday.",)), emb)

    hits = search_facts(client, emb.embed("anything"), domain="shared", entity_type="project")

    assert list(hits) == ["project:hub"]
    alice = search_facts(client, emb.embed("anything"), domain="shared")["person:alice"]
    assert [s for s, _ in alice] == sorted((s for s, _ in alice), reverse=True)


def test_search_facts_without_collection_is_empty():
    assert search_facts(QdrantClient(":memory:"), _unit(0), domain="shared") == {}


# --- search_entities fusion ---------------------------------------------------

QUERY = "quokka habitat"


def _seed_buried_fact(client):
    """Alice's entity vector is orthogonal to the query, but one of her facts
    matches it exactly. Three decoys' entity vectors sit at 0.6 cosine."""
    alice = _entity(facts=("Owns a cat.", "Saw quokkas on Rottnest Island."))
    target = fact_vector_text(alice, alice.facts[1])
    emb = _Embedder({QUERY: _unit(0), target: _unit(0)})
    write_entity(client, alice, emb)
    upsert_entity(client, alice, _unit(1))  # re-pin entity vector: orthogonal
    for i in range(3):
        decoy = _entity(f"person:decoy{i}", (f"Decoy fact {i}.",))
        write_entity(client, decoy, emb)
        v = [0.0] * DIMS
        v[0], v[2 + i] = 0.6, 0.8
        upsert_entity(client, decoy, v)
    return emb


def test_search_finds_entity_through_a_buried_fact(client):
    emb = _seed_buried_fact(client)

    (hit,) = search_entities(client, QUERY, emb, limit=1)

    assert hit.entity.id == "person:alice"
    assert hit.score == pytest.approx(1.0)
    assert hit.entity.facts[0].text == "Saw quokkas on Rottnest Island."


def test_search_keeps_the_higher_of_entity_and_fact_score(client):
    emb = _seed_buried_fact(client)

    hits = {h.entity.id: h.score for h in search_entities(client, QUERY, emb, limit=4)}

    assert hits["person:alice"] == pytest.approx(1.0)  # fact beat entity (0.0)
    assert hits["person:decoy0"] == pytest.approx(0.6)  # entity beat its fact


def test_search_falls_back_to_entity_vectors_without_facts_collection(client):
    emb = _seed_buried_fact(client)
    client.delete_collection(collection_name("shared", "facts"))

    hits = search_entities(client, QUERY, emb, limit=1)

    assert hits[0].entity.id.startswith("person:decoy")
    assert hits[0].score == pytest.approx(0.6)


def test_matched_facts_lead_over_lexical_order(client):
    e = _entity(facts=("Commute route notes: unrelated.", "Victoria line is the backup."))
    target = fact_vector_text(e, e.facts[1])
    emb = _Embedder({"commute route": _unit(0), target: _unit(0)})
    write_entity(client, e, emb)

    (hit,) = search_entities(client, "commute route", emb, limit=1)

    # Lexical overlap alone would put the "Commute route" fact first.
    assert [f.text for f in hit.entity.facts] == [
        "Victoria line is the backup.", "Commute route notes: unrelated.",
    ]


def test_search_skips_fact_points_whose_entity_is_gone(client):
    emb = _Embedder({QUERY: _unit(0), "[person] person:ghost: Gone.": _unit(0)})
    sync_fact_points(client, _entity("person:ghost", ("Gone.",)), emb, domain="shared")
    write_entity(client, _entity(), emb)

    ids = [h.entity.id for h in search_entities(client, QUERY, emb, limit=5)]

    assert ids == ["person:alice"]


def test_as_of_search_finds_entity_by_superseded_fact(client):
    e = _entity(facts=("Lives in London.", "Owns a cat."))
    e.facts[0].valid_from, e.facts[0].superseded_at = "2025-01-01", "2026-06-01"
    target = fact_vector_text(e, e.facts[0])
    emb = _Embedder({"where did she live": _unit(0), target: _unit(0)})
    write_entity(client, e, emb)
    upsert_entity(client, e, _unit(1))

    (hit,) = search_entities(client, "where did she live", emb, limit=1, as_of="2026-01-01")

    assert hit.score == pytest.approx(1.0)
    assert hit.entity.facts[0].text == "Lives in London."


# --- reindex ------------------------------------------------------------------


def test_reindex_backfills_fact_points_and_prunes_orphans(client):
    emb = _Embedder()
    legacy = _entity()
    upsert_entity(client, legacy, emb.embed("x"), domain="dev")  # pre-Phase-2 write
    sync_fact_points(client, _entity("person:ghost", ("Gone.",)), emb, domain="dev")

    reindex_domain(client, emb, "dev", NOW)

    pts = _points(client, "dev")
    assert set(pts) == {fact_point_id(legacy.id, f) for f in legacy.facts}



def test_prune_revive_rereads_until_stable(client, monkeypatch):
    """A store landing between the revive's read and its sync must win."""
    emb = _Embedder()
    ghost = _entity("person:ghost", ("Old.",))
    newer = _entity("person:ghost", ("Old.", "Newer."))
    sync_fact_points(client, ghost, emb, domain="shared")
    real_delete, real_sync = client.delete, facts_mod.sync_fact_points
    state = {"recreated": False, "raced": False}

    def delete_then_recreate(*args, **kwargs):
        real_delete(*args, **kwargs)
        if not state["recreated"]:
            state["recreated"] = True
            upsert_entity(client, ghost, emb.embed("x"))

    def sync_racing(client_, entity, embedder, *, domain, rebuild=False):
        if not state["raced"]:  # the newer store lands and finishes first
            state["raced"] = True
            upsert_entity(client, newer, emb.embed("x"))
            real_sync(client_, newer, embedder, domain=domain)
        real_sync(client_, entity, embedder, domain=domain, rebuild=rebuild)

    monkeypatch.setattr(client, "delete", delete_then_recreate)
    monkeypatch.setattr(facts_mod, "sync_fact_points", sync_racing)
    prune_orphan_fact_points(client, "shared", emb)

    assert {p["text"] for p in _points(client).values()} == {"Old.", "Newer."}


def test_prune_resyncs_an_entity_recreated_mid_prune(client, monkeypatch):
    emb = _Embedder()
    ghost = _entity("person:ghost", ("Back again.",))
    sync_fact_points(client, ghost, emb, domain="shared")
    real_delete = client.delete

    def delete_then_recreate(*args, **kwargs):
        real_delete(*args, **kwargs)
        upsert_entity(client, ghost, emb.embed("x"))  # a store lands right after

    monkeypatch.setattr(client, "delete", delete_then_recreate)
    prune_orphan_fact_points(client, "shared", emb)

    assert {p["text"] for p in _points(client).values()} == {"Back again."}


def test_write_entity_logs_when_sync_never_settles(client, monkeypatch, caplog):
    emb = _Embedder()
    n = {"i": 0}

    def ever_changing(client, entity_id, *, domain):
        n["i"] += 1
        return _entity(facts=(f"Version {n['i']}.",))

    monkeypatch.setattr(facts_mod, "get_entity", ever_changing)
    with caplog.at_level("WARNING", logger="kipp.facts"):
        write_entity(client, _entity(), emb)

    assert "may be stale" in caplog.text


def test_reindex_reembeds_existing_fact_points(client):
    e = _entity(facts=("Owns a cat.",))
    write_entity(client, e, _Embedder(), domain="dev")
    pid = fact_point_id(e.id, e.facts[0])
    new_vec = _unit(7)
    target = fact_vector_text(e, e.facts[0])

    reindex_domain(client, _Embedder({target: new_vec}), "dev", NOW)

    (p,) = client.retrieve(collection_name("dev", "facts"), ids=[pid], with_vectors=True)
    assert p.vector == pytest.approx(new_vec)


def test_ensure_fact_collection_tolerates_a_racing_creator(monkeypatch):
    from kipp.client import ensure_fact_collection

    c = QdrantClient(":memory:")
    real_exists = c.collection_exists
    seen = {"n": 0}

    def stale_exists(name):  # first check misses a collection another process just made
        seen["n"] += 1
        if seen["n"] == 1:
            ensure_fact_collection_real(name)
            return False
        return real_exists(name)

    def ensure_fact_collection_real(name):
        from qdrant_client.models import Distance, VectorParams
        c.create_collection(name, vectors_config=VectorParams(size=DIMS, distance=Distance.COSINE))

    monkeypatch.setattr(c, "collection_exists", stale_exists)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert ensure_fact_collection(c, "dev") == "dev_facts"
    assert real_exists("dev_facts")


def test_search_facts_is_not_crowded_out_by_many_fact_entities(client):
    """Three entities with more perfect fact matches than FACTS_PER_ENTITY must
    not use up the window: a fourth entity's single good fact still comes back."""
    q = _unit(0)
    table = {}
    for i in range(3):
        e = _entity(f"person:noisy{i}", tuple(f"Noisy {i} fact {j}." for j in range(FACTS_PER_ENTITY + 10)))
        for f in e.facts:
            table[fact_vector_text(e, f)] = q
        write_entity(client, e, _Embedder(table))
    quiet = _entity("person:quiet", ("The one relevant fact.",))
    near = [0.0] * DIMS
    near[0], near[1] = 0.9, math.sqrt(1 - 0.81)
    table[fact_vector_text(quiet, quiet.facts[0])] = near
    write_entity(client, quiet, _Embedder(table))

    hits = search_facts(client, q, domain="shared", limit=4)

    assert list(hits)[-1] == "person:quiet"
    assert all(len(v) <= FACTS_PER_ENTITY for v in hits.values())
    assert len(hits["person:noisy0"]) == FACTS_PER_ENTITY


def test_keyword_fallback_score_survives_a_weaker_fact_hit(client):
    """An entity found only by keyword keeps its 0.5 when its best fact is weaker."""
    q = _unit(0)
    target = _entity("project:zephyr", ("Zephyr rollout notes.",))
    weak = [0.0] * DIMS
    weak[0], weak[1] = 0.1, math.sqrt(1 - 0.01)
    emb = _Embedder({"zephyr": q, fact_vector_text(target, target.facts[0]): weak})
    write_entity(client, target, emb)
    upsert_entity(client, target, _unit(1))  # entity vector misses the query
    for i in range(3):
        decoy = _entity(f"person:decoy{i}", (f"Decoy fact {i}.",))
        write_entity(client, decoy, emb)
        v = [0.0] * DIMS
        v[0], v[2 + i] = 0.4, math.sqrt(1 - 0.16)
        upsert_entity(client, decoy, v)

    (hit,) = search_entities(client, "zephyr", emb, limit=1)

    assert hit.entity.id == "project:zephyr"
    assert hit.score == pytest.approx(0.5)
