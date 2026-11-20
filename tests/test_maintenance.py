"""Per-domain expire/compact (issue #32), against Qdrant's in-process local mode."""

import warnings
from datetime import datetime, timedelta

import pytest
from click.testing import CliRunner
from qdrant_client import QdrantClient

import kipp.cli as cli_mod
from kipp.client import (
    DOMAINS,
    collection_name,
    ensure_collections,
    entity_point_id,
    get_entity,
    store_event,
    upsert_entity,
)
from kipp.embedder import MockEmbedder
from kipp.maintenance import compact_domain, expire_events, expire_facts, reindex_domain
from kipp.merge import build_search_text, build_vector_text
from kipp.models import Entity, Fact

NOW = datetime(2026, 11, 19, 12, 0, 0)
DIMS = 384  # collections are created at VECTOR_SIZE


class _PaddedEmbedder(MockEmbedder):
    """MockEmbedder caps at 32 dims (sha256); pad to the collection size."""

    def embed(self, text: str) -> list[float]:
        v = super().embed(text)
        return v + [0.0] * (DIMS - len(v))


@pytest.fixture
def emb():
    return _PaddedEmbedder()


@pytest.fixture
def client():
    c = QdrantClient(":memory:")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # local mode ignores payload indexes
        ensure_collections(c)
    return c


def _event(client, emb, domain, eid, expires):
    store_event(
        client, event_id=eid, text=f"event {eid}", vector=emb.embed(eid),
        timestamp=NOW.isoformat(), expires=expires, domain=domain,
    )


def _ids(client, domain, kind="events"):
    pts, _ = client.scroll(collection_name(domain, kind), limit=100)
    return {str(p.id) for p in pts}


EXPIRED = "00000000-0000-0000-0000-000000000001"
LIVE = "00000000-0000-0000-0000-000000000002"
NO_TTL = "00000000-0000-0000-0000-000000000003"


@pytest.mark.parametrize("domain", DOMAINS)
def test_expire_events_deletes_only_expired_in_that_domain(client, emb, domain):
    _event(client, emb, domain, EXPIRED, (NOW - timedelta(days=1)).isoformat())
    _event(client, emb, domain, LIVE, (NOW + timedelta(days=1)).isoformat())
    _event(client, emb, domain, NO_TTL, None)
    other = next(d for d in DOMAINS if d != domain)
    _event(client, emb, other, EXPIRED, (NOW - timedelta(days=1)).isoformat())

    assert expire_events(client, domain, NOW) == 1
    assert _ids(client, domain) == {LIVE, NO_TTL}
    assert _ids(client, other) == {EXPIRED}  # other domain untouched


def _entity(eid, n_facts, expires=None):
    return Entity(
        id=eid, type=eid.split(":")[0],
        facts=[
            Fact(text=f"fact {i} of {eid}", added="2026-11-01", source="t",
                 expires=expires if i == 0 else None)
            for i in range(n_facts)
        ],
    )


def test_expire_facts_drops_expired_facts_in_domain(client, emb):
    e = _entity("person:a", 3, expires="2026-11-18")
    upsert_entity(client, e, emb.embed("x"), domain="dev")

    assert expire_facts(client, emb, "dev", NOW) == 1
    assert len(get_entity(client, "person:a", domain="dev").facts) == 2
    assert expire_facts(client, emb, "dev", NOW) == 0  # idempotent


def test_compact_domain_trims_only_target_domain(client, emb):
    upsert_entity(client, _entity("project:big", 25), emb.embed("x"), domain="personal")
    upsert_entity(client, _entity("project:big", 25), emb.embed("x"), domain="shared")
    upsert_entity(client, _entity("project:small", 5), emb.embed("y"), domain="personal")

    assert compact_domain(client, emb, "personal", NOW, max_facts=20) == 1
    assert len(get_entity(client, "project:big", domain="personal").facts) == 20
    assert len(get_entity(client, "project:small", domain="personal").facts) == 5
    assert len(get_entity(client, "project:big", domain="shared").facts) == 25


def test_cli_expire_defaults_to_shared_and_all_covers_every_domain(client, emb, monkeypatch):
    monkeypatch.setattr(cli_mod, "get_client", lambda: client)
    monkeypatch.setattr(cli_mod, "_get_embedder", lambda: emb)
    for d in DOMAINS:
        _event(client, emb, d, EXPIRED, (NOW - timedelta(days=1)).isoformat())

    runner = CliRunner()
    res = runner.invoke(cli_mod.main, ["expire"])
    assert res.exit_code == 0, res.output
    assert "[shared] Expired 1 events" in res.output
    assert _ids(client, "dev") == {EXPIRED}

    res = runner.invoke(cli_mod.main, ["expire", "--domain", "all"])
    assert res.exit_code == 0, res.output
    assert all(_ids(client, d) == set() for d in DOMAINS)
    assert "[dev] Expired 1 events" in res.output
    assert "[personal] Expired 1 events" in res.output


def _point(client, coll, eid):
    (p,) = client.retrieve(coll, ids=[entity_point_id(eid)], with_payload=True, with_vectors=True)
    return p


def test_reindex_rebuilds_vector_and_search_text_without_touching_facts(client, emb):
    long_fact = " ".join(f"w{i}" for i in range(60))
    e = Entity(id="project:p", type="project",
               facts=[Fact(text=long_fact, added="2026-11-01", source="t")])
    d = Entity(id="decision:d", type="decision",
               facts=[Fact(text="use qdrant", added="2026-11-01", source="t")])
    upsert_entity(client, e, emb.embed("stale"), domain="dev")
    upsert_entity(client, d, emb.embed("stale"), domain="dev")
    upsert_entity(client, e, emb.embed("stale"), domain="shared")
    client.set_payload(collection_name("dev", "entities"), {"search_text": "old"},
                       points=[entity_point_id("project:p")])
    before = get_entity(client, "project:p", domain="dev").facts

    assert reindex_domain(client, emb, "dev", NOW) == 2

    p = _point(client, collection_name("dev", "entities"), "project:p")
    assert p.vector == pytest.approx(emb.embed(build_vector_text(e, now=NOW)))
    assert p.payload["search_text"] == build_search_text(e, now=NOW)
    assert get_entity(client, "project:p", domain="dev").facts == before
    dp = _point(client, collection_name("dev", "decisions"), "decision:d")
    assert dp.vector == pytest.approx(emb.embed(build_vector_text(d, now=NOW)))
    # Other domains are untouched.
    sp = _point(client, collection_name("shared", "entities"), "project:p")
    assert sp.vector == pytest.approx(emb.embed("stale"))


def test_cli_reindex_all_domains(client, emb, monkeypatch):
    monkeypatch.setattr(cli_mod, "get_client", lambda: client)
    monkeypatch.setattr(cli_mod, "_get_embedder", lambda: emb)
    for dom in DOMAINS:
        upsert_entity(client, _entity("person:a", 2), emb.embed("x"), domain=dom)
    res = CliRunner().invoke(cli_mod.main, ["reindex", "--domain", "all"])
    assert res.exit_code == 0, res.output
    assert all(f"[{dom}] Re-embedded 1 entities." in res.output for dom in DOMAINS)


def test_reindex_rebuilds_from_fresh_facts_when_a_store_lands_mid_pass(client, emb, monkeypatch):
    old = Entity(id="person:a", type="person",
                 facts=[Fact(text="lives in London", added="2026-11-01", source="t")])
    new = Entity(id="person:a", type="person",
                 facts=[Fact(text="lives in Berlin", added="2026-11-01", source="t")])
    upsert_entity(client, old, emb.embed("stale"), domain="dev")

    real_update = client.update_vectors
    calls = []

    def racing_update(*args, **kwargs):
        calls.append(1)
        real_update(*args, **kwargs)
        if len(calls) == 1:  # a memory_store lands right after our first write
            upsert_entity(client, new, emb.embed("writer"), domain="dev")

    monkeypatch.setattr(client, "update_vectors", racing_update)
    assert reindex_domain(client, emb, "dev", NOW) == 1

    p = _point(client, collection_name("dev", "entities"), "person:a")
    assert len(calls) == 2
    assert p.vector == pytest.approx(emb.embed(build_vector_text(new, now=NOW)))
    assert "Berlin" in p.payload["search_text"] and "London" not in p.payload["search_text"]
    assert [f.text for f in get_entity(client, "person:a", domain="dev").facts] == ["lives in Berlin"]


def test_cli_oversized_lists_every_domain_by_size(client, emb, monkeypatch):
    monkeypatch.setattr(cli_mod, "get_client", lambda: client)
    big = _entity("person:big", 5)
    big.facts += [Fact(text=f"old {i}", added="2026-01-01", source="t",
                       superseded_at="2026-02-01") for i in range(5)]
    upsert_entity(client, big, emb.embed("x"), domain="personal")
    upsert_entity(client, _entity("project:mid", 4), emb.embed("x"), domain="dev")
    upsert_entity(client, _entity("project:small", 3), emb.embed("x"), domain="shared")

    res = CliRunner().invoke(cli_mod.main, ["oversized", "--threshold", "3"])

    assert res.exit_code == 0, res.output
    lines = res.output.splitlines()
    assert [l.split() for l in lines] == [
        ["5", "personal", "person:big"],  # superseded history not counted
        ["4", "dev", "project:mid"],
    ]

    res = CliRunner().invoke(cli_mod.main, ["oversized", "--domain", "shared", "--threshold", "3"])
    assert "No entities over 3 current facts." in res.output
