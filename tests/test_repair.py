"""Repair of misattributed extracted facts (issue #34)."""

import warnings
from datetime import datetime

import pytest
from click.testing import CliRunner
from qdrant_client import QdrantClient

import entity_memory.cli as cli_mod
from entity_memory.client import ensure_collections, get_entity, upsert_entity
from entity_memory.embedder import MockEmbedder
from entity_memory.extract import parse_lead_tag
from entity_memory.models import Entity, Fact
from entity_memory.repair import apply_repair, plan_repair

NOW = datetime(2026, 11, 19, 12, 0, 0)
DIMS = 384


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


def _fact(text, source="extract", **kw):
    return Fact(text=text, added="2026-11-01", source=source, last_seen="2026-11-02", **kw)


def _corpus():
    hub = Entity(id="project:hub", type="project", facts=[
        _fact("Hub's own fact.", source="mcp:store"),
        _fact("[person:alice] Notice period is one month."),
        _fact("[person:alice] Has a limited company."),
        _fact("[person:ghost] Nobody knows this person."),
        _fact("Untagged sentence that landed here."),
        _fact("[project:hub] Correctly placed tagged fact."),
        _fact("[person:alice] Old superseded fact.", superseded_at="2026-11-05"),
    ])
    alice = Entity(id="person:alice", type="person", facts=[
        _fact("Lives in London.", source="mcp:store"),
    ])
    return [hub, alice]


def test_plan_classifies_extracted_facts():
    plan = plan_repair(_corpus())

    assert [(m.source_id, m.target_id, m.body) for m in plan.moves] == [
        ("project:hub", "person:alice", "Notice period is one month."),
        ("project:hub", "person:alice", "Has a limited company."),
    ]
    assert dict(plan.missing_targets) == {"person:ghost": 1}
    assert dict(plan.untagged_extracted) == {"project:hub": 1}


def test_apply_moves_facts_and_preserves_provenance(client, emb):
    entities = _corpus()
    for e in entities:
        upsert_entity(client, e, emb.embed(e.id), domain="personal")

    rewritten = apply_repair(client, emb, "personal", entities, plan_repair(entities), NOW)

    assert rewritten == 2
    hub = get_entity(client, "project:hub", domain="personal")
    alice = get_entity(client, "person:alice", domain="personal")
    hub_texts = [f.text for f in hub.facts]
    assert "[person:alice] Notice period is one month." not in hub_texts
    assert "[person:ghost] Nobody knows this person." in hub_texts  # left in place
    assert "[person:alice] Old superseded fact." in hub_texts  # history untouched

    moved = next(f for f in alice.facts if f.text == "Notice period is one month.")
    assert (moved.source, moved.added, moved.last_seen) == ("extract", "2026-11-01", "2026-11-02")
    assert len(alice.facts) == 3


def test_apply_dedups_against_target(client, emb):
    hub = Entity(id="project:hub", type="project", facts=[_fact("[person:alice] Lives in London.")])
    alice = Entity(id="person:alice", type="person", facts=[_fact("Lives in London.", source="mcp:store")])
    entities = [hub, alice]

    apply_repair(client, emb, "dev", entities, plan_repair(entities), NOW)

    stored = get_entity(client, "person:alice", domain="dev")
    assert len(stored.facts) == 1
    assert stored.facts[0].hit_count == 2
    assert get_entity(client, "project:hub", domain="dev").facts == []


def test_cli_dry_run_writes_nothing(client, emb, monkeypatch):
    monkeypatch.setattr(cli_mod, "get_client", lambda: client)
    monkeypatch.setattr(cli_mod, "_get_embedder", lambda: emb)
    for e in _corpus():
        upsert_entity(client, e, emb.embed(e.id), domain="dev")

    res = CliRunner().invoke(cli_mod.main, ["repair", "--domain", "dev"])

    assert res.exit_code == 0, res.output
    assert "[dev] 2 facts to move, 1 with a missing target, 1 untagged extracted." in res.output
    assert "move    2  project:hub -> person:alice" in res.output
    assert "DRY-RUN" in res.output
    assert len(get_entity(client, "person:alice", domain="dev").facts) == 1

    res = CliRunner().invoke(cli_mod.main, ["repair", "--domain", "dev", "--apply"])
    assert res.exit_code == 0, res.output
    assert "applied: 2 entities rewritten" in res.output
    assert len(get_entity(client, "person:alice", domain="dev").facts) == 3


class _TagBlindEmbedder(_PaddedEmbedder):
    """Embeds the tag-stripped body, so "[x:y] Same." dedups against "Same."."""

    def embed(self, text: str) -> list[float]:
        return super().embed(parse_lead_tag(text)[1].strip())


def test_chained_moves_lose_nothing(client):
    """A→B and B→C where A's fact would dedup into B's outgoing fact (Codex P1)."""
    emb = _TagBlindEmbedder()
    a = Entity(id="project:a", type="project", facts=[_fact("[person:b] Same text here.")])
    b = Entity(id="person:b", type="person", facts=[_fact("[project:c] Same text here.")])
    c = Entity(id="project:c", type="project", facts=[_fact("C's own fact.", source="mcp:store")])
    entities = [a, b, c]

    apply_repair(client, emb, "dev", entities, plan_repair(entities), NOW)

    assert [f.text for f in get_entity(client, "person:b", domain="dev").facts] == ["Same text here."]
    assert "Same text here." in [f.text for f in get_entity(client, "project:c", domain="dev").facts]
    assert get_entity(client, "project:a", domain="dev").facts == []


@pytest.mark.parametrize(
    ("target_seen", "moved_seen", "expected"),
    [("2026-10-01", "2026-11-02", "2026-11-02"), ("2026-11-10", "2026-11-02", "2026-11-10")],
)
def test_dedup_keeps_latest_original_last_seen(client, emb, target_seen, moved_seen, expected):
    """A move into a duplicate is not a fresh observation (Codex P2)."""
    hub = Entity(id="project:hub", type="project", facts=[
        Fact(text="[person:alice] Lives in London.", added="2026-09-01",
             source="extract", last_seen=moved_seen),
    ])
    alice = Entity(id="person:alice", type="person", facts=[
        Fact(text="Lives in London.", added="2026-09-01", source="mcp:store",
             last_seen=target_seen),
    ])
    entities = [hub, alice]

    apply_repair(client, emb, "dev", entities, plan_repair(entities), NOW)

    (fact,) = get_entity(client, "person:alice", domain="dev").facts
    assert fact.last_seen == expected


def _dupe_pair(moved_kw, target_kw, target_text="Lives in London."):
    hub = Entity(id="project:hub", type="project", facts=[
        Fact(text="[person:alice] Lives in London.", source="extract", **moved_kw),
    ])
    alice = Entity(id="person:alice", type="person", facts=[
        Fact(text=target_text, source="mcp:store", **target_kw),
    ])
    return [hub, alice]


def test_dedup_uses_effective_dates(client, emb):
    """Missing last_seen/valid_from fall back to added on both sides."""
    entities = _dupe_pair(
        {"added": "2026-11-01", "last_seen": "2026-11-02"},
        {"added": "2026-11-10"},
    )

    apply_repair(client, emb, "dev", entities, plan_repair(entities), NOW)

    (fact,) = get_entity(client, "person:alice", domain="dev").facts
    assert fact.last_seen == "2026-11-10"  # not moved backwards to 11-02
    assert fact.valid_from == "2026-11-01"  # widened to the moved fact's start
    assert fact.valid_at("2026-11-05")


def test_expired_duplicate_does_not_absorb(client, emb):
    entities = _dupe_pair(
        {"added": "2026-11-01"}, {"added": "2026-11-01", "expires": "2026-11-15"},
    )

    apply_repair(client, emb, "dev", entities, plan_repair(entities), NOW)

    facts = get_entity(client, "person:alice", domain="dev").facts
    # The moved fact lands as its own permanent fact; the expired one is left
    # for `memory expire` instead of swallowing it.
    assert sorted((f.expires or "") for f in facts) == ["", "2026-11-15"]


def test_foreign_tagged_fact_does_not_absorb(client):
    """A fact tagged for a missing entity stays put and isn't a dedup target."""
    emb = _TagBlindEmbedder()
    entities = _dupe_pair(
        {"added": "2026-11-01"}, {"added": "2026-11-01"},
        target_text="[person:ghost] Lives in London.",
    )

    apply_repair(client, emb, "dev", entities, plan_repair(entities), NOW)

    texts = sorted(f.text for f in get_entity(client, "person:alice", domain="dev").facts)
    assert texts == ["Lives in London.", "[person:ghost] Lives in London."]


def test_interrupted_apply_duplicates_but_never_loses(client, emb):
    """Crash after the first write: the fact survives somewhere (Codex P1)."""
    entities = _corpus()
    for e in entities:
        upsert_entity(client, e, emb.embed(e.id), domain="dev")

    import entity_memory.repair as repair_mod

    real, calls = repair_mod.upsert_entity, {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated crash")
        return real(*args, **kwargs)

    repair_mod.upsert_entity = flaky
    try:
        with pytest.raises(RuntimeError):
            apply_repair(client, emb, "dev", entities, plan_repair(entities), NOW)
    finally:
        repair_mod.upsert_entity = real

    alice = [f.text for f in get_entity(client, "person:alice", domain="dev").facts]
    assert "Notice period is one month." in alice

    # Re-running on fresh state converges without duplicating on the target.
    fresh = [get_entity(client, eid, domain="dev") for eid in ("project:hub", "person:alice")]
    apply_repair(client, emb, "dev", fresh, plan_repair(fresh), NOW)
    alice = [f.text for f in get_entity(client, "person:alice", domain="dev").facts]
    assert alice.count("Notice period is one month.") == 1
    hub = [f.text for f in get_entity(client, "project:hub", domain="dev").facts]
    assert "[person:alice] Notice period is one month." not in hub


def test_permanent_moved_fact_keeps_dupe_alive(client, emb):
    entities = _dupe_pair(
        {"added": "2026-11-01"}, {"added": "2026-11-01", "expires": "2026-11-20"},
    )

    apply_repair(client, emb, "dev", entities, plan_repair(entities), NOW)

    (fact,) = get_entity(client, "person:alice", domain="dev").facts
    assert fact.expires is None

