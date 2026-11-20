"""Splitting oversized entities into topic entities (#52)."""

import io
import json
import warnings
from datetime import datetime

import numpy as np
import pytest
from click.testing import CliRunner
from qdrant_client import QdrantClient

import kipp.cli as cli_mod
import kipp.split as split_mod
from kipp.client import collection_name, delete_entity, ensure_collections, get_entity, list_children
from kipp.embedder import MockEmbedder
from kipp.export import export_json, import_json
from kipp.facts import sync_fact_points, write_entity
from kipp.merge import mark_superseded
from kipp.models import Entity, Fact
from kipp.split import (
    apply_plan,
    cluster,
    default_group_count,
    keywords,
    move_facts,
    plan_split,
    suggest_targets,
    validate_plan,
)

NOW = datetime(2026, 11, 20, 12, 0, 0)
DIMS = 384


class _Embedder(MockEmbedder):
    """Hash vectors padded to 384 dims, with exact vectors for chosen texts."""

    def __init__(self, table=None):
        super().__init__()
        self.table = table or {}

    def embed(self, text):
        if text in self.table:
            return self.table[text]
        v = super().embed(text)
        return v + [0.0] * (DIMS - len(v))


@pytest.fixture
def emb():
    return _Embedder()


@pytest.fixture
def client():
    c = QdrantClient(":memory:")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # local mode ignores payload indexes
        ensure_collections(c)
    return c


def _fact(text, **kw):
    kw.setdefault("added", "2026-10-01")
    kw.setdefault("source", "mcp:store")
    return Fact(text=text, **kw)


def _seed(client, emb, entity, domain="personal"):
    write_entity(client, entity, emb, domain=domain, now=NOW)


def facts_mod_sync(client, entity, emb):
    sync_fact_points(client, entity, emb, domain="personal")


def _texts(entity):
    return [f.text for f in entity.facts]


def _fact_point_texts(client, domain, entity_id):
    points, _ = client.scroll(
        collection_name(domain, "facts"), limit=1000, with_payload=True,
    )
    return sorted(p.payload["text"] for p in points if p.payload["entity_id"] == entity_id)


def _alice():
    return Entity(id="person:alice", type="person", facts=[
        _fact("Lives in London."),
        _fact("Runs 5k three times a week.", hit_count=4, last_seen="2026-11-01",
              valid_from="2026-01-01", expires="2027-01-01"),
        _fact("Allergic to penicillin."),
        _fact("Works at Initech."),
    ])


# --- move_facts --------------------------------------------------------------


def test_move_creates_target_with_parent_and_keeps_fact_fields(client, emb):
    _seed(client, emb, _alice())

    r = move_facts(
        client, emb, "person:alice", "person:alice-health",
        ["Runs 5k three times a week.", "Allergic to penicillin."],
        domain="personal", now=NOW,
    )

    assert r.created and r.moved == ["Runs 5k three times a week.", "Allergic to penicillin."]
    assert (r.source_facts, r.target_facts, r.unmatched, r.already) == (2, 2, [], [])
    target = get_entity(client, "person:alice-health", domain="personal")
    assert (target.type, target.parent) == ("person", "person:alice")
    run = target.facts[0]
    assert (run.hit_count, run.last_seen, run.valid_from, run.expires, run.added, run.source) == (
        4, "2026-11-01", "2026-01-01", "2027-01-01", "2026-10-01", "mcp:store",
    )
    source = get_entity(client, "person:alice", domain="personal")
    assert _texts(source) == ["Lives in London.", "Works at Initech."]
    assert source.parent is None
    # The fact index follows both writes.
    assert _fact_point_texts(client, "personal", "person:alice") == ["Lives in London.", "Works at Initech."]
    assert _fact_point_texts(client, "personal", "person:alice-health") == [
        "Allergic to penicillin.", "Runs 5k three times a week.",
    ]


def test_superseded_history_follows_its_replacement(client, emb):
    alice = _alice()
    alice.facts += [
        _fact("Runs 3k twice a week.", superseded_at="2026-06-01", superseded_by="Runs 4k weekly."),
        _fact("Runs 4k weekly.", superseded_at="2026-09-01", superseded_by="Runs 5k three times a week."),
        _fact("Lived in Leeds.", superseded_at="2025-01-01", superseded_by="Lives in London."),
    ]
    _seed(client, emb, alice)

    r = move_facts(client, emb, "person:alice", "person:alice-health",
                   ["Runs 5k three times a week."], domain="personal", now=NOW)

    assert r.carried == 2
    target = get_entity(client, "person:alice-health", domain="personal")
    assert _texts(target) == ["Runs 5k three times a week.", "Runs 3k twice a week.", "Runs 4k weekly."]
    assert "Lived in Leeds." in _texts(get_entity(client, "person:alice", domain="personal"))


def test_rerun_reports_already_moved_and_changes_nothing(client, emb):
    _seed(client, emb, _alice())
    args = (client, emb, "person:alice", "person:alice-health", ["Allergic to penicillin.", "Owns a boat."])
    move_facts(*args, domain="personal", now=NOW)
    before = get_entity(client, "person:alice-health", domain="personal")

    r = move_facts(*args, domain="personal", now=NOW)

    assert (r.moved, r.already, r.unmatched, r.created) == (
        [], ["Allergic to penicillin."], ["Owns a boat."], False,
    )
    assert get_entity(client, "person:alice-health", domain="personal") == before


def test_crash_after_target_write_loses_nothing_and_rerun_finishes(client, emb, monkeypatch):
    _seed(client, emb, _alice())
    real = split_mod.write_entity
    calls = []

    def crash_on_source(client_, entity, *a, **kw):
        calls.append(entity.id)
        if entity.id == "person:alice":
            raise RuntimeError("killed")
        real(client_, entity, *a, **kw)

    monkeypatch.setattr(split_mod, "write_entity", crash_on_source)
    with pytest.raises(RuntimeError):
        move_facts(client, emb, "person:alice", "person:alice-health",
                   ["Runs 5k three times a week."], domain="personal", now=NOW)
    assert calls == ["person:alice-health", "person:alice"]
    # On both ends, nowhere lost.
    assert "Runs 5k three times a week." in _texts(get_entity(client, "person:alice", domain="personal"))
    assert "Runs 5k three times a week." in _texts(get_entity(client, "person:alice-health", domain="personal"))

    # The source copy picked up a hit meanwhile.
    source = get_entity(client, "person:alice", domain="personal")
    source.facts[1].hit_count = 9
    _seed(client, emb, source)
    monkeypatch.setattr(split_mod, "write_entity", real)
    r = move_facts(client, emb, "person:alice", "person:alice-health",
                   ["Runs 5k three times a week."], domain="personal", now=NOW)

    assert r.moved == ["Runs 5k three times a week."]
    target = get_entity(client, "person:alice-health", domain="personal")
    assert _texts(target) == ["Runs 5k three times a week."]
    assert target.facts[0].hit_count == 9
    assert "Runs 5k three times a week." not in _texts(get_entity(client, "person:alice", domain="personal"))


def test_landing_on_a_leftover_copy_keeps_the_stronger_fields(client, emb):
    _seed(client, emb, Entity(id="person:alice", type="person", facts=[
        _fact("Allergic to penicillin.", added="2025-01-01", hit_count=2, last_seen="2026-01-01"),
    ]))
    _seed(client, emb, Entity(id="person:alice-health", type="person", facts=[
        _fact("Allergic to penicillin.", added="2025-01-01", hit_count=5, last_seen="2025-06-01",
              expires="2026-12-01"),
    ]))

    move_facts(client, emb, "person:alice", "person:alice-health", ["Allergic to penicillin."],
               domain="personal", now=NOW)

    [fact] = get_entity(client, "person:alice-health", domain="personal").facts
    assert (fact.added, fact.last_seen, fact.hit_count, fact.expires) == (
        "2025-01-01", "2026-01-01", 5, None,
    )


def test_same_text_filed_separately_on_the_target_stays_separate(client, emb):
    _seed(client, emb, Entity(id="person:alice", type="person", facts=[
        _fact("Allergic to penicillin.", added="2025-01-01"),
    ]))
    _seed(client, emb, Entity(id="person:alice-health", type="person", facts=[
        _fact("Allergic to penicillin.", added="2026-10-01", expires="2026-12-01"),
    ]))

    move_facts(client, emb, "person:alice", "person:alice-health", ["Allergic to penicillin."],
               domain="personal", now=NOW)

    target = get_entity(client, "person:alice-health", domain="personal")
    assert [(f.added, f.expires) for f in target.facts] == [("2026-10-01", "2026-12-01"), ("2025-01-01", None)]


def test_histories_ending_on_different_dates_stay_separate(client, emb):
    old = dict(added="2020-01-01", superseded_by="Lives in Leeds.")
    _seed(client, emb, Entity(id="person:alice", type="person", facts=[
        _fact("Lives in Leeds."), _fact("Lives in York.", superseded_at="2026-09-01", **old),
    ]))
    _seed(client, emb, Entity(id="person:alice-home", type="person", facts=[
        _fact("Lives in York.", superseded_at="2026-06-01", **old),
    ]))

    move_facts(client, emb, "person:alice", "person:alice-home", ["Lives in Leeds."], domain="personal", now=NOW)

    target = get_entity(client, "person:alice-home", domain="personal")
    assert sorted(f.superseded_at or "" for f in target.facts) == ["", "2026-06-01", "2026-09-01"]


def test_fact_superseded_mid_move_lands_superseded(client, emb, monkeypatch):
    _seed(client, emb, Entity(id="person:alice", type="person", facts=[_fact("Works at Initech.")]))
    real = split_mod.write_entity
    raced = []

    def racing_supersede(client_, entity, *a, **kw):
        real(client_, entity, *a, **kw)
        if entity.id == "person:alice-work" and not raced:
            raced.append(True)
            fresh = get_entity(client, "person:alice", domain="personal")
            mark_superseded(fresh, "Works at Initech.", by="Works at Acme.", on_date="2026-11-20")
            fresh.facts.append(_fact("Works at Acme."))
            real(client_, fresh, emb, domain="personal", now=NOW)

    monkeypatch.setattr(split_mod, "write_entity", racing_supersede)
    move_facts(client, emb, "person:alice", "person:alice-work", ["Works at Initech."],
               domain="personal", now=NOW)

    [moved] = get_entity(client, "person:alice-work", domain="personal").facts
    assert (moved.text, moved.superseded_at) == ("Works at Initech.", "2026-11-20")
    assert _texts(get_entity(client, "person:alice", domain="personal")) == ["Works at Acme."]
    points, _ = client.scroll(collection_name("personal", "facts"), limit=100, with_payload=True)
    current = {(p.payload["entity_id"], p.payload["text"]) for p in points if p.payload["current"]}
    assert current == {("person:alice", "Works at Acme.")}


def test_move_gives_up_after_bounded_attempts_without_losing(client, emb, monkeypatch, caplog):
    _seed(client, emb, Entity(id="person:alice", type="person", facts=[_fact("Counter.", hit_count=1)]))
    real = split_mod.write_entity

    def always_bump(client_, entity, *a, **kw):
        real(client_, entity, *a, **kw)
        if entity.id == "person:alice-x":
            fresh = get_entity(client, "person:alice", domain="personal")
            fresh.facts[0].hit_count += 1
            real(client_, fresh, emb, domain="personal", now=NOW)

    monkeypatch.setattr(split_mod, "write_entity", always_bump)
    move_facts(client, emb, "person:alice", "person:alice-x", ["Counter."], domain="personal", now=NOW)

    assert "changed on each of 5 attempts" in caplog.text
    # The last bump never reached the target, so that copy stays on the source.
    [left] = get_entity(client, "person:alice", domain="personal").facts
    [landed] = get_entity(client, "person:alice-x", domain="personal").facts
    assert (left.text, left.hit_count, landed.hit_count) == ("Counter.", 6, 5)


def test_history_arriving_after_the_last_target_write_stays_on_source(client, emb, monkeypatch, caplog):
    _seed(client, emb, Entity(id="person:alice", type="person", facts=[_fact("Lives in York.")]))
    real = split_mod.write_entity
    added = []

    def add_history(client_, entity, *a, **kw):
        real(client_, entity, *a, **kw)
        if entity.id == "person:alice-home":
            fresh = get_entity(client, "person:alice", domain="personal")
            added.append(f"Lived in place {len(added)}.")
            fresh.facts.append(_fact(added[-1], superseded_at="2026-01-01", superseded_by="Lives in York."))
            real(client_, fresh, emb, domain="personal", now=NOW)

    monkeypatch.setattr(split_mod, "write_entity", add_history)
    move_facts(client, emb, "person:alice", "person:alice-home", ["Lives in York."], domain="personal", now=NOW)

    assert "changed on each of 5 attempts" in caplog.text
    source = _texts(get_entity(client, "person:alice", domain="personal"))
    target = _texts(get_entity(client, "person:alice-home", domain="personal"))
    assert source == [added[-1]]  # read after the last target write: never landed, so kept
    assert sorted(source + target) == sorted(["Lives in York.", *added])


def test_retry_keeps_a_write_to_the_target(client, emb, monkeypatch):
    _seed(client, emb, Entity(id="person:alice", type="person", facts=[_fact("Works at Initech.")]))
    real = split_mod.write_entity
    raced = []

    def race_both(client_, entity, *a, **kw):
        real(client_, entity, *a, **kw)
        if entity.id == "person:alice-work" and not raced:
            raced.append(True)
            target = get_entity(client, "person:alice-work", domain="personal")
            target.facts.append(_fact("Desk on floor 3."))
            real(client_, target, emb, domain="personal", now=NOW)
            source = get_entity(client, "person:alice", domain="personal")
            source.facts[0].hit_count = 7  # changes the selection: forces a retry
            real(client_, source, emb, domain="personal", now=NOW)

    monkeypatch.setattr(split_mod, "write_entity", race_both)
    move_facts(client, emb, "person:alice", "person:alice-work", ["Works at Initech."],
               domain="personal", now=NOW)

    target = get_entity(client, "person:alice-work", domain="personal")
    assert sorted(_texts(target)) == ["Desk on floor 3.", "Works at Initech."]
    assert next(f for f in target.facts if f.text == "Works at Initech.").hit_count == 7


def test_history_does_not_retire_a_separate_assertion_on_the_target(client, emb):
    _seed(client, emb, Entity(id="person:alice", type="person", facts=[
        _fact("Lives in Leeds.", added="2026-03-01"),
        _fact("Lives in York.", added="2020-01-01", superseded_at="2026-03-01", superseded_by="Lives in Leeds."),
    ]))
    _seed(client, emb, Entity(id="person:alice-home", type="person", facts=[
        _fact("Lives in York.", added="2026-11-01", source="cli:store"),
    ]))

    move_facts(client, emb, "person:alice", "person:alice-home", ["Lives in Leeds."], domain="personal", now=NOW)

    target = get_entity(client, "person:alice-home", domain="personal")
    assert [(f.text, f.added, f.is_current) for f in target.facts] == [
        ("Lives in York.", "2026-11-01", True),
        ("Lives in Leeds.", "2026-03-01", True),
        ("Lives in York.", "2020-01-01", False),
    ]


def test_resuming_with_three_identical_copies_is_idempotent(client, emb, monkeypatch):
    _seed(client, emb, Entity(id="person:alice", type="person", facts=[_fact("Said hello.")] * 3))
    real = split_mod.write_entity

    def crash_on_source(client_, entity, *a, **kw):
        if entity.id == "person:alice":
            raise RuntimeError("killed")
        real(client_, entity, *a, **kw)

    monkeypatch.setattr(split_mod, "write_entity", crash_on_source)
    with pytest.raises(RuntimeError):
        move_facts(client, emb, "person:alice", "person:alice-x", ["Said hello."], domain="personal", now=NOW)
    monkeypatch.setattr(split_mod, "write_entity", real)
    move_facts(client, emb, "person:alice", "person:alice-x", ["Said hello."], domain="personal", now=NOW)

    assert len(get_entity(client, "person:alice-x", domain="personal").facts) == 3
    assert get_entity(client, "person:alice", domain="personal").facts == []


def test_each_target_fact_absorbs_one_incoming_copy(client, emb):
    _seed(client, emb, Entity(id="person:alice", type="person", facts=[_fact("Said hello."), _fact("Said hello.")]))
    _seed(client, emb, Entity(id="person:alice-x", type="person", facts=[_fact("Said hello.")]))
    move_facts(client, emb, "person:alice", "person:alice-x", ["Said hello."], domain="personal", now=NOW)
    assert len(get_entity(client, "person:alice-x", domain="personal").facts) == 2


def test_existing_target_keeps_parent_and_facts_are_not_merged(client, emb):
    _seed(client, emb, _alice())
    _seed(client, emb, Entity(id="person:alice-work", type="person", parent="person:bob",
                              facts=[_fact("Works at Initech")]))

    r = move_facts(client, emb, "person:alice", "person:alice-work", ["Works at Initech."],
                   domain="personal", now=NOW)

    assert not r.created
    target = get_entity(client, "person:alice-work", domain="personal")
    assert target.parent == "person:bob"
    assert _texts(target) == ["Works at Initech", "Works at Initech."]


def test_dry_run_writes_nothing(client, emb):
    _seed(client, emb, _alice())

    r = move_facts(client, emb, "person:alice", "person:alice-health", ["Allergic to penicillin."],
                   domain="personal", now=NOW, apply=False)

    assert (r.moved, r.created, r.source_facts, r.target_facts) == (["Allergic to penicillin."], True, 3, 1)
    assert get_entity(client, "person:alice-health", domain="personal") is None
    assert len(get_entity(client, "person:alice", domain="personal").facts) == 4


def test_write_landing_mid_move_is_kept(client, emb, monkeypatch):
    _seed(client, emb, _alice())
    real = split_mod.write_entity

    def racing_store(client_, entity, *a, **kw):
        real(client_, entity, *a, **kw)
        if entity.id == "person:alice-health":
            fresh = get_entity(client, "person:alice", domain="personal")
            fresh.facts.append(_fact("Has a cat."))
            real(client_, fresh, emb, domain="personal", now=NOW)

    monkeypatch.setattr(split_mod, "write_entity", racing_store)
    move_facts(client, emb, "person:alice", "person:alice-health", ["Allergic to penicillin."],
               domain="personal", now=NOW)

    assert _texts(get_entity(client, "person:alice", domain="personal")) == [
        "Lives in London.", "Runs 5k three times a week.", "Works at Initech.", "Has a cat.",
    ]


def test_source_deleted_mid_move_is_not_resurrected(client, emb, monkeypatch):
    _seed(client, emb, _alice())
    real = split_mod.write_entity

    def delete_after_target(client_, entity, *a, **kw):
        real(client_, entity, *a, **kw)
        if entity.id == "person:alice-health":
            delete_entity(client, "person:alice", domain="personal")

    monkeypatch.setattr(split_mod, "write_entity", delete_after_target)
    r = move_facts(client, emb, "person:alice", "person:alice-health", ["Allergic to penicillin."],
                   domain="personal", now=NOW)

    assert r.source_facts == 0
    assert get_entity(client, "person:alice", domain="personal") is None
    assert get_entity(client, "person:alice-health", domain="personal") is not None


@pytest.mark.parametrize("source,target,message", [
    ("alice", "person:alice-x", "full entity id"),
    ("person:alice", "alice-x", "full entity id"),
    ("person:alice", "person:", "full entity id"),
    ("person:alice", "person:alice", "same entity"),
    ("person:nobody", "person:nobody-x", "not found"),
])
def test_move_rejects_bad_ids(client, emb, source, target, message):
    _seed(client, emb, _alice())
    with pytest.raises(ValueError, match=message):
        move_facts(client, emb, source, target, ["Lives in London."], domain="personal", now=NOW)


def test_move_stays_in_its_domain(client, emb):
    _seed(client, emb, _alice(), domain="dev")
    with pytest.raises(ValueError, match="not found"):
        move_facts(client, emb, "person:alice", "person:alice-x", ["Lives in London."],
                   domain="personal", now=NOW)


# --- planning ----------------------------------------------------------------


def _blob(center, n, spread=0.05, seed=0):
    rng = np.random.default_rng(seed)
    return center + spread * rng.standard_normal((n, len(center)))


def test_cluster_finds_separated_groups_deterministically():
    a, b, c = np.eye(8)[0], np.eye(8)[1], np.eye(8)[2]
    vectors = np.vstack([_blob(a, 5, seed=1), _blob(b, 4, seed=2), _blob(c, 6, seed=3)])
    order = np.random.default_rng(7).permutation(len(vectors))
    shuffled = vectors[order]

    groups = cluster(shuffled, 3)

    assert groups == cluster(shuffled, 3)
    assert sorted(len(g) for g in groups) == [4, 5, 6]
    labels = np.repeat([0, 1, 2], [5, 4, 6])[order]
    assert all(len({labels[i] for i in g}) == 1 for g in groups)


def test_cluster_edge_cases():
    assert cluster(np.zeros((0, 4)), 3) == []
    assert cluster(np.eye(3), 5) == [[0], [1], [2]]
    assert cluster(np.eye(3), 1) == [[0, 1, 2]]


def test_default_group_count_is_clamped():
    assert [default_group_count(n) for n in (5, 50, 121, 400)] == [2, 4, 10, 12]


def test_keywords_skip_numbers_versions_and_excluded_words():
    groups = [
        ["Espresso dial-in at 18 clicks.", "Espresso shots pulled at 2025-11-02.", "alice's espresso v0.8.0"],
        ["Knee physio on Tuesdays.", "Knee brace from 2024.", "alice's knee surgery"],
    ]
    kws = keywords(groups, exclude=frozenset({"alice"}))
    assert kws[0][0] == "espresso" and kws[1][0] == "knee"
    flat = {w for k in kws for w in k}
    assert not flat & {"alice", "2025-11-02", "0.8.0", "v0.8.0", "18", "2024"}


def test_suggest_targets_are_unique():
    assert suggest_targets("person:alice", [["health", "knee"], ["health"], ["health", "knee"], []]) == [
        "person:alice-health", "person:alice-topic2", "person:alice-knee", "person:alice-topic4",
    ]


def _topical_alice():
    """Two clear topics of four facts each, plus two loners."""
    health = ["Knee physio on Tuesdays.", "Allergic to penicillin.", "Knee brace since 2024.", "Sees Dr Patel."]
    work = ["Works at Initech.", "Manager is Bill.", "Initech office in Leeds.", "Work laptop is a ThinkPad."]
    loners = ["Born in 1990.", "Likes jazz."]
    table = {}
    for i, t in enumerate(health):
        table[t] = list(np.eye(DIMS)[0] + 0.01 * i * np.eye(DIMS)[10])
    for i, t in enumerate(work):
        table[t] = list(np.eye(DIMS)[1] + 0.01 * i * np.eye(DIMS)[11])
    table[loners[0]] = list(np.eye(DIMS)[2])
    table[loners[1]] = list(np.eye(DIMS)[3])
    entity = Entity(id="person:alice", type="person", facts=[_fact(t) for t in health + work + loners])
    entity.facts.append(_fact("Old superseded fact.", superseded_at="2026-01-01"))
    return entity, table, health, work, loners


def test_plan_split_groups_topics_and_keeps_small_groups():
    entity, table, health, work, loners = _topical_alice()

    plan = plan_split(entity, _Embedder(table), domain="personal", now=NOW, groups=4)

    assert plan["source"] == "person:alice" and plan["domain"] == "personal"
    assert sorted(sorted(g["facts"]) for g in plan["groups"]) == sorted([sorted(health), sorted(work)])
    assert sorted(plan["keep"]) == sorted(loners)
    assert {g["target"] for g in plan["groups"]} == {"person:alice-knee", "person:alice-initech"}
    assert "Old superseded fact." not in json.dumps(plan)
    validate_plan(plan)


@pytest.mark.parametrize("mutate,message", [
    (lambda p: p.pop("source"), "missing"),
    (lambda p: p["groups"][1].update(target=p["groups"][0]["target"]), "two groups"),
    (lambda p: p["groups"][0].update(target=p["source"]), "source itself"),
    (lambda p: p["groups"][0].update(target="alice-x"), "full entity id"),
    (lambda p: p["groups"][1]["facts"].append(p["groups"][0]["facts"][0]), "in both"),
    (lambda p: p["keep"].append(p["groups"][0]["facts"][0]), "keep"),
])
def test_validate_plan_rejects_ambiguous_plans(mutate, message):
    entity, table, *_ = _topical_alice()
    plan = plan_split(entity, _Embedder(table), domain="personal", now=NOW, groups=4)
    mutate(plan)
    with pytest.raises(ValueError, match=message):
        validate_plan(plan)


def test_apply_plan_moves_every_group_and_is_rerunnable(client):
    entity, table, health, work, loners = _topical_alice()
    emb = _Embedder(table)
    _seed(client, emb, entity)
    plan = plan_split(entity, emb, domain="personal", now=NOW, groups=4)

    dry = apply_plan(client, emb, plan, now=NOW)
    assert [len(r.moved) for r in dry] == [4, 4]
    assert get_entity(client, "person:alice-knee", domain="personal") is None

    apply_plan(client, emb, plan, now=NOW, apply=True)
    again = apply_plan(client, emb, plan, now=NOW, apply=True)

    assert [(len(r.moved), len(r.already)) for r in again] == [(0, 4), (0, 4)]
    assert sorted(_texts(get_entity(client, "person:alice-knee", domain="personal"))) == sorted(health)
    assert sorted(_texts(get_entity(client, "person:alice-initech", domain="personal"))) == sorted(work)
    assert _texts(get_entity(client, "person:alice", domain="personal")) == [*loners, "Old superseded fact."]
    assert list_children(client, "person:alice", domain="personal") == [
        "person:alice-initech", "person:alice-knee",
    ]


def test_cli_split_plan_then_apply(client, tmp_path, monkeypatch):
    entity, table, health, work, loners = _topical_alice()
    emb = _Embedder(table)
    _seed(client, emb, entity)
    monkeypatch.setattr(cli_mod, "get_client", lambda: client)
    monkeypatch.setattr(cli_mod, "_get_embedder", lambda: emb)
    plan_file = tmp_path / "plan.json"

    res = CliRunner().invoke(cli_mod.main, [
        "split-plan", "person:alice", "--domain", "personal", "--groups", "4", "-o", str(plan_file),
    ])
    assert res.exit_code == 0, res.output
    plan = json.loads(plan_file.read_text())
    plan["groups"][0]["target"] = "person:alice-renamed"
    plan["groups"][0]["facts"].append("Not a real fact.")
    plan_file.write_text(json.dumps(plan))

    res = CliRunner().invoke(cli_mod.main, ["split-apply", str(plan_file)])
    assert res.exit_code == 0, res.output
    assert "DRY-RUN" in res.output and "not found: Not a real fact." in res.output
    assert get_entity(client, "person:alice-renamed", domain="personal") is None

    res = CliRunner().invoke(cli_mod.main, ["split-apply", str(plan_file), "--apply"])
    assert res.exit_code == 0, res.output
    assert "person:alice-renamed (new): moved 4" in res.output
    assert "person:alice keeps 2 current facts." in res.output

    res = CliRunner().invoke(cli_mod.main, ["get", "person:alice", "--domain", "personal"])
    assert "Children: person:alice-initech, person:alice-renamed" in res.output
    res = CliRunner().invoke(cli_mod.main, ["get", "person:alice-renamed", "--domain", "personal"])
    assert "Parent: person:alice" in res.output


def test_cli_split_apply_rejects_invalid_plan(client, tmp_path, monkeypatch):
    monkeypatch.setattr(cli_mod, "get_client", lambda: client)
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps({"source": "person:alice", "domain": "personal", "groups": [
        {"target": "person:alice", "facts": []},
    ]}))
    res = CliRunner().invoke(cli_mod.main, ["split-apply", str(plan_file)])
    assert res.exit_code != 0 and "source itself" in res.output


# --- the parent field ----------------------------------------------------------


def test_parent_survives_point_and_export_round_trips(client, emb):
    _seed(client, emb, Entity(id="person:alice", type="person", facts=[_fact("a")]))
    _seed(client, emb, Entity(id="person:alice-x", type="person", parent="person:alice", facts=[_fact("b")]))

    child = get_entity(client, "person:alice-x", domain="personal")
    assert child.parent == "person:alice"
    out = io.StringIO()
    export_json([child, get_entity(client, "person:alice", domain="personal")], out)
    restored = import_json(json.loads(out.getvalue()))
    assert [e.parent for e in restored] == ["person:alice", None]
    # Pre-parent backups import with no parent.
    assert import_json([{"id": "person:z", "type": "person", "facts": []}])[0].parent is None


def test_list_children_is_per_domain(client, emb):
    _seed(client, emb, Entity(id="person:alice-x", type="person", parent="person:alice"), domain="dev")
    _seed(client, emb, Entity(id="decision:alice-y", type="decision", parent="person:alice"), domain="dev")
    assert list_children(client, "person:alice", domain="dev") == ["decision:alice-y", "person:alice-x"]
    assert list_children(client, "person:alice", domain="personal") == []


def test_cli_store_parent_must_exist(client, emb, monkeypatch):
    monkeypatch.setattr(cli_mod, "get_client", lambda: client)
    monkeypatch.setattr(cli_mod, "_get_embedder", lambda: emb)
    args = ["store", "--type", "person", "--id", "alice-x", "--content", "c", "--parent"]

    res = CliRunner().invoke(cli_mod.main, [*args, "person:alice"])
    assert res.exit_code != 0 and "not found" in res.output

    _seed(client, emb, Entity(id="person:alice", type="person", facts=[_fact("a")]), domain="shared")
    res = CliRunner().invoke(cli_mod.main, [*args, "person:alice"])
    assert res.exit_code == 0, res.output
    assert get_entity(client, "person:alice-x").parent == "person:alice"


def _interrupt(client, emb, monkeypatch, *args):
    """Run a move that dies right after writing the target."""
    real = split_mod.write_entity

    def crash_on_source(client_, entity, *a, **kw):
        if entity.id == args[0]:
            raise RuntimeError("killed")
        real(client_, entity, *a, **kw)

    monkeypatch.setattr(split_mod, "write_entity", crash_on_source)
    with pytest.raises(RuntimeError):
        move_facts(client, emb, *args, domain="personal", now=NOW)
    monkeypatch.setattr(split_mod, "write_entity", real)


@pytest.mark.parametrize("side", ["person:alice", "person:alice-work"])
def test_a_copy_changed_before_the_rerun_leaves_a_duplicate_not_a_loss(client, emb, monkeypatch, side):
    _seed(client, emb, Entity(id="person:alice", type="person", facts=[_fact("Works at Initech.")]))
    _interrupt(client, emb, monkeypatch, "person:alice", "person:alice-work", ["Works at Initech."])
    changed = get_entity(client, side, domain="personal")
    mark_superseded(changed, "Works at Initech.", by="Works at Acme.", on_date="2026-11-20")
    changed.facts.append(_fact("Works at Acme."))
    _seed(client, emb, changed)

    move_facts(client, emb, "person:alice", "person:alice-work", ["Works at Initech."],
               domain="personal", now=NOW)

    target = get_entity(client, "person:alice-work", domain="personal")
    source = get_entity(client, "person:alice", domain="personal")
    initech = sorted(f.is_current for f in target.facts if f.text == "Works at Initech.")
    assert initech == [False, True]  # both versions kept; the operator resolves it
    assert "Works at Initech." not in _texts(source)
    assert "Works at Acme." in _texts(source) + _texts(target)


def test_a_requested_text_takes_current_facts_not_old_history(client, emb):
    _seed(client, emb, Entity(id="person:alice", type="person", facts=[
        _fact("Said hello."), _fact("Said hello.", superseded_at="2026-05-01"), _fact("Said hello."),
    ]))
    r = move_facts(client, emb, "person:alice", "person:alice-x", ["Said hello."], domain="personal", now=NOW)
    assert len(get_entity(client, "person:alice-x", domain="personal").facts) == 2
    assert [f.is_current for f in get_entity(client, "person:alice", domain="personal").facts] == [False]
    # With no current occurrence left, the text selects its history.
    move_facts(client, emb, "person:alice", "person:alice-x", ["Said hello."], domain="personal", now=NOW)
    assert get_entity(client, "person:alice", domain="personal").facts == []
    assert r.moved == ["Said hello."]


@pytest.mark.parametrize("order", [("linux", "mac"), ("mac", "linux")])
def test_history_of_a_reasserted_text_follows_its_replacement_in_any_order(client, emb, order):
    _seed(client, emb, Entity(id="person:alice", type="person", facts=[
        _fact("Uses Linux.", added="2020-01-01", superseded_at="2023-01-01", superseded_by="Uses macOS."),
        _fact("Uses macOS.", added="2023-01-01"),
        _fact("Uses Linux.", added="2026-01-01"),
    ]))
    groups = {"linux": ("person:alice-linux", "Uses Linux."), "mac": ("person:alice-mac", "Uses macOS.")}
    for name in order:
        target, text = groups[name]
        move_facts(client, emb, "person:alice", target, [text], domain="personal", now=NOW)

    linux = get_entity(client, "person:alice-linux", domain="personal")
    mac = get_entity(client, "person:alice-mac", domain="personal")
    assert [(f.text, f.added) for f in linux.facts] == [("Uses Linux.", "2026-01-01")]
    assert sorted((f.text, f.added) for f in mac.facts) == [("Uses Linux.", "2020-01-01"), ("Uses macOS.", "2023-01-01")]


def test_rerun_repairs_the_index_after_a_crash_inside_the_source_write(client, emb):
    _seed(client, emb, _alice())
    move_facts(client, emb, "person:alice", "person:alice-health", ["Allergic to penicillin."],
               domain="personal", now=NOW)
    # Put back a stale point, as if the source write died after its entity
    # upsert but before its fact sync.
    stale = get_entity(client, "person:alice", domain="personal")
    stale.facts.append(_fact("Allergic to penicillin."))
    facts_mod_sync(client, stale, emb)
    assert "Allergic to penicillin." in _fact_point_texts(client, "personal", "person:alice")

    r = move_facts(client, emb, "person:alice", "person:alice-health", ["Allergic to penicillin."],
                   domain="personal", now=NOW)

    assert r.already == ["Allergic to penicillin."]
    assert "Allergic to penicillin." not in _fact_point_texts(client, "personal", "person:alice")


def test_dry_run_rerun_writes_nothing(client, emb, monkeypatch):
    _seed(client, emb, _alice())
    move_facts(client, emb, "person:alice", "person:alice-health", ["Allergic to penicillin."],
               domain="personal", now=NOW)
    monkeypatch.setattr(split_mod, "sync_fact_points", lambda *a, **k: pytest.fail("dry run wrote"))
    monkeypatch.setattr(split_mod, "write_entity", lambda *a, **k: pytest.fail("dry run wrote"))
    move_facts(client, emb, "person:alice", "person:alice-health", ["Allergic to penicillin."],
               domain="personal", now=NOW, apply=False)


def test_move_lock_is_exclusive_per_domain(tmp_path, monkeypatch):
    import fcntl

    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    with split_mod.move_lock("personal"):
        with open(tmp_path / "kipp-move-personal.lock", "a") as other:
            with pytest.raises(BlockingIOError):
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with split_mod.move_lock("dev"):
            pass
    with open(tmp_path / "kipp-move-personal.lock", "a") as other:
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)

