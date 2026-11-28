"""Tests for flushing the capture spool into events (issue #71). Fixture data only."""

import json
import os
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from click.testing import CliRunner

import kipp.cli as cli_mod
from kipp import capture, flush
from kipp.client import collection_name, get_unextracted_events, mark_event_extracted
from kipp.flush import flush_spool, naive_utc, version_id

NOW = datetime(2026, 11, 28, 8, 0, 0)


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    config = tmp_path / "config.json"
    config.write_text("{}")
    monkeypatch.setenv("KIPP_CONFIG", str(config))
    return tmp_path


def rec(n: int, text: str = "", *, domain: str = "shared", **over) -> dict:
    r = {
        "id": str(uuid.uuid5(capture.TURN_NAMESPACE, f"t{n}")),
        "anchor": f"u-{n}", "client": "claude-code", "session_id": "s1",
        "timestamp": "2026-11-27T10:00:00Z", "cwd": "/home/u/dev/app",
        "user": "q", "assistant": "a", "domain": domain,
        "text": text or f"User: question {n}\n\nAssistant: answer {n}",
    }
    return {**r, **over}


def spool(*records, raw: str = "") -> None:
    capture.ensure_state_dir()
    capture._append_to_spool("".join(json.dumps(r) + "\n" for r in records) + raw)


def event(client, r: dict, domain: str = "shared"):
    vid = version_id(r["id"], r["text"])
    points = client.retrieve(collection_name(domain, "events"), [vid], with_payload=True)
    return points[0].payload if points else None


class CountingEmbedder:
    def __init__(self, inner, on_load=None):
        self.inner, self.on_load, self.loads, self.embedded = inner, on_load, 0, []

    def __call__(self):
        self.loads += 1
        if self.on_load:
            self.on_load()
        return self

    def embed_batch(self, texts):
        self.embedded += texts
        return self.inner.embed_batch(texts)


def no_embedder():
    raise AssertionError("the embedder must not load")


def test_flush_writes_each_turn_to_its_domain_and_empties_the_spool(cli_env, embedder):
    a, b = rec(1), rec(2, domain="dev")
    spool(a, b)

    r = flush_spool(cli_env, CountingEmbedder(embedder), now=NOW)

    assert (r.written, r.unchanged, r.superseded) == (2, 0, 0)
    assert dict(r.domains) == {"shared": 1, "dev": 1}
    p = event(cli_env, a)
    assert p["text"] == a["text"] and p["source"] == "capture" and p["extracted"] is False
    assert p["session_id"] == "s1" and p["client"] == "claude-code" and p["cwd"] == "/home/u/dev/app"
    assert p["turn_id"] == a["id"]
    assert p["timestamp"] == "2026-11-27T10:00:00"
    assert p["expires"] == (NOW + timedelta(days=30)).isoformat()
    assert event(cli_env, b, "dev")["text"] == b["text"] and event(cli_env, b) is None
    assert not capture.spool_path().exists() and not capture.flushing_path().exists()


def test_last_copy_of_a_turn_wins(cli_env, embedder):
    spool(rec(1, "User: q\n\nAssistant: partial"), rec(2), rec(1, "User: q\n\nAssistant: complete"))
    emb = CountingEmbedder(embedder)

    r = flush_spool(cli_env, emb, now=NOW)

    assert r.written == 2 and len(emb.embedded) == 2
    assert event(cli_env, rec(1, "User: q\n\nAssistant: complete")) is not None
    assert event(cli_env, rec(1, "User: q\n\nAssistant: partial")) is None


def test_unchanged_turn_is_skipped_and_keeps_its_extracted_state(cli_env, embedder):
    spool(rec(1))
    flush_spool(cli_env, CountingEmbedder(embedder), now=NOW)
    mark_event_extracted(cli_env, version_id(rec(1)["id"], rec(1)["text"]), domain="shared",
                         resolved_into="project:x")
    spool(rec(1))  # the hook re-spools the last turn every run

    r = flush_spool(cli_env, no_embedder, now=NOW)

    assert (r.written, r.unchanged) == (0, 1)
    p = event(cli_env, rec(1))
    assert p["extracted"] is True and p["resolved_into"] == "project:x"


V1 = "User: q\n\nAssistant: partial"
V2 = "User: q\n\nAssistant: partial, then the rest"


def test_changed_turn_after_consolidation_is_a_new_event(cli_env, embedder):
    spool(rec(1, V1))
    flush_spool(cli_env, CountingEmbedder(embedder), now=NOW)
    mark_event_extracted(cli_env, version_id(rec(1)["id"], V1), domain="shared",
                         resolved_into="project:x")
    spool(rec(1, V2))

    r = flush_spool(cli_env, CountingEmbedder(embedder), now=NOW)

    assert (r.written, r.superseded) == (1, 0)
    assert event(cli_env, rec(1, V2))["extracted"] is False
    old = event(cli_env, rec(1, V1))
    assert old["extracted"] is True and old["resolved_into"] == "project:x"


def test_changed_turn_retires_the_unconsolidated_older_version(cli_env, embedder):
    spool(rec(1, V1), rec(2))
    flush_spool(cli_env, CountingEmbedder(embedder), now=NOW)
    spool(rec(1, V2))

    r = flush_spool(cli_env, CountingEmbedder(embedder), now=NOW)

    assert (r.written, r.superseded) == (1, 1)
    old = event(cli_env, rec(1, V1))
    assert old["extracted"] is True and old["superseded_by"] == version_id(rec(1)["id"], V2)
    assert event(cli_env, rec(2))["extracted"] is False  # another turn is untouched
    queued = {e["id"] for e in get_unextracted_events(cli_env)}
    assert queued == {version_id(rec(1)["id"], V2), version_id(rec(2)["id"], rec(2)["text"])}


def test_resolving_a_version_read_before_an_update_leaves_the_update_queued(cli_env, embedder):
    spool(rec(1, V1))
    flush_spool(cli_env, CountingEmbedder(embedder), now=NOW)
    (read,) = get_unextracted_events(cli_env)  # a consolidator reads V1...
    spool(rec(1, V2))
    flush_spool(cli_env, CountingEmbedder(embedder), now=NOW)  # ...V2 lands...

    mark_event_extracted(cli_env, read["id"], domain="shared")  # ...then V1 is resolved

    (left,) = get_unextracted_events(cli_env)
    assert left["text"] == V2


def test_rewound_turn_requeues_the_text_that_is_current_again(cli_env, embedder):
    spool(rec(1, V1))
    flush_spool(cli_env, CountingEmbedder(embedder), now=NOW)
    spool(rec(1, V2))
    flush_spool(cli_env, CountingEmbedder(embedder), now=NOW)  # V1 retired
    spool(rec(1, V1))  # the transcript was rewound

    r = flush_spool(cli_env, no_embedder, now=NOW)

    assert (r.written, r.requeued, r.superseded, r.unchanged) == (0, 1, 1, 0)
    (left,) = get_unextracted_events(cli_env)
    assert left["text"] == V1 and "superseded_by" not in left
    assert event(cli_env, rec(1, V2))["superseded_by"] == version_id(rec(1)["id"], V1)


def test_rewind_to_a_consolidated_text_retires_the_queued_one(cli_env, embedder):
    spool(rec(1, V1))
    flush_spool(cli_env, CountingEmbedder(embedder), now=NOW)
    mark_event_extracted(cli_env, version_id(rec(1)["id"], V1), domain="shared")
    spool(rec(1, V2))
    flush_spool(cli_env, CountingEmbedder(embedder), now=NOW)
    spool(rec(1, V1))

    r = flush_spool(cli_env, no_embedder, now=NOW)

    assert (r.unchanged, r.requeued, r.superseded) == (1, 0, 1)
    assert get_unextracted_events(cli_env) == []


def test_resolved_version_is_not_requeued_after_it_was_retired(cli_env, embedder):
    spool(rec(1, V1))
    flush_spool(cli_env, CountingEmbedder(embedder), now=NOW)
    (read,) = get_unextracted_events(cli_env)
    spool(rec(1, V2))
    flush_spool(cli_env, CountingEmbedder(embedder), now=NOW)  # retires V1
    mark_event_extracted(cli_env, read["id"], domain="shared")  # V1 was handled anyway
    spool(rec(1, V1))

    r = flush_spool(cli_env, no_embedder, now=NOW)

    assert (r.unchanged, r.requeued, r.superseded) == (1, 0, 1)
    assert get_unextracted_events(cli_env) == []


def test_a_failed_retirement_is_finished_by_the_next_flush(cli_env, embedder, monkeypatch):
    spool(rec(1, V1))
    flush_spool(cli_env, CountingEmbedder(embedder), now=NOW)
    spool(rec(1, V2))

    def boom(*a, **k):
        raise RuntimeError("qdrant went away")

    with monkeypatch.context() as m, pytest.raises(RuntimeError):
        m.setattr(flush, "mark_events_superseded", boom)
        flush_spool(cli_env, CountingEmbedder(embedder), now=NOW)
    assert len(get_unextracted_events(cli_env)) == 2  # both queued, never neither

    r = flush_spool(cli_env, no_embedder, now=NOW)

    assert (r.unchanged, r.superseded) == (1, 1)
    (left,) = get_unextracted_events(cli_env)
    assert left["text"] == V2


def test_empty_or_missing_spool_never_loads_the_model(cli_env):
    assert flush_spool(cli_env, no_embedder).written == 0
    spool()
    assert flush_spool(cli_env, no_embedder).written == 0
    assert not capture.spool_path().exists()


def test_malformed_records_are_skipped_and_counted(cli_env, embedder):
    bad = [rec(2, domain="nowhere"), rec(3, id="not-a-uuid"), {**rec(4), "text": ""},
           {**rec(5), "session_id": 7}, ["not", "a", "dict"]]
    spool(rec(1), *bad, raw='{"id": "half a rec')

    r = flush_spool(cli_env, CountingEmbedder(embedder), now=NOW)

    assert r.written == 1 and r.malformed == 6
    assert event(cli_env, rec(1)) is not None


def test_failed_flush_leaves_the_moved_spool_and_the_next_one_finishes_it(cli_env, embedder):
    spool(rec(1), rec(2))

    def broken():
        raise RuntimeError("model download failed")

    with pytest.raises(RuntimeError):
        flush_spool(cli_env, broken, now=NOW)
    assert capture.flushing_path().exists() and not capture.spool_path().exists()
    spool(rec(2, "User: q2\n\nAssistant: newer"), rec(3))  # hooks kept appending

    r = flush_spool(cli_env, CountingEmbedder(embedder), now=NOW)

    # rec(2): the moved spool's copy, then the newer one, which retires it.
    assert (r.written, r.superseded) == (4, 1)
    assert event(cli_env, rec(2, "User: q2\n\nAssistant: newer"))["extracted"] is False
    assert not capture.flushing_path().exists() and not capture.spool_path().exists()


def test_hook_appends_during_a_flush_land_in_a_fresh_spool(cli_env, embedder, tmp_path):
    t = tmp_path / "s9.jsonl"
    t.write_text(json.dumps({"type": "user", "uuid": "u-01", "timestamp": "2026-11-27T10:00:01Z",
                             "cwd": "/x", "message": {"role": "user", "content": "Late."}}) + "\n")
    cfg = capture.CaptureConfig(routes={}, default_domain="shared",
                                strip_tags=capture.DEFAULT_STRIP_TAGS)
    spool(rec(1))
    # The embedder loads after the spool is moved aside and the lock released,
    # which is when a hook would run.
    emb = CountingEmbedder(embedder, on_load=lambda: capture.capture_transcript(
        t, session_id="s9", config=cfg, lock_timeout=1))

    r = flush_spool(cli_env, emb, now=NOW)

    assert r.written == 1
    (late,) = [json.loads(line) for line in capture.spool_path().read_text().splitlines()]
    assert late["text"] == "User: Late."


def test_concurrent_flush_is_refused(cli_env):
    spool(rec(1))
    with capture._Lock(0, name="flush.lock"):
        with pytest.raises(TimeoutError, match="flush lock busy"):
            flush_spool(cli_env, no_embedder)
    assert capture.spool_path().exists()


def test_spool_with_loose_permissions_is_refused(cli_env):
    spool(rec(1))
    os.chmod(capture.spool_path(), 0o644)
    with pytest.raises(ValueError, match="readable or writable by others"):
        flush_spool(cli_env, no_embedder)


@pytest.mark.parametrize("ts, want", [
    ("2026-11-27T10:00:00.123Z", "2026-11-27T10:00:00.123000"),
    ("2026-11-27T11:00:00+01:00", "2026-11-27T10:00:00"),
    ("2026-11-27T10:00:00", "2026-11-27T10:00:00"),
    ("", None),
    ("yesterday", None),
    ("0001-01-01T00:00:00+01:00", None),
    ("9999-12-31T23:59:59-01:00", None),
])
def test_timestamps_become_naive_utc(ts, want):
    assert naive_utc(ts) == want


@pytest.mark.parametrize("ts", ["garbage", "0001-01-01T00:00:00+01:00"])
def test_unusable_timestamp_falls_back_to_now(cli_env, embedder, ts):
    spool(rec(1, timestamp=ts))
    flush_spool(cli_env, CountingEmbedder(embedder), now=NOW)
    assert event(cli_env, rec(1))["timestamp"] == NOW.isoformat()
    assert not capture.flushing_path().exists()


def test_captured_events_work_with_the_since_filter(cli_env, embedder):
    now = datetime.now(timezone.utc)
    fresh = rec(1, timestamp=(now - timedelta(minutes=5)).isoformat())
    spool(fresh, rec(2, timestamp="2026-01-01T00:00:00Z"))
    flush_spool(cli_env, CountingEmbedder(embedder))

    since = (now - timedelta(hours=1)).replace(tzinfo=None)  # how callers pass it
    recent = get_unextracted_events(cli_env, since=since)

    assert [e["id"] for e in recent] == [version_id(fresh["id"], fresh["text"])]


def test_cli_flush_and_stats(cli_env):
    runner = CliRunner()
    spool(rec(1), rec(2, domain="dev"), rec(3, domain="dev"))

    out = runner.invoke(cli_mod.main, ["stats"])
    assert out.exit_code == 0, out.output
    assert "captures: 0 unconsolidated; 3 spooled, not yet flushed" in out.output

    out = runner.invoke(cli_mod.main, ["capture", "flush"])
    assert out.exit_code == 0, out.output
    assert "Flushed 3 turns (dev 2, shared 1)." in out.output

    out = runner.invoke(cli_mod.main, ["stats"])
    assert "captures: 3 unconsolidated (shared 1, dev 2); 0 spooled, not yet flushed" in out.output

    spool(rec(1))
    out = runner.invoke(cli_mod.main, ["capture", "flush"])
    assert "Nothing new to flush." in out.output and "Skipped 1 turns already stored" in out.output

    spool(rec(1, "User: q\n\nAssistant: more"))
    out = runner.invoke(cli_mod.main, ["capture", "flush"])
    assert "Flushed 1 turns (shared 1)." in out.output
    assert "Retired 1 older unconsolidated versions" in out.output
    out = runner.invoke(cli_mod.main, ["stats"])
    assert "captures: 3 unconsolidated (shared 1, dev 2)" in out.output


def test_cli_flush_reports_a_busy_lock(cli_env):
    spool(rec(1))
    with capture._Lock(0, name="flush.lock"):
        out = CliRunner().invoke(cli_mod.main, ["capture", "flush"])
    assert out.exit_code == 1 and "flush lock busy" in out.output


def test_flush_constants_are_sane():
    assert flush.SPOOL_LOCK_TIMEOUT > capture.HOOK_TIME_BUDGET + capture.HOOK_LOCK_TIMEOUT
