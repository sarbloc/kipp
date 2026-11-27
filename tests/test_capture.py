"""Transcript capture into the spool (issue #71). Fixture transcripts only, no real data."""

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from kipp import capture
from kipp.capture import (
    CaptureConfig,
    capture_transcript,
    claude_code_turns,
    clip,
    redact,
    resolve_domain,
    spool_path,
    strip_injected,
)

CFG = CaptureConfig(routes={}, default_domain="shared", strip_tags=capture.DEFAULT_STRIP_TAGS)


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    config = tmp_path / "config.json"
    config.write_text("{}")
    monkeypatch.setenv("KIPP_CONFIG", str(config))
    monkeypatch.delenv("QDRANT_URL", raising=False)
    return tmp_path


# --- transcript builders -------------------------------------------------------


def user(uid, content, **flags):
    return {"type": "user", "uuid": uid, "timestamp": f"2026-11-27T10:00:{uid[-2:]}Z",
            "cwd": "/home/u/dev/app", "message": {"role": "user", "content": content}, **flags}


def assistant(uid, *blocks, **flags):
    return {"type": "assistant", "uuid": uid, "timestamp": f"2026-11-27T10:01:{uid[-2:]}Z",
            "message": {"role": "assistant", "content": list(blocks)}, **flags}


def text(t):
    return {"type": "text", "text": t}


def tool_use():
    return {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}}


def tool_result(out):
    return [{"type": "tool_result", "tool_use_id": "toolu_1", "content": out}]


def write(path: Path, records, *, partial: str = ""):
    path.write_text("".join(json.dumps(r) + "\n" for r in records) + partial)


def append(path: Path, records):
    with open(path, "a") as f:
        f.write("".join(json.dumps(r) + "\n" for r in records))


def spooled():
    p = spool_path()
    return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []


def turns(records):
    return list(claude_code_turns(records, session_id="s1"))


# --- filtering and grouping ------------------------------------------------------


def test_keeps_dialogue_and_drops_tools_thinking_and_injections():
    (t,) = turns([
        user("u-01", "<system-reminder>be nice</system-reminder>\nUse pnpm, not npm."),
        assistant("a-01", {"type": "thinking", "thinking": "hmm"}, tool_use()),
        user("u-02", tool_result("node_modules\nsrc\n" * 500)),
        assistant("a-02", text("Noted: pnpm from now on."), tool_use()),
    ])

    assert t.user == "Use pnpm, not npm."
    assert t.assistant == "Noted: pnpm from now on."
    assert t.text == "User: Use pnpm, not npm.\n\nAssistant: Noted: pnpm from now on."
    assert "node_modules" not in t.text and "hmm" not in t.text and "be nice" not in t.text


def test_skips_sidechain_meta_and_compaction_records():
    out = turns([
        user("u-01", "Real question?"),
        assistant("a-01", text("Real answer.")),
        user("u-02", "subagent prompt", isSidechain=True),
        assistant("a-02", text("subagent reply"), isSidechain=True),
        user("u-03", "Caveat: local command output", isMeta=True),
        user("u-04", "This session is being continued...", isCompactSummary=True),
    ])

    assert [(t.user, t.assistant) for t in out] == [("Real question?", "Real answer.")]


def test_one_turn_per_user_message_collecting_every_reply_block():
    out = turns([
        user("u-01", [text("First ask.")]),
        assistant("a-01", text("Part one.")),
        assistant("a-02", text("(no content)")),
        assistant("a-03", text("Part two.")),
        user("u-02", "Second ask."),
        assistant("a-04", text("Done.")),
    ])

    assert [(t.user, t.assistant) for t in out] == [
        ("First ask.", "Part one.\n\nPart two."), ("Second ask.", "Done."),
    ]
    assert out[0].anchor == "u-01" and out[0].cwd == "/home/u/dev/app"


def test_injection_only_user_message_does_not_open_a_turn():
    out = turns([
        user("u-01", "Ask."),
        assistant("a-01", text("Answer.")),
        user("u-02", "<system-reminder>todo list changed</system-reminder>"),
        assistant("a-02", text("More answer.")),
    ])

    assert [(t.user, t.assistant) for t in out] == [("Ask.", "Answer.\n\nMore answer.")]


def test_unprompted_reply_gets_a_turn_with_no_user_side():
    (t,) = turns([assistant("a-01", text("Background build finished."))])

    assert t.user == "" and t.text == "Assistant: Background build finished."


def test_unknown_and_malformed_records_are_skipped():
    out = turns([
        {"type": "ai-title", "title": "x"},
        {"type": "user", "message": {"content": "no uuid"}},
        user("u-01", {"unexpected": "shape"}),
        user("u-02", "Kept."),
    ])

    assert [t.user for t in out] == ["Kept."]


def test_turn_ids_are_stable_per_session_and_anchor():
    records = [user("u-01", "Ask."), assistant("a-01", text("Answer."))]
    a = list(claude_code_turns(records, session_id="s1"))[0].id
    b = list(claude_code_turns(records, session_id="s1"))[0].id
    c = list(claude_code_turns(records, session_id="s2"))[0].id

    assert a == b != c


# --- cleaning ----------------------------------------------------------------------


def test_strip_injected_handles_attributes_and_multiline():
    raw = 'Hi\n<context>\nLocal time: now\n</context>\n\n\n<command-name x="1">/foo</command-name>there'
    assert strip_injected(raw) == "Hi\n\nthere"


@pytest.mark.parametrize("secret", [
    "sk-ant-api03-" + "a" * 40,
    "ghp_" + "A" * 36,
    "github_pat_" + "B" * 30,
    "AKIA" + "ABCDEFGHIJKLMNOP",
    "AIza" + "c" * 35,
    "xoxb-1234567890-abcdef",
    "Bearer " + "e" * 30,
    "-----BEGIN RSA PRIVATE KEY-----\nMIIabc\n-----END RSA PRIVATE KEY-----",
])
def test_redacts_known_secret_formats(secret):
    out = redact(f"my key is {secret} ok")

    assert secret not in out and "[REDACTED]" in out and out.endswith(" ok")


def test_redaction_runs_before_clipping():
    secret = "ghp_" + "Z" * 36
    (t,) = turns([user("u-01", "x" * (capture.CLIP_CHARS - 10) + " " + secret)])

    assert "ghp_" not in t.user and "ZZZZZZZZZZ" not in t.user and "[REDACTED" in t.user


def test_clip_marks_what_it_cut():
    assert clip("abcdef", limit=4) == "abcd [...clipped 2 chars]"
    assert clip("abc", limit=4) == "abc"


# --- domain routing ------------------------------------------------------------------


def test_longest_route_prefix_wins_on_whole_components(monkeypatch):
    monkeypatch.setenv("HOME", "/home/u")
    routes = {"~/dev": "dev", "~/dev/notes": "personal"}

    assert resolve_domain("/home/u/dev/app", routes) == "dev"
    assert resolve_domain("/home/u/dev/notes/x", routes) == "personal"
    assert resolve_domain("/home/u/devices", routes) == "shared"
    assert resolve_domain("/home/u/dev", routes) == "dev"
    assert resolve_domain("", routes, default="dev") == "dev"


def test_config_rejects_unknown_domains(isolated_state):
    (isolated_state / "config.json").write_text(json.dumps({"capture": {"domains": {"~/x": "work"}}}))

    with pytest.raises(ValueError, match="unknown domain 'work'"):
        CaptureConfig.load()


def test_config_reads_routes_and_default(isolated_state):
    (isolated_state / "config.json").write_text(json.dumps(
        {"capture": {"domains": {"~/dev": "dev"}, "default_domain": "personal"}}))

    cfg = CaptureConfig.load()

    assert cfg.routes == {"~/dev": "dev"} and cfg.default_domain == "personal"


# --- capture: spool and cursor ---------------------------------------------------------


def test_capture_spools_turns_with_domain(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", "/home/u")
    t = tmp_path / "s1.jsonl"
    write(t, [user("u-01", "Ask."), assistant("a-01", text("Answer."))])
    cfg = CaptureConfig(routes={"~/dev": "dev"}, default_domain="shared",
                        strip_tags=capture.DEFAULT_STRIP_TAGS)

    assert capture_transcript(t, session_id="s1", config=cfg) == 1

    (rec,) = spooled()
    assert rec["domain"] == "dev" and rec["session_id"] == "s1" and rec["client"] == "claude-code"
    assert rec["text"] == "User: Ask.\n\nAssistant: Answer."


def test_recapture_resends_only_the_last_turn_then_new_ones(tmp_path):
    t = tmp_path / "s1.jsonl"
    write(t, [user("u-01", "One."), assistant("a-01", text("R1.")),
              user("u-02", "Two."), assistant("a-02", text("R2."))])
    capture_transcript(t, session_id="s1", config=CFG)
    append(t, [assistant("a-03", text("R2 continued.")), user("u-03", "Three.")])

    capture_transcript(t, session_id="s1", config=CFG)

    texts = [r["text"] for r in spooled()]
    assert texts == [
        "User: One.\n\nAssistant: R1.",
        "User: Two.\n\nAssistant: R2.",
        "User: Two.\n\nAssistant: R2.\n\nR2 continued.",
        "User: Three.",
    ]
    ids = [r["id"] for r in spooled()]
    assert ids[1] == ids[2]  # the flush keeps the later, complete version


def test_partial_trailing_line_waits_for_the_next_run(tmp_path):
    t = tmp_path / "s1.jsonl"
    half = json.dumps(user("u-02", "Two."))[:20]
    write(t, [user("u-01", "One.")], partial=half)

    capture_transcript(t, session_id="s1", config=CFG)
    assert [r["text"] for r in spooled()] == ["User: One."]

    write(t, [user("u-01", "One."), user("u-02", "Two.")])
    capture_transcript(t, session_id="s1", config=CFG)
    assert [r["text"] for r in spooled()][-1] == "User: Two."


def test_shrunk_transcript_is_read_from_the_start(tmp_path):
    t = tmp_path / "s1.jsonl"
    write(t, [user("u-01", "One." * 50), user("u-02", "Two.")])
    capture_transcript(t, session_id="s1", config=CFG)
    write(t, [user("u-09", "Fresh.")])

    capture_transcript(t, session_id="s1", config=CFG)

    assert spooled()[-1]["text"] == "User: Fresh."


def test_no_new_dialogue_spools_nothing(tmp_path):
    t = tmp_path / "s1.jsonl"
    write(t, [{"type": "ai-title", "title": "x"}])

    assert capture_transcript(t, session_id="s1", config=CFG) == 0
    assert not spool_path().exists()


# --- entry point ---------------------------------------------------------------------------


def run_hook(monkeypatch, payload: str) -> int:
    monkeypatch.setattr(sys, "stdin", io.StringIO(payload))
    return capture.main(["--hook"])


def test_hook_captures_from_the_payload(tmp_path, monkeypatch):
    t = tmp_path / "s1.jsonl"
    write(t, [user("u-01", "Ask."), assistant("a-01", text("Answer."))])
    payload = json.dumps({"session_id": "s1", "transcript_path": str(t), "cwd": "/tmp",
                          "hook_event_name": "Stop"})

    assert run_hook(monkeypatch, payload) == 0
    assert [r["session_id"] for r in spooled()] == ["s1"]


@pytest.mark.parametrize("payload", [
    "not json",
    json.dumps({"session_id": "s1"}),
    json.dumps({"session_id": "s1", "transcript_path": "/nonexistent/x.jsonl"}),
])
def test_hook_never_fails_and_logs_why(monkeypatch, payload):
    assert run_hook(monkeypatch, payload) == 0

    log_text = (capture.state_dir() / "capture.log").read_text()
    assert "ERROR" in log_text
    assert not spool_path().exists()


def test_file_mode_reports_errors_with_exit_1(tmp_path, capsys):
    assert capture.main(["--file", str(tmp_path / "missing.jsonl")]) == 1


def test_hook_module_does_not_import_qdrant_or_the_model():
    code = ("import sys, kipp.capture; "
            "heavy = [m for m in sys.modules if m.split('.')[0] in "
            "('qdrant_client', 'sentence_transformers', 'torch', 'httpx', 'click')]; "
            "print(heavy)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True, env={"PYTHONPATH": str(Path(capture.__file__).parents[1])})

    assert out.stdout.strip() == "[]"
