"""Transcript capture into the spool (issue #71). Fixture transcripts only, no real data."""

import io
import json
import os
import stat
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


def test_prose_beside_tool_results_opens_a_turn():
    out = turns([
        user("u-01", "Run it."),
        assistant("a-01", text("Running."), tool_use()),
        user("u-02", [*tool_result("ok"), text("Actually, stop after this.")]),
        assistant("a-02", text("Stopping.")),
    ])

    assert [(t.user, t.assistant) for t in out] == [
        ("Run it.", "Running."), ("Actually, stop after this.", "Stopping."),
    ]


def test_image_only_message_is_its_own_turn():
    out = turns([
        user("u-01", "Ask."),
        assistant("a-01", text("Answer.")),
        user("u-02", [{"type": "image", "source": {"type": "base64", "data": "AAAA"}}]),
        assistant("a-02", text("That's a cat.")),
    ])

    assert [(t.user, t.assistant) for t in out] == [("Ask.", "Answer."), ("[image]", "That's a cat.")]
    assert "AAAA" not in out[1].text


def test_unprompted_reply_gets_a_turn_with_no_user_side():
    (t,) = turns([assistant("a-01", text("Background build finished."))])

    assert t.user == "" and t.text == "Assistant: Background build finished."


def test_unknown_and_malformed_records_are_skipped():
    out = turns([
        {"type": "ai-title", "title": "x"},
        {"type": "user", "message": {"content": "no uuid"}},
        {"type": "user", "uuid": "u-07", "message": ["not", "a", "dict"]},
        {"type": "assistant", "uuid": "a-07", "message": "a string"},
        {"type": "user", "uuid": 7, "message": {"content": "numeric uuid"}},
        {"type": "user", "uuid": "u-08", "cwd": ["x"], "timestamp": 5,
         "message": {"content": [{"type": "text", "text": 12}, text("Odd fields.")]}},
        user("u-01", {"unexpected": "shape"}),
        user("u-02", "Kept."),
    ])

    assert [t.user for t in out] == ["Odd fields.", "Kept."]
    assert out[0].cwd == "" and out[0].timestamp == ""


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


def test_strip_injected_removes_nested_blocks_whole():
    raw = "<context>outer<context>inner</context>injected</context>question"
    assert strip_injected(raw) == "question"
    mixed = "<system-reminder>a<context>b</context>c</system-reminder>kept"
    assert strip_injected(mixed) == "kept"


def test_strip_injected_keeps_unclosed_and_stray_tags_as_text():
    assert strip_injected("I typed <context> here") == "I typed <context> here"
    assert strip_injected("stray </context> close") == "stray </context> close"
    assert strip_injected("<context>open <context>x</context> tail") == "<context>open  tail"


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


def test_spool_and_state_dir_are_private(tmp_path):
    t = tmp_path / "s1.jsonl"
    write(t, [user("u-01", "Ask.")])
    old = os.umask(0o022)
    try:
        capture_transcript(t, session_id="s1", config=CFG)
    finally:
        os.umask(old)

    assert stat.S_IMODE(os.stat(capture.state_dir()).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(spool_path()).st_mode) == 0o600


def test_rewritten_transcript_longer_than_the_cursor_restarts(tmp_path):
    t = tmp_path / "s1.jsonl"
    write(t, [user("u-01", "One."), user("u-02", "Two.")])
    capture_transcript(t, session_id="s1", config=CFG)
    write(t, [user("u-07", "Replacement first message, long enough to pass the cursor."),
              user("u-08", "Replacement second.")])

    capture_transcript(t, session_id="s1", config=CFG)

    texts = [r["text"] for r in spooled()]
    assert "User: Replacement first message, long enough to pass the cursor." in texts


def test_replaced_file_with_same_prefix_bytes_restarts_on_inode(tmp_path):
    t = tmp_path / "s1.jsonl"
    write(t, [user("u-01", "One."), user("u-02", "Two.")])
    capture_transcript(t, session_id="s1", config=CFG)
    new = tmp_path / "new.jsonl"
    write(new, [user("u-01", "One."), user("u-02", "Two.")])
    new.replace(t)

    capture_transcript(t, session_id="s1", config=CFG)

    assert [r["text"] for r in spooled()].count("User: One.") == 2


def test_record_with_non_string_uuid_does_not_stall_capture(tmp_path):
    t = tmp_path / "s1.jsonl"
    write(t, [{"type": "user", "uuid": ["bad"], "message": {"content": "x"}},
              {"type": "user", "uuid": {"bad": 1}, "message": {"content": "y"}},
              user("u-01", "Good.")])

    assert capture_transcript(t, session_id="s1", config=CFG) == 1


def test_undecodable_and_broken_lines_are_skipped(tmp_path):
    t = tmp_path / "s1.jsonl"
    t.write_bytes(b'{"type": "user", "text": "\xff\xfe"}\n{not json\n' + b"[" * 100000 + b"\n"
                  + (json.dumps(user("u-01", "Good.")) + "\n").encode())

    assert capture_transcript(t, session_id="s1", config=CFG) == 1


def test_partial_spool_record_is_cut_before_appending(tmp_path):
    capture.ensure_state_dir()
    spool_path().write_text('{"id": "old", "text": "complete"}\n{"id": "half", "te')
    os.chmod(spool_path(), 0o600)
    t = tmp_path / "s1.jsonl"
    write(t, [user("u-01", "Ask.")])

    capture_transcript(t, session_id="s1", config=CFG)

    assert [r.get("text") for r in spooled()] == ["complete", "User: Ask."]


def test_lone_surrogate_does_not_block_capture(tmp_path):
    t = tmp_path / "s1.jsonl"
    t.write_text('{"type": "user", "uuid": "u-01", "message": {"content": "bad \\ud800 char"}}\n'
                 + json.dumps(user("u-02", "Fine.")) + "\n")

    assert capture_transcript(t, session_id="s1", config=CFG) == 2
    assert [r["text"] for r in spooled()] == ["User: bad ? char", "User: Fine."]


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


def test_hook_exits_0_when_the_state_dir_is_unusable(tmp_path, monkeypatch, capsys):
    blocker = tmp_path / "blocked"
    blocker.write_text("a file where the state dir should be")
    monkeypatch.setenv("XDG_STATE_HOME", str(blocker))

    assert run_hook(monkeypatch, json.dumps({"session_id": "s1", "transcript_path": "x"})) == 0
    assert "can't open the capture log" in capsys.readouterr().err


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


# --- security bounds (Codex security pass on #71) ---------------------------------


@pytest.mark.parametrize("wrapped", [
    "_ghp_" + "A" * 36 + "_",
    "*sk-" + "b" * 30 + "*",
    "_AKIA" + "ABCDEFGHIJKLMNOP" + "_",
    "__AIza" + "c" * 35 + "__",
])
def test_redacts_keys_wrapped_in_markdown(wrapped):
    assert "[REDACTED]" in redact(wrapped)


def test_word_containing_sk_dash_is_not_redacted():
    assert redact("a risk-assessment-framework-document") == "a risk-assessment-framework-document"


def test_unterminated_private_key_is_redacted_to_the_end():
    out = redact("before -----BEGIN PRIVATE KEY-----\nMIIsecretmaterial")
    assert out == "before [REDACTED]"


def test_key_straddling_the_clip_point_is_redacted():
    secret = "ghp_" + "Q" * 40
    (t,) = turns([user("u-01", "x" * (capture.CLIP_CHARS - 5) + " " + secret + " tail" * 2000)])

    assert "QQQQQ" not in t.user and "ghp_" not in t.user


def test_secret_at_window_edge_after_an_earlier_redaction_is_caught():
    pem = "-----BEGIN PRIVATE KEY-----\n" + "M" * 1500 + "\n-----END PRIVATE KEY-----"
    token = "ghp_" + "T" * 36
    msg = pem + " " + "p" * (4985 - len(pem) - 2) + " " + token + " end" * 500
    (t,) = turns([user("u-01", msg)])

    assert "TTTTTTTTTT" not in t.user and "ghp_" not in t.user


def test_text_made_of_secrets_redacts_in_linear_time():
    assert _fast(lambda: turns([user("u-01", ("ghp_" + "A" * 36 + " ") * 20000)]))


def _fast(fn, *args, limit=1.0):
    import time
    t0 = time.perf_counter()
    fn(*args)
    return time.perf_counter() - t0 < limit


def test_stripping_many_unclosed_tags_is_linear():
    assert _fast(strip_injected, "<context> " * 50000 + "</interrupt>" * 50000)


def test_stripping_many_incomplete_tags_is_linear():
    assert _fast(strip_injected, "<context " * 50000)


def test_hook_exits_0_when_the_state_dir_is_a_symlink(tmp_path, monkeypatch, capsys):
    real = tmp_path / "real"
    (real / "kipp").mkdir(parents=True)
    (real / "kipp" / "capture").symlink_to(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(real))

    assert run_hook(monkeypatch, json.dumps({"session_id": "s1", "transcript_path": "x"})) == 0
    assert "symlink" in capsys.readouterr().err


def test_redacting_many_unterminated_pem_headers_is_linear():
    assert _fast(redact, "-----BEGIN PRIVATE KEY----- " * 50000)


def test_fifo_transcript_is_refused_without_blocking(tmp_path):
    fifo = tmp_path / "pipe.jsonl"
    os.mkfifo(fifo)
    env = {**os.environ, "PYTHONPATH": str(Path(capture.__file__).parents[1])}

    out = subprocess.run([sys.executable, "-m", "kipp.capture", "--file", str(fifo)],
                         capture_output=True, text=True, timeout=10, env=env)

    assert out.returncode == 1 and "not a regular file" in out.stderr


def test_oversized_line_is_skipped_and_capture_continues(tmp_path, monkeypatch):
    monkeypatch.setattr(capture, "MAX_LINE_BYTES", 1000)
    t = tmp_path / "s1.jsonl"
    write(t, [user("u-01", "x" * 5000), user("u-02", "After the big one.")])

    assert capture_transcript(t, session_id="s1", config=CFG) == 1
    assert spooled()[0]["text"] == "User: After the big one."


def test_oversized_partial_line_waits(tmp_path, monkeypatch):
    monkeypatch.setattr(capture, "MAX_LINE_BYTES", 1000)
    t = tmp_path / "s1.jsonl"
    write(t, [user("u-01", "First.")], partial="y" * 5000)
    capture_transcript(t, session_id="s1", config=CFG)
    with open(t, "a") as f:
        f.write("\n" + json.dumps(user("u-02", "Second.")) + "\n")

    capture_transcript(t, session_id="s1", config=CFG)

    assert [r["text"] for r in spooled()][-1] == "User: Second."


def test_symlinked_spool_is_refused(tmp_path):
    capture.ensure_state_dir()
    target = tmp_path / "elsewhere.txt"
    target.write_text("")
    spool_path().symlink_to(target)
    t = tmp_path / "s1.jsonl"
    write(t, [user("u-01", "Private words.")])

    with pytest.raises(OSError):
        capture_transcript(t, session_id="s1", config=CFG)
    assert target.read_text() == ""


def test_symlinked_log_is_refused(tmp_path, monkeypatch, capsys):
    capture.ensure_state_dir()
    target = tmp_path / "elsewhere.log"
    target.write_text("")
    (capture.state_dir() / "capture.log").symlink_to(target)

    assert run_hook(monkeypatch, "{}") == 0
    assert target.read_text() == ""
    assert "can't open the capture log" in capsys.readouterr().err


def test_symlinked_state_dir_is_refused(tmp_path, monkeypatch):
    real = tmp_path / "real"
    (real / "kipp").mkdir(parents=True)
    (real / "kipp" / "capture").symlink_to(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(real))

    with pytest.raises(ValueError, match="symlink"):
        capture.ensure_state_dir()


# --- security pass 2 -------------------------------------------------------------


def test_a_bearer_token_containing_another_key_is_redacted_whole():
    token = "Bearer head.sk-" + "A" * 30 + "." + "S" * 2000
    assert redact(f"auth: {token} end") == "auth: [REDACTED] end"


def test_overlapping_secret_matches_merge():
    out = redact("x Bearer ghp_" + "a" * 40 + " and sk-" + "b" * 25)
    assert out == "x [REDACTED] and [REDACTED]"


def test_a_long_assistant_run_is_capped_and_counted(monkeypatch):
    monkeypatch.setattr(capture, "MAX_SIDE_CHARS", 10_000)
    records = [user("u-01", "go")] + [assistant(f"a-{i:04d}", text("y" * 1000)) for i in range(50)]
    [t] = list(claude_code_turns(records, session_id="s"))
    # 10 blocks kept (joined with "\n\n"), 40 dropped before cleaning; both counted.
    total = (10 * 1000 + 9 * 2) + 40 * 1000
    assert t.assistant.endswith(f" [...clipped {total - capture.CLIP_CHARS} chars]")


def test_turns_are_spooled_in_batches(tmp_path, monkeypatch):
    monkeypatch.setattr(capture, "SPOOL_BATCH", 3)
    writes = []
    real = capture._append_to_spool
    monkeypatch.setattr(capture, "_append_to_spool", lambda data: (writes.append(data), real(data)))
    t = tmp_path / "s.jsonl"
    records = []
    for i in range(7):
        records += [user(f"u-{i:02d}", f"q{i}"), assistant(f"a-{i:02d}", text(f"r{i}"))]
    write(t, records)

    assert capture_transcript(t, session_id="s", config=CFG) == 7
    assert [w.count("\n") for w in writes] == [3, 3, 1]
    assert [r["user"] for r in spooled()] == [f"q{i}" for i in range(7)]


def test_a_failed_batch_leaves_the_cursor_unmoved(tmp_path, monkeypatch):
    monkeypatch.setattr(capture, "SPOOL_BATCH", 2)
    t = tmp_path / "s.jsonl"
    records = []
    for i in range(5):
        records += [user(f"u-{i:02d}", f"q{i}"), assistant(f"a-{i:02d}", text(f"r{i}"))]
    write(t, records)
    real, calls = capture._append_to_spool, []

    def flaky(data):
        calls.append(data)
        if len(calls) == 2:
            raise OSError("disk full")
        real(data)

    monkeypatch.setattr(capture, "_append_to_spool", flaky)
    with pytest.raises(OSError):
        capture_transcript(t, session_id="s", config=CFG)
    monkeypatch.setattr(capture, "_append_to_spool", real)

    assert capture_transcript(t, session_id="s", config=CFG) == 5
    assert sorted({r["user"] for r in spooled()}) == [f"q{i}" for i in range(5)]


def test_a_group_writable_kipp_dir_is_tightened(tmp_path, monkeypatch):
    base = tmp_path / "st"
    (base / "kipp").mkdir(parents=True)
    os.chmod(base / "kipp", 0o777)
    monkeypatch.setenv("XDG_STATE_HOME", str(base))

    d = capture.ensure_state_dir()
    assert stat.S_IMODE(os.stat(base / "kipp").st_mode) == 0o755
    assert stat.S_IMODE(os.stat(d).st_mode) == 0o700


def test_a_world_writable_state_base_is_refused(tmp_path, monkeypatch):
    base = tmp_path / "st"
    base.mkdir()
    os.chmod(base, 0o777)
    monkeypatch.setenv("XDG_STATE_HOME", str(base))

    with pytest.raises(ValueError, match="world-writable"):
        capture.ensure_state_dir()


def test_a_sticky_world_writable_base_is_allowed(tmp_path, monkeypatch):
    base = tmp_path / "st"
    base.mkdir()
    os.chmod(base, 0o1777)
    monkeypatch.setenv("XDG_STATE_HOME", str(base))

    assert capture.ensure_state_dir() == base / "kipp" / "capture"


def test_a_kipp_dir_that_is_a_symlink_is_refused(tmp_path, monkeypatch):
    base = tmp_path / "st"
    base.mkdir()
    (tmp_path / "elsewhere").mkdir()
    (base / "kipp").symlink_to(tmp_path / "elsewhere")
    monkeypatch.setenv("XDG_STATE_HOME", str(base))

    with pytest.raises(ValueError, match="symlink"):
        capture.ensure_state_dir()


def test_hook_exits_0_on_a_world_writable_state_base(tmp_path, monkeypatch, capsys):
    base = tmp_path / "st"
    base.mkdir()
    os.chmod(base, 0o777)
    monkeypatch.setenv("XDG_STATE_HOME", str(base))

    assert run_hook(monkeypatch, "{}") == 0
    assert "world-writable" in capsys.readouterr().err


def test_a_hard_linked_spool_is_refused(tmp_path):
    capture.ensure_state_dir()
    outside = tmp_path / "outside.txt"
    outside.write_text("")
    os.link(outside, spool_path())
    t = tmp_path / "s.jsonl"
    write(t, [user("u-01", "private prose")])

    with pytest.raises(ValueError, match="private regular file"):
        capture_transcript(t, session_id="s", config=CFG)
    assert outside.read_text() == ""


def test_an_existing_loose_spool_is_refused(tmp_path):
    capture.ensure_state_dir()
    spool_path().write_text("")
    os.chmod(spool_path(), 0o644)
    t = tmp_path / "s.jsonl"
    write(t, [user("u-01", "hi")])

    with pytest.raises(ValueError, match="readable or writable by others"):
        capture_transcript(t, session_id="s", config=CFG)
    assert spool_path().read_text() == ""


def test_a_loose_capture_dir_is_refused(tmp_path):
    d = capture.ensure_state_dir()
    os.chmod(d, 0o750)

    with pytest.raises(ValueError, match="accessible to other users"):
        capture.ensure_state_dir()


def test_a_fifo_capture_log_does_not_hang_the_hook(tmp_path):
    state = tmp_path / "state"
    d = state / "kipp" / "capture"
    d.mkdir(parents=True, mode=0o700)
    os.mkfifo(d / "capture.log")
    env = {**os.environ, "PYTHONPATH": str(Path(capture.__file__).parents[1]),
           "XDG_STATE_HOME": str(state)}

    out = subprocess.run([sys.executable, "-m", "kipp.capture", "--hook"], input="{}",
                         capture_output=True, text=True, timeout=10, env=env)

    assert out.returncode == 0 and "can't open the capture log" in out.stderr


def test_a_cut_at_the_side_cap_leaves_no_key_fragment(monkeypatch):
    monkeypatch.setattr(capture, "MAX_SIDE_CHARS", 20_000)
    pem = "-----BEGIN PRIVATE KEY-----\n" + "M" * 19_900 + "\n-----END PRIVATE KEY-----"
    token = "ghp_" + "Z" * 40
    records = [user("u-01", "go"), assistant("a-01", text(pem[:19_980 - 30] + pem[-25:] + " " + token))]
    [t] = list(claude_code_turns(records, session_id="s"))
    assert "ZZZZ" not in t.assistant and "ghp_" not in t.assistant


def test_a_hard_linked_cursor_tmp_is_refused_without_truncating(tmp_path):
    t = tmp_path / "s.jsonl"
    write(t, [user("u-01", "q1"), assistant("a-01", text("r1")), user("u-02", "q2")])
    cursor = capture._cursor_path("claude-code", "s")
    capture.ensure_state_dir()
    cursor.parent.mkdir(mode=0o700)
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me")
    os.link(victim, cursor.with_suffix(".tmp"))

    with pytest.raises(ValueError, match="private regular file"):
        capture_transcript(t, session_id="s", config=CFG)
    assert victim.read_text() == "keep me"


def test_a_long_injected_block_is_stripped_before_the_side_cap(monkeypatch):
    monkeypatch.setattr(capture, "MAX_SIDE_CHARS", 1000)
    block = "<context>" + "c" * 5000 + "</context>The actual answer."
    [t] = list(claude_code_turns([user("u-01", "go"), assistant("a-01", text(block))], session_id="s"))
    assert t.assistant == "The actual answer."


def test_an_unencodable_uuid_is_skipped_and_capture_continues(tmp_path):
    t = tmp_path / "s.jsonl"
    bad = json.dumps(user("u-00", "lost")).replace('"u-00"', '"\\ud800"')
    t.write_text(bad + "\n" + "".join(json.dumps(r) + "\n" for r in
                                      [user("u-01", "q1"), assistant("a-01", text("r1")), user("u-02", "q2")]))

    assert capture_transcript(t, session_id="s", config=CFG) == 2
    assert [r["user"] for r in spooled()] == ["q1", "q2"]


def test_the_hook_gives_up_on_a_busy_lock(tmp_path, monkeypatch, capsys):
    import fcntl
    monkeypatch.setattr(capture, "HOOK_LOCK_TIMEOUT", 0.2)
    t = tmp_path / "s.jsonl"
    write(t, [user("u-01", "q1")])
    lock = os.fdopen(os.open(capture.ensure_state_dir() / "capture.lock", os.O_RDWR | os.O_CREAT, 0o600), "r+b")
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        payload = json.dumps({"transcript_path": str(t), "session_id": "s"})
        assert run_hook(monkeypatch, payload) == 0
    finally:
        lock.close()
    assert spooled() == []
    assert "capture lock busy" in (capture.state_dir() / "capture.log").read_text()

    assert run_hook(monkeypatch, payload) == 0
    assert [r["user"] for r in spooled()] == ["q1"]


def test_stripping_a_block_cannot_glue_a_key_to_a_word():
    key = "ghp_" + "A" * 36
    assert strip_injected(f"check<context>x</context>{key}", ("context",)) == f"check {key}"
    [t] = list(claude_code_turns([user("u-01", f"check<context>x</context>{key}")], session_id="s"))
    assert t.user == "check [REDACTED]"


def test_a_bearer_split_across_blocks_is_redacted_whole_at_the_side_cap(monkeypatch):
    monkeypatch.setattr(capture, "MAX_SIDE_CHARS", 5000)
    token = "head.sk-" + "A" * 30 + "." + "S" * 2000
    records = [user("u-01", "go"), assistant("a-01", text("Bearer"), text(token + " " + "p" * 6000))]
    [t] = list(claude_code_turns(records, session_id="s"))
    assert "SSSS" not in t.assistant and "AAAA" not in t.assistant


def test_an_injected_block_split_across_blocks_is_dropped_at_the_side_cap(monkeypatch):
    monkeypatch.setattr(capture, "MAX_SIDE_CHARS", 3000)
    records = [user("u-01", "go"),
               assistant("a-01", text("Intro. <context>" + "private " * 500), text("</context>answer"))]
    [t] = list(claude_code_turns(records, session_id="s"))
    assert "private" not in t.assistant and "<context>" not in t.assistant
    assert t.assistant.startswith("Intro.")


def test_the_side_cap_cuts_on_whitespace():
    turn = capture._new_turn("a", {})
    capture.MAX_SIDE_CHARS, saved = 10, capture.MAX_SIDE_CHARS
    try:
        capture._add(turn, "assistant", ["abc defghijklmnop", "more"])
    finally:
        capture.MAX_SIDE_CHARS = saved
    assert turn["assistant"] == ["abc"]
    assert turn["dropped"]["assistant"] == len("abc defghijklmnop") - 3 + 4


def test_a_fifo_cursor_does_not_hang_the_hook(tmp_path):
    state = tmp_path / "state"
    cursors = state / "kipp" / "capture" / "cursors"
    cursors.mkdir(parents=True)
    for d in (cursors, cursors.parent):
        os.chmod(d, 0o700)
    os.mkfifo(cursors / "claude-code-s1.json")
    t = tmp_path / "s1.jsonl"
    write(t, [user("u-01", "q1")])
    env = {**os.environ, "PYTHONPATH": str(Path(capture.__file__).parents[1]),
           "XDG_STATE_HOME": str(state)}
    payload = json.dumps({"transcript_path": str(t), "session_id": "s1"})

    out = subprocess.run([sys.executable, "-m", "kipp.capture", "--hook"], input=payload,
                         capture_output=True, text=True, timeout=10, env=env)
    assert out.returncode == 0


def test_a_huge_cursor_is_ignored(tmp_path):
    capture.ensure_state_dir()
    cursor = capture._cursor_path("claude-code", "s")
    cursor.parent.mkdir(mode=0o700)
    cursor.write_text(" " * (capture._CURSOR_MAX_BYTES + 1))
    t = tmp_path / "s.jsonl"
    write(t, [user("u-01", "q1")])

    assert capture_transcript(t, session_id="s", config=CFG) == 1


def test_turns_carry_their_anchor_offset_and_capture_resumes_there(tmp_path):
    t = tmp_path / "s.jsonl"
    records = [user("u-00", "q0")]
    for i in range(1, 6):
        records += [assistant(f"a-{i:02d}", {"type": "tool_use", "id": f"t{i}", "name": "x", "input": {}}),
                    user(f"r-{i:02d}", [{"type": "tool_result", "tool_use_id": f"t{i}", "content": "ok"}])]
    records += [user("u-01", "q1"), assistant("a-99", text("done"))]
    write(t, records)
    anchor_at = t.read_bytes().index(b'"uuid": "u-01"')
    line_start = t.read_bytes().rindex(b"\n", 0, anchor_at) + 1

    assert capture_transcript(t, session_id="s", config=CFG) == 2
    cursor = json.loads(capture._cursor_path("claude-code", "s").read_text())
    assert cursor["offset"] == line_start
    assert capture_transcript(t, session_id="s", config=CFG) == 1
    assert all("offset" not in r and capture.OFFSET_KEY not in r for r in spooled())


def test_a_time_budget_spreads_a_backlog_over_runs(tmp_path, monkeypatch):
    clock = iter(range(1_000_000))
    monkeypatch.setattr(capture.time, "monotonic", lambda: next(clock))
    t = tmp_path / "s.jsonl"
    records = []
    for i in range(10):
        records += [user(f"u-{i:02d}", f"q{i}"), assistant(f"a-{i:02d}", text(f"r{i}"))]
    write(t, records)

    runs = 0
    while {r["user"] for r in spooled()} != {f"q{i}" for i in range(10)}:
        capture_transcript(t, session_id="s", config=CFG, time_budget=5)
        runs += 1
        assert runs < 20
    assert runs > 1
    last = [r for r in spooled() if r["user"] == "q9"][-1]
    assert last["assistant"] == "r9"


def test_an_oversized_line_past_any_old_byte_budget_is_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(capture, "MAX_LINE_BYTES", 1000)
    t = tmp_path / "s.jsonl"
    write(t, [user("u-01", "x" * 3_000_000), user("u-02", "after")])

    assert capture_transcript(t, session_id="s", config=CFG, time_budget=10) == 1
    assert spooled()[0]["user"] == "after"


def test_absurd_tag_nesting_is_bounded():
    text_ = "<context>" * 100_000 + "kept"
    assert strip_injected(text_, ("context",)).endswith("kept")
    closed = "<context>" * 10 + "x" + "</context>" * 10 + "after"
    assert strip_injected(closed, ("context",)) == "after"


def test_oversized_metadata_is_dropped():
    r = user("u-01", "q")
    r["timestamp"] = "9" * 10_000
    r["cwd"] = "/x" * 5000
    [t] = list(claude_code_turns([r, user("u" * 300, "q2")], session_id="s"))
    assert t.timestamp == "" and t.cwd == ""


def test_spool_batches_are_bounded_by_size(tmp_path, monkeypatch):
    monkeypatch.setattr(capture, "SPOOL_BATCH_CHARS", 10_000)
    writes = []
    real = capture._append_to_spool
    monkeypatch.setattr(capture, "_append_to_spool", lambda data: (writes.append(data), real(data)))
    t = tmp_path / "s.jsonl"
    records = []
    for i in range(6):
        records += [user(f"u-{i:02d}", "y " * 1800), assistant(f"a-{i:02d}", text("ok"))]
    write(t, records)

    assert capture_transcript(t, session_id="s", config=CFG) == 6
    assert len(writes) >= 3 and all(len(w) < 20_000 for w in writes)


@pytest.mark.parametrize("cwd", ["~\0x", "~\ud800"])
def test_a_malformed_cwd_uses_the_default_domain(cwd):
    assert resolve_domain(cwd, {"~/dev": "dev"}, "personal") == "personal"


def test_a_turn_longer_than_the_budget_does_not_stall_capture(tmp_path, monkeypatch):
    clock = iter(range(1_000_000))
    monkeypatch.setattr(capture.time, "monotonic", lambda: next(clock))
    t = tmp_path / "s.jsonl"
    records = [user("u-00", "q0")]
    for i in range(20):  # one turn far longer than the 5-tick budget
        records += [assistant(f"a-{i:02d}", {"type": "tool_use", "id": f"t{i}", "name": "x", "input": {}}),
                    user(f"r-{i:02d}", [{"type": "tool_result", "tool_use_id": f"t{i}", "content": "ok"}])]
    records += [assistant("a-99", text("final reply")), user("u-01", "q1"), assistant("a-98", text("r1"))]
    write(t, records)

    for _ in range(30):
        capture_transcript(t, session_id="s", config=CFG, time_budget=5)
    texts = " ".join(r["text"] for r in spooled())
    # The tail of the over-long turn is dropped; capture moves on past it.
    assert "q0" in texts and "q1" in texts and "r1" in texts
    assert "final reply" not in texts


def test_a_key_split_by_a_forced_checkpoint_is_not_spooled(tmp_path, monkeypatch):
    clock = iter(range(1_000_000))
    monkeypatch.setattr(capture.time, "monotonic", lambda: next(clock))
    t = tmp_path / "s.jsonl"
    records = [user("u-00", "q0"), assistant("a-00", text("-----BEGIN PRIVATE KEY-----"))]
    for i in range(10):
        records += [assistant(f"a-{i + 1:02d}", text(f"KEYMATERIAL{i}"))]
    records += [assistant("a-90", text("-----END PRIVATE KEY----- done")),
                user("u-01", "q1"), assistant("a-91", text("r1"))]
    write(t, records)

    for _ in range(30):
        capture_transcript(t, session_id="s", config=CFG, time_budget=4)
    texts = " ".join(r["text"] for r in spooled())
    assert "KEYMATERIAL" not in texts
    assert "q1" in texts and "r1" in texts


def test_skipping_an_oversized_line_keeps_its_progress_across_runs(tmp_path, monkeypatch):
    clock = iter(range(1_000_000))
    monkeypatch.setattr(capture.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(capture, "MAX_LINE_BYTES", 1000)
    t = tmp_path / "s.jsonl"
    write(t, [user("u-01", "x" * (30 << 20)), user("u-02", "after"), assistant("a-02", text("r"))])

    for _ in range(40):
        capture_transcript(t, session_id="s", config=CFG, time_budget=5)
        if any(r["user"] == "after" for r in spooled()):
            break
    else:
        pytest.fail("capture never got past the oversized line")
