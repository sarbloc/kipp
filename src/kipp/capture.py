"""Capture conversation turns from agent transcripts into a local spool (issue #71).

An MCP server only sees the tool calls made to it, so KIPP can't capture a
conversation from the server side. Instead a client hook (Claude Code's
``Stop`` and ``SessionEnd``) runs ``kipp-capture --hook``, which reads the new
part of the session transcript and appends one record per turn to a spool
file. ``kipp capture --flush`` later embeds the spool into the events
collections.

This module is the hook path, so it imports only the standard library: no
Qdrant, no embedder. ``kipp`` itself takes most of a second to start; this
takes tens of milliseconds.

What a turn keeps: the user's message and the assistant's prose. Tool calls,
tool output, thinking, meta entries, compaction summaries and subagent
(sidechain) messages are dropped, and injected context blocks such as
``<system-reminder>`` are stripped. On real transcripts that leaves under 1%
of the file. Known secret formats are redacted and each side is clipped.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import os
import re
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator

from kipp.config import DOMAINS, load_config

log = logging.getLogger("kipp.capture")

# Stable namespace for turn ids: uuid5(NAMESPACE, "<client>:<session>:<anchor>").
# Never change it, or re-captured turns stop deduplicating against old ones.
TURN_NAMESPACE = uuid.UUID("6f1f3c2e-9b3a-4c39-8a51-2b7d0c6f4e11")

CLIP_CHARS = 4000

# Blocks a harness injects into user messages. Stripped with their contents.
# Override with config ``capture.strip_tags``.
DEFAULT_STRIP_TAGS = (
    "system-reminder",
    "command-name",
    "command-message",
    "command-args",
    "local-command-stdout",
    "local-command-caveat",
    "user-prompt-submit-hook",
    # Endurance's per-turn injections.
    "retrieved_memory",
    "context",
    "interrupt",
)

# Best effort only: a format not listed here is stored as-is.
_SECRET_PATTERNS = [
    re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S,
    ),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{35}"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{20,}"),
]
REDACTED = "[REDACTED]"

# Records Claude Code marks as not part of the visible dialogue.
_SKIP_FLAGS = ("isSidechain", "isMeta", "isCompactSummary", "isApiErrorMessage",
               "isVisibleInTranscriptOnly")


@dataclass
class Turn:
    """One captured exchange: a user message and the assistant's reply text."""

    id: str
    anchor: str  # transcript id of the record that opened the turn
    client: str
    session_id: str
    timestamp: str
    cwd: str
    user: str
    assistant: str
    domain: str = "shared"

    @property
    def text(self) -> str:
        parts = []
        if self.user:
            parts.append(f"User: {self.user}")
        if self.assistant:
            parts.append(f"Assistant: {self.assistant}")
        return "\n\n".join(parts)

    def spool_record(self) -> dict:
        return {**asdict(self), "text": self.text}


# --- text cleaning -------------------------------------------------------------


def strip_injected(text: str, tags: Iterable[str] = DEFAULT_STRIP_TAGS) -> str:
    """Remove injected ``<tag ...>...</tag>`` blocks and collapse the leftover blank lines."""
    for tag in tags:
        text = re.sub(rf"<{re.escape(tag)}(\s[^>]*)?>.*?</{re.escape(tag)}>", "", text, flags=re.S)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def redact(text: str) -> str:
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(REDACTED, text)
    return text


def clip(text: str, limit: int = CLIP_CHARS) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]} [...clipped {len(text) - limit} chars]"


def clean(text: str, tags: Iterable[str]) -> str:
    # Redact before clipping, so a clip can't cut a secret in half and leave
    # a fragment the patterns no longer match.
    return clip(redact(strip_injected(text, tags)))


# --- Claude Code transcripts ---------------------------------------------------


def _text_blocks(content) -> list[str]:
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        return [b["text"] for b in content
                if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str)]
    return []


def _str(value) -> str:
    return value if isinstance(value, str) else ""


def _is_tool_result(content) -> bool:
    return isinstance(content, list) and any(
        isinstance(b, dict) and b.get("type") == "tool_result" for b in content
    )


def claude_code_turns(
    records: Iterable[dict], *, session_id: str, cwd: str = "",
    tags: Iterable[str] = DEFAULT_STRIP_TAGS,
) -> Iterator[Turn]:
    """Group Claude Code transcript records into turns.

    A turn opens at each user message with text left after stripping, and
    collects the assistant's text blocks until the next one. Assistant text
    with no open turn (a reply the user didn't prompt, e.g. after a background
    task) opens a turn with an empty user side. The transcript format is
    undocumented, so unknown or malformed records are skipped, never fatal.
    """
    tags = tuple(tags)
    turn: dict | None = None

    def finish(t: dict | None) -> Turn | None:
        if t is None:
            return None
        user = clean("\n\n".join(t["user"]), tags)
        assistant = clean("\n\n".join(t["assistant"]), tags)
        if not user and not assistant:
            return None
        return Turn(
            id=str(uuid.uuid5(TURN_NAMESPACE, f"claude-code:{session_id}:{t['anchor']}")),
            anchor=t["anchor"], client="claude-code", session_id=session_id, timestamp=t["timestamp"],
            cwd=t["cwd"] or cwd, user=user, assistant=assistant,
        )

    for r in records:
        kind = r.get("type")
        if kind not in ("user", "assistant") or any(r.get(f) for f in _SKIP_FLAGS):
            continue
        message = r.get("message")
        anchor = r.get("uuid")
        if not isinstance(message, dict) or not isinstance(anchor, str) or not anchor:
            continue
        content = message.get("content")
        if kind == "user":
            if _is_tool_result(content):
                continue
            texts = [strip_injected(t, tags) for t in _text_blocks(content)]
            texts = [t for t in texts if t]
            if not texts:
                continue
            done = finish(turn)
            if done:
                yield done
            turn = {"anchor": anchor, "timestamp": _str(r.get("timestamp")),
                    "cwd": _str(r.get("cwd")), "user": texts, "assistant": []}
        else:
            texts = [t for t in _text_blocks(content) if t.strip() and t.strip() != "(no content)"]
            if not texts:
                continue
            if turn is None:
                turn = {"anchor": anchor, "timestamp": _str(r.get("timestamp")),
                        "cwd": _str(r.get("cwd")), "user": [], "assistant": []}
            turn["assistant"].extend(texts)
    done = finish(turn)
    if done:
        yield done


ADAPTERS: dict[str, Callable[..., Iterator[Turn]]] = {"claude-code": claude_code_turns}


# --- reading new transcript lines ---------------------------------------------


def read_new_records(path: Path, offset: int) -> tuple[list[tuple[int, dict]], int]:
    """Parse the complete JSONL lines after ``offset``.

    Returns ``(start_offset, record)`` pairs and the offset just past the last
    complete line. A trailing line without a newline is still being written,
    so it's left for the next run. The caller validates ``offset`` against the
    cursor's fingerprint first (``_read_cursor``).
    """
    size = path.stat().st_size
    if offset > size:
        offset = 0
    with open(path, "rb") as f:
        f.seek(offset)
        chunk = f.read()
    end = chunk.rfind(b"\n")
    if end < 0:
        return [], offset
    records = []
    pos = offset
    for line in chunk[: end + 1].split(b"\n")[:-1]:
        start, pos = pos, pos + len(line) + 1
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            log.warning("skipping malformed transcript line at byte %d in %s", start, path)
            continue
        if isinstance(rec, dict):
            records.append((start, rec))
    return records, offset + end + 1


# --- state: cursors, spool, lock -----------------------------------------------


def state_dir() -> Path:
    xdg = os.environ.get("XDG_STATE_HOME")
    base = Path(xdg) if xdg and os.path.isabs(xdg) else Path.home() / ".local" / "state"
    return base / "kipp" / "capture"


def ensure_state_dir() -> Path:
    """Create the capture directory, private to the user: it holds conversation text."""
    d = state_dir()
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(d, 0o700)  # mkdir's mode is masked by umask and skipped if it exists
    return d


def spool_path() -> Path:
    return state_dir() / "spool.jsonl"


def _cursor_path(client: str, session_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{client}-{session_id}")
    return state_dir() / "cursors" / f"{safe}.json"


_FINGERPRINT_BYTES = 256


def _fingerprint(path: Path, offset: int) -> str:
    """Hash of the bytes just before ``offset``. Appends never change them; a rewrite does."""
    start = max(0, offset - _FINGERPRINT_BYTES)
    with open(path, "rb") as f:
        f.seek(start)
        return hashlib.sha256(f.read(offset - start)).hexdigest()


def _read_cursor(client: str, session_id: str, transcript: Path) -> int:
    """The saved offset, or 0 when there is none or the transcript was replaced or rewritten.

    Restarting is always safe: turn ids are stable, so the flush deduplicates.
    """
    p = _cursor_path(client, session_id)
    try:
        data = json.loads(p.read_text())
    except FileNotFoundError:
        return 0
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("unreadable cursor %s (%s); recapturing from the start", p, exc)
        return 0
    offset = data.get("offset") if isinstance(data, dict) else None
    if not isinstance(offset, int) or offset < 0 or data.get("path") != str(transcript):
        return 0
    st = transcript.stat()
    if (data.get("inode") != st.st_ino or offset > st.st_size
            or data.get("fingerprint") != _fingerprint(transcript, offset)):
        log.info("transcript %s changed since the last capture; recapturing from the start",
                 transcript)
        return 0
    return offset


def _write_cursor(client: str, session_id: str, transcript: Path, offset: int) -> None:
    p = _cursor_path(client, session_id)
    p.parent.mkdir(exist_ok=True, mode=0o700)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({
        "path": str(transcript), "offset": offset, "inode": transcript.stat().st_ino,
        "fingerprint": _fingerprint(transcript, offset),
    }))
    tmp.replace(p)


class _Lock:
    """Exclusive lock over the whole read-cursor → append → write-cursor sequence.

    Stop and SessionEnd can fire close together, and several sessions share
    one spool, so captures are serialised.
    """

    def __enter__(self):
        self._f = open(ensure_state_dir() / "capture.lock", "w")
        fcntl.flock(self._f, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        fcntl.flock(self._f, fcntl.LOCK_UN)
        self._f.close()


# --- domain routing ------------------------------------------------------------


def resolve_domain(cwd: str, routes: dict[str, str], default: str = "shared") -> str:
    """Pick the domain for a working directory: the longest matching route prefix wins.

    ``routes`` maps directory prefixes (``~`` expanded) to domains, e.g.
    ``{"~/dev": "dev"}``. A prefix matches whole path components only, so
    ``~/dev`` doesn't match ``~/devices``.
    """
    if not cwd:
        return default
    path = os.path.normpath(os.path.expanduser(cwd))
    best, best_len = default, -1
    for prefix, domain in routes.items():
        root = os.path.normpath(os.path.expanduser(prefix))
        if (path == root or path.startswith(root.rstrip(os.sep) + os.sep)) and len(root) > best_len:
            best, best_len = domain, len(root)
    return best


@dataclass
class CaptureConfig:
    routes: dict[str, str]
    default_domain: str
    strip_tags: tuple[str, ...]

    @classmethod
    def load(cls) -> CaptureConfig:
        raw = load_config().get("capture", {}) or {}
        routes = dict(raw.get("domains", {}) or {})
        default = raw.get("default_domain", "shared")
        for domain in [*routes.values(), default]:
            if domain not in DOMAINS:
                raise ValueError(
                    f"capture config names unknown domain {domain!r}; expected one of {DOMAINS}"
                )
        tags = tuple(raw.get("strip_tags", DEFAULT_STRIP_TAGS))
        return cls(routes=routes, default_domain=default, strip_tags=tags)


# --- the capture operation -------------------------------------------------------


def capture_transcript(
    transcript: Path, *, session_id: str, cwd: str = "", client: str = "claude-code",
    config: CaptureConfig | None = None,
) -> int:
    """Append the turns added to ``transcript`` since the last capture to the spool.

    Returns the number of turns spooled. The cursor advances only after the
    spool write succeeds, so a failure re-captures next time rather than
    skipping turns.

    The last turn may still be growing when the hook fires (its reply written
    after the hook ran, or more of it on the next turn). So the cursor stops at
    the start of the last turn, and the next capture spools it again, complete,
    under the same id. The flush keeps the last version of each id.
    """
    config = config or CaptureConfig.load()
    adapter = ADAPTERS[client]
    with _Lock():
        offset = _read_cursor(client, session_id, transcript)
        located, end_offset = read_new_records(transcript, offset)
        starts = {rec["uuid"]: start for start, rec in located if isinstance(rec.get("uuid"), str)}
        records = [rec for _, rec in located]
        turns = list(adapter(records, session_id=session_id, cwd=cwd, tags=config.strip_tags))
        new_offset = starts[turns[-1].anchor] if turns else end_offset
        if turns:
            lines = []
            for t in turns:
                t.domain = resolve_domain(t.cwd, config.routes, config.default_domain)
                lines.append(json.dumps(t.spool_record(), ensure_ascii=False) + "\n")
            fd = os.open(spool_path(), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as f:
                f.write("".join(lines))
                f.flush()
                os.fsync(f.fileno())
        if new_offset != offset:
            _write_cursor(client, session_id, transcript, new_offset)
    return len(turns)


# --- entry point -----------------------------------------------------------------


def _setup_logging(hook: bool) -> None:
    handler: logging.Handler = logging.StreamHandler(sys.stderr)
    if hook:
        # A hook's stderr isn't shown to the operator, so errors go to a file,
        # falling back to stderr if the state directory isn't writable.
        try:
            handler = logging.FileHandler(ensure_state_dir() / "capture.log")
        except OSError as exc:
            print(f"kipp-capture: can't open the capture log ({exc}); logging to stderr",
                  file=sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    for old in list(log.handlers):  # main() can run more than once in one process
        log.removeHandler(old)
        old.close()
    log.addHandler(handler)
    log.setLevel(logging.INFO)


def _run_hook(stdin_text: str) -> int:
    payload = json.loads(stdin_text)
    transcript = payload.get("transcript_path")
    session_id = payload.get("session_id")
    if not transcript or not session_id:
        log.error("hook payload missing transcript_path or session_id: keys=%s",
                  sorted(payload))
        return 0
    n = capture_transcript(Path(transcript), session_id=session_id, cwd=payload.get("cwd", ""))
    log.info("captured %d turns from %s (%s, %s)", n, transcript, session_id,
             payload.get("hook_event_name", "?"))
    return n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kipp-capture",
        description="Spool conversation turns from an agent transcript for KIPP.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--hook", action="store_true",
                      help="Read a Claude Code hook payload (Stop/SessionEnd) from stdin.")
    mode.add_argument("--file", type=Path, help="Capture a transcript file directly.")
    parser.add_argument("--format", choices=sorted(ADAPTERS), default="claude-code")
    parser.add_argument("--session-id", help="Session id for --file (default: the file name).")
    parser.add_argument("--cwd", default="", help="Working directory for --file domain routing.")
    args = parser.parse_args(argv)
    _setup_logging(args.hook)

    if args.hook:
        # Never fail or block the agent's turn: log and exit 0 whatever happens.
        try:
            _run_hook(sys.stdin.read())
        except Exception:
            log.exception("capture hook failed at %s", datetime.now(timezone.utc).isoformat())
        return 0

    session_id = args.session_id or args.file.stem
    try:
        n = capture_transcript(args.file, session_id=session_id, cwd=args.cwd, client=args.format)
    except (OSError, ValueError) as exc:
        log.error("capture failed for %s: %s", args.file, exc)
        return 1
    print(f"Spooled {n} turns from {args.file} to {spool_path()}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
