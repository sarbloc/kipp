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
import stat
import sys
import time
import uuid
from collections import Counter
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

# Best effort only: a format not listed here is stored as-is. A key may follow
# any non-alphanumeric character, including Markdown's `_` and `*`. Every
# pattern is a fixed prefix plus one character class, so matching stays linear.
_B = r"(?<![A-Za-z0-9])"
_SECRET_PATTERNS = [
    re.compile(_B + r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(_B + r"gh[pousr]_[A-Za-z0-9]{36,}"),
    re.compile(_B + r"github_pat_[A-Za-z0-9_]{22,}"),
    re.compile(_B + r"AKIA[0-9A-Z]{16}(?![A-Za-z0-9])"),
    re.compile(_B + r"AIza[0-9A-Za-z_-]{35}"),
    re.compile(_B + r"xox[abprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"(?i)" + _B + r"bearer\s+[A-Za-z0-9._~+/=-]{20,}"),
]
_PEM_BEGIN = re.compile(r"-----BEGIN [A-Z0-9 ]{0,40}PRIVATE KEY-----")
_PEM_END = re.compile(r"-----END [A-Z0-9 ]{0,40}PRIVATE KEY-----")
REDACTED = "[REDACTED]"
MAX_TAG_DEPTH = 64  # open injected blocks tracked at once; deeper tags are left as text
# Transcript metadata (timestamp, cwd) longer than this is dropped, and ids
# longer than MAX_ID_CHARS make a record unusable: real values are far shorter.
MAX_META_CHARS = 4096
MAX_ID_CHARS = 256
# Key the capture loop adds to each transcript record: its byte offset.
OFFSET_KEY = "_kipp_offset"
# Redaction sees this much text past the clip point, so a key that starts
# before the clip is matched whole. Longer than any single-line key.
_REDACT_MARGIN = 1024
# Raw text kept per side of a turn before cleaning. Far past the clip, so
# stripping and redaction still see whole blocks and keys; caps memory when an
# assistant reply runs to thousands of records.
MAX_SIDE_CHARS = 1 << 20
# A transcript line longer than this is skipped unread, so one huge tool
# output can't exhaust memory. Dialogue is clipped far below it anyway.
MAX_LINE_BYTES = 16 << 20

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
    # Byte offset of the anchor record in the transcript, when read from a
    # file (``OFFSET_KEY`` on the record); where the next capture resumes.
    offset: int | None = None

    @property
    def text(self) -> str:
        parts = []
        if self.user:
            parts.append(f"User: {self.user}")
        if self.assistant:
            parts.append(f"Assistant: {self.assistant}")
        return "\n\n".join(parts)

    def spool_record(self) -> dict:
        record = asdict(self)
        del record["offset"]
        return {**record, "text": self.text}


# --- text cleaning -------------------------------------------------------------


def strip_injected(text: str, tags: Iterable[str] = DEFAULT_STRIP_TAGS, *,
                   truncated: bool = False) -> str:
    """Remove injected ``<tag ...>...</tag>`` blocks and collapse the leftover blank lines.

    Nesting is balanced, so a block quoting another block of the same tag is
    removed whole. An opening tag that's never closed is kept as literal text
    (the user may simply have typed it), and so is a stray closing tag. When
    ``truncated``, the text was cut short and the closing tag may have been
    cut with it, so an unclosed block is dropped to the end instead. One pass
    over the tags, so the cost is linear in the text.
    """
    names = "|".join(re.escape(t) for t in tags)
    if not names:
        return text.strip()
    # Attributes are capped so an unterminated `<tag ` can't make each match
    # rescan the rest of the text.
    token = re.compile(rf"<(/?)({names})(?:\s[^<>]{{0,200}})?>")
    drops: list[tuple[int, int]] = []
    # Open blocks: (tag, start, closed regions inside it). A region is only
    # final once its outermost block closes, or when its parent never does.
    stack: list[tuple[str, int, list[tuple[int, int]]]] = []
    open_count: Counter[str] = Counter()
    for m in token.finditer(text):
        name = m.group(2)
        if not m.group(1):
            if len(stack) >= MAX_TAG_DEPTH:
                continue  # absurd nesting: treat the tag as text rather than track it
            stack.append((name, m.start(), []))
            open_count[name] += 1
            continue
        if not open_count[name]:
            continue  # stray closing tag: literal text
        while True:  # blocks opened inside this one but never closed go with it
            top, top_start, _ = stack.pop()
            open_count[top] -= 1
            if top == name:
                break
        (stack[-1][2] if stack else drops).append((top_start, m.end()))
    if truncated and stack:
        drops.append((stack[0][1], len(text)))  # covers every block opened inside it
    else:
        for _, _, regions in stack:  # unclosed blocks stay as text; their closed children don't
            drops.extend(regions)
    out, i = [], 0
    for a, b in sorted(drops):
        if b <= i:
            continue
        a = max(a, i)
        out.append(text[i:a])
        # A space where a block sat between two words, so dropping it can't
        # glue text together (``check<context>x</context>ghp_...`` must not
        # become ``checkghp_...``, which the redaction boundary would miss).
        if a > 0 and b < len(text) and not text[a - 1].isspace() and not text[b].isspace():
            out.append(" ")
        i = b
    out.append(text[i:])
    return re.sub(r"\n{3,}", "\n\n", "".join(out)).strip()


def _redact_pem(text: str) -> str:
    """Replace private-key blocks. One with no END marker is redacted to the end of the text."""
    out, i = [], 0
    while (begin := _PEM_BEGIN.search(text, i)) is not None:
        out.append(text[i:begin.start()])
        out.append(REDACTED)
        end = _PEM_END.search(text, begin.end())
        if end is None:
            return "".join(out)
        i = end.end()
    out.append(text[i:])
    return "".join(out)


def redact(text: str) -> str:
    """Replace secrets. Every pattern matches the same text and overlapping
    matches merge, so one pattern's replacement can't break another's match
    (``Bearer x.sk-...`` is redacted whole, not just its ``sk-`` part)."""
    text = _redact_pem(text)
    spans = sorted(m.span() for pattern in _SECRET_PATTERNS for m in pattern.finditer(text))
    out, i = [], 0
    for a, b in spans:
        if b <= i:
            continue
        if a >= i:
            out.append(text[i:a])
            out.append(REDACTED)
        i = b
    out.append(text[i:])
    return "".join(out)


def clip(text: str, limit: int = CLIP_CHARS, hidden: int = 0) -> str:
    """Cut ``text`` to ``limit`` chars; ``hidden`` counts chars already cut before this call."""
    cut = max(0, len(text) - limit) + hidden
    return f"{text[:limit]} [...clipped {cut} chars]" if cut else text


def clean(text: str, tags: Iterable[str], dropped: int = 0) -> str:
    """Strip, redact and clip; ``dropped`` counts chars already cut upstream."""
    stripped = strip_injected(text, tags, truncated=dropped > 0)
    # Redact a window reaching past the clip point, then clip, so a clip can't
    # cut a secret in half and leave a fragment the patterns miss. Redaction can
    # shrink the window (a long key becomes "[REDACTED]"), which would pull a
    # secret cut at the window's edge back inside the clip; so the window
    # doubles until the redacted text still reaches the margin. Cost stays
    # linear in the text.
    need = CLIP_CHARS + _REDACT_MARGIN
    size = need
    while True:
        window = stripped[:size]
        red = redact(window)
        if len(red) >= need or size >= len(stripped):
            break
        size *= 2
    return clip(red, hidden=len(stripped) - len(window) + dropped)


# --- Claude Code transcripts ---------------------------------------------------


def _text_blocks(content) -> list[str]:
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        return [b["text"] for b in content
                if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str)]
    return []


def _attachment_markers(content) -> list[str]:
    """``[image]``-style placeholders for a message with attachments but no text.

    The attachment isn't stored, but the marker keeps it a turn of its own, so
    its reply isn't merged into the previous exchange.
    """
    if not isinstance(content, list):
        return []
    return [f"[{b['type']}]" for b in content
            if isinstance(b, dict) and b.get("type") in ("image", "document")]


def _str(value) -> str:
    return value if isinstance(value, str) and len(value) <= MAX_META_CHARS else ""


def _usable_id(value, limit: int = MAX_ID_CHARS) -> bool:
    """A non-empty, bounded string that encodes as UTF-8 (JSON can carry lone surrogates)."""
    if not isinstance(value, str) or not value or len(value) > limit:
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _new_turn(anchor: str, r: dict) -> dict:
    offset = r.get(OFFSET_KEY)
    return {"anchor": anchor, "timestamp": _str(r.get("timestamp")), "cwd": _str(r.get("cwd")),
            "offset": offset if isinstance(offset, int) else None,
            "user": [], "assistant": [], "size": {"user": 0, "assistant": 0},
            "dropped": {"user": 0, "assistant": 0}}


def _add(turn: dict, side: str, texts: list[str]) -> None:
    """Append text to one side of a turn, keeping at most ``MAX_SIDE_CHARS``.

    A long run of assistant records would otherwise pile up in memory only to
    be clipped to ``CLIP_CHARS``. The cut is counted in the clip marker. It
    falls on whitespace, dropping the partial word before it, so it can't
    leave a key fragment too short for the redaction patterns; redaction
    itself runs later on the joined text, so matches can span blocks.
    """
    for t in texts:
        room = MAX_SIDE_CHARS - turn["size"][side]
        if room <= 0:
            turn["dropped"][side] += len(t)
            continue
        keep = len(t)
        if keep > room:
            keep = room
            while keep and not t[keep].isspace():  # t[room] exists: len(t) > room
                keep -= 1
            turn["size"][side] = MAX_SIDE_CHARS  # full: later blocks are dropped
        else:
            turn["size"][side] += keep
        turn[side].append(t[:keep])
        turn["dropped"][side] += len(t) - keep


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
        user = clean("\n\n".join(t["user"]), tags, t["dropped"]["user"])
        assistant = clean("\n\n".join(t["assistant"]), tags, t["dropped"]["assistant"])
        if not user and not assistant:
            return None
        return Turn(
            id=str(uuid.uuid5(TURN_NAMESPACE, f"claude-code:{session_id}:{t['anchor']}")),
            anchor=t["anchor"], client="claude-code", session_id=session_id, timestamp=t["timestamp"],
            cwd=t["cwd"] or cwd, user=user, assistant=assistant, offset=t["offset"],
        )

    for r in records:
        kind = r.get("type")
        if kind not in ("user", "assistant") or any(r.get(f) for f in _SKIP_FLAGS):
            continue
        message = r.get("message")
        anchor = r.get("uuid")
        if not isinstance(message, dict) or not _usable_id(anchor):
            continue
        content = message.get("content")
        if kind == "user":
            # Only text blocks count, so a tool-result message opens no turn,
            # but prose sent alongside tool results still does.
            texts = [strip_injected(t, tags) for t in _text_blocks(content)]
            texts = [t for t in texts if t] or _attachment_markers(content)
            if not texts:
                continue
            done = finish(turn)
            if done:
                yield done
            turn = _new_turn(anchor, r)
            _add(turn, "user", texts)
        else:
            # Stripped per block, like the user side, so the side cap can't cut
            # an injected block's closing tag off and keep the block.
            texts = [strip_injected(t, tags) for t in _text_blocks(content)]
            texts = [t for t in texts if t and t != "(no content)"]
            if not texts:
                continue
            if turn is None:
                turn = _new_turn(anchor, r)
            _add(turn, "assistant", texts)
    done = finish(turn)
    if done:
        yield done


ADAPTERS: dict[str, Callable[..., Iterator[Turn]]] = {"claude-code": claude_code_turns}


# --- reading new transcript lines ---------------------------------------------


def open_transcript(path: Path):
    """Open a transcript for reading, refusing anything but a regular file.

    Opened non-blocking first, so a FIFO path can't hang the hook.
    """
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError(f"{path} is not a regular file")
        os.set_blocking(fd, True)
        return os.fdopen(fd, "rb")
    except BaseException:
        os.close(fd)
        raise


class TranscriptReader:
    """Iterate ``(start_offset, record)`` over the complete JSONL lines after ``offset``.

    Streams line by line, so memory is bounded by the longest line, and lines
    over ``MAX_LINE_BYTES`` are skipped without being held. A trailing line
    without a newline is still being written, so it's left for the next run.
    After iterating, ``end`` is the offset just past the last complete line.
    """

    def __init__(self, f, offset: int, name: str = "transcript", deadline: float | None = None):
        self._f, self.end, self._name, self._deadline = f, offset, name, deadline
        self.timed_out = False

    def _out_of_time(self) -> bool:
        if self._deadline is None or time.monotonic() < self._deadline:
            return False
        log.warning("capture time budget used up at byte %d in %s; resuming next run",
                    self.end, self._name)
        self.timed_out = True
        return True

    def __iter__(self) -> Iterator[tuple[int, dict]]:
        f = self._f
        f.seek(self.end)
        while not self._out_of_time() and (line := f.readline(MAX_LINE_BYTES + 1)):
            start = self.end
            if not line.endswith(b"\n"):
                if len(line) <= MAX_LINE_BYTES:
                    return  # partial last line
                length = self._skip_rest(len(line))
                if length is None:
                    return  # oversized and still being written
                self.end += length
                log.warning("skipping a %d-byte transcript line at byte %d in %s",
                            length, start, self._name)
                continue
            self.end += len(line)
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except (ValueError, RecursionError):  # bad JSON, bad UTF-8, absurd nesting
                log.warning("skipping malformed transcript line at byte %d in %s",
                            start, self._name)
                continue
            if isinstance(rec, dict):
                yield start, rec

    def _skip_rest(self, length: int) -> int | None:
        """Read past the rest of an oversized line; its total length, or None at
        EOF or when the time budget runs out (the line is retried next run)."""
        while not self._out_of_time() and (chunk := self._f.readline(1 << 20)):
            length += len(chunk)
            if chunk.endswith(b"\n"):
                return length
        return None


# --- state: cursors, spool, lock -----------------------------------------------


def state_dir() -> Path:
    xdg = os.environ.get("XDG_STATE_HOME")
    base = Path(xdg) if xdg and os.path.isabs(xdg) else Path.home() / ".local" / "state"
    return base / "kipp" / "capture"


def _owned_dir(path: Path, *, mine: bool) -> os.stat_result:
    """lstat ``path``, refusing a symlink, a non-directory, or another user's directory."""
    st = os.lstat(path)
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
        raise ValueError(f"{path} is a symlink or not a directory; refusing to write conversation text there")
    if st.st_uid != os.getuid() and (mine or st.st_uid != 0):
        raise ValueError(f"{path} is owned by uid {st.st_uid}; refusing to write conversation text there")
    return st


def ensure_state_dir() -> Path:
    """Create the capture directory, private to the user: it holds conversation text.

    No other user may be able to swap a directory on the path between these
    checks and the writes: ``capture`` and ``kipp`` must be ours and lose any
    group/other write bit, and the state base must be ours or root's and not
    world-writable, unless sticky (like /tmp, where others can't rename our
    entries). A group-writable base is allowed: with umask 002 that's the
    user's own private group.
    """
    d = state_dir()
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    base = _owned_dir(d.parent.parent, mine=False)
    if base.st_mode & stat.S_IWOTH and not base.st_mode & stat.S_ISVTX:
        raise ValueError(f"{d.parent.parent} is world-writable; refusing to write conversation text there")
    kipp = _owned_dir(d.parent, mine=True)
    if kipp.st_mode & 0o022:
        os.chmod(d.parent, stat.S_IMODE(kipp.st_mode) & ~0o022)
    _private_dir(d)
    return d


def _private_dir(path: Path) -> None:
    """Refuse a state directory others can or could get into.

    We always create these 0700, so a looser one was made or changed by
    someone else, who may still hold it open: refuse rather than tighten.
    """
    st = _owned_dir(path, mine=True)
    if stat.S_IMODE(st.st_mode) & 0o077:
        raise ValueError(f"{path} is accessible to other users; refusing to write conversation "
                         "text there (check it and remove it)")


_PRIVATE_FLAGS = os.O_CREAT | os.O_NOFOLLOW


def _open_private(path: Path, flags: int):
    """Open a state file 0600, refusing a symlink or hard link planted in its place.

    O_NOFOLLOW stops symlinks only, and the create mode applies only to new
    files, so the opened file is checked: ours, a regular file, one link.
    """
    truncate = flags & os.O_TRUNC  # only once the file is known to be ours alone
    fd = os.open(path, (flags & ~os.O_TRUNC) | _PRIVATE_FLAGS | os.O_NONBLOCK, 0o600)  # a FIFO can't hang it
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid() or st.st_nlink != 1:
            raise ValueError(f"{path} is not a private regular file; refusing to write conversation text to it")
        if stat.S_IMODE(st.st_mode) & 0o077:
            # Tightening wouldn't revoke a descriptor someone opened meanwhile.
            raise ValueError(f"{path} is readable or writable by others; refusing to write "
                             "conversation text to it (check it and remove it)")
        if truncate:
            os.ftruncate(fd, 0)
        os.set_blocking(fd, True)
        return fd
    except BaseException:
        os.close(fd)
        raise


def spool_path() -> Path:
    return state_dir() / "spool.jsonl"


def _cursor_path(client: str, session_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{client}-{session_id}")
    return state_dir() / "cursors" / f"{safe}.json"


_FINGERPRINT_BYTES = 256


def _fingerprint(f, offset: int) -> str:
    """Hash of the bytes just before ``offset``. Appends never change them; a rewrite does."""
    start = max(0, offset - _FINGERPRINT_BYTES)
    f.seek(start)
    return hashlib.sha256(f.read(offset - start)).hexdigest()


_CURSOR_MAX_BYTES = 64 << 10


def _read_private(path: Path, limit: int) -> bytes:
    """Read a state file we wrote: no symlink, FIFO, device or other user's file, and at most ``limit`` bytes."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid():
            raise ValueError(f"{path} is not a private regular file")
        with os.fdopen(fd, "rb") as f:
            fd = -1
            data = f.read(limit + 1)
    finally:
        if fd >= 0:
            os.close(fd)
    if len(data) > limit:
        raise ValueError(f"{path} is over {limit} bytes")
    return data


def _read_cursor(client: str, session_id: str, transcript: Path, f) -> int:
    """The saved offset, or 0 when there is none or the transcript was replaced or rewritten.

    Restarting is always safe: turn ids are stable, so the flush deduplicates.
    """
    p = _cursor_path(client, session_id)
    try:
        data = json.loads(_read_private(p, _CURSOR_MAX_BYTES))
    except FileNotFoundError:
        return 0
    except (OSError, ValueError) as exc:  # JSONDecodeError and bad UTF-8 are ValueErrors
        log.warning("unreadable cursor %s (%s); recapturing from the start", p, exc)
        return 0
    offset = data.get("offset") if isinstance(data, dict) else None
    if not isinstance(offset, int) or offset < 0 or data.get("path") != str(transcript):
        return 0
    st = os.fstat(f.fileno())
    if (data.get("inode") != st.st_ino or offset > st.st_size
            or data.get("fingerprint") != _fingerprint(f, offset)):
        log.info("transcript %s changed since the last capture; recapturing from the start",
                 transcript)
        return 0
    return offset


def _write_cursor(client: str, session_id: str, transcript: Path, f, offset: int) -> None:
    p = _cursor_path(client, session_id)
    p.parent.mkdir(exist_ok=True, mode=0o700)
    _private_dir(p.parent)
    tmp = p.with_suffix(".tmp")
    data = json.dumps({
        "path": str(transcript), "offset": offset, "inode": os.fstat(f.fileno()).st_ino,
        "fingerprint": _fingerprint(f, offset),
    })
    with os.fdopen(_open_private(tmp, os.O_WRONLY | os.O_TRUNC), "w") as out:
        out.write(data)
    tmp.replace(p)


class _Lock:
    """Exclusive lock over the whole read-cursor → append → write-cursor sequence.

    Stop and SessionEnd can fire close together, and several sessions share
    one spool, so captures are serialised.
    """

    def __init__(self, timeout: float | None = None):
        self._timeout = timeout

    def __enter__(self):
        self._f = os.fdopen(_open_private(ensure_state_dir() / "capture.lock", os.O_RDWR), "r+b")
        try:
            if self._timeout is None:
                fcntl.flock(self._f, fcntl.LOCK_EX)
            else:
                self._acquire_within(self._timeout)
        except BaseException:
            self._f.close()
            raise
        return self

    def _acquire_within(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(self._f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"capture lock busy for {timeout:g}s") from None
                time.sleep(0.05)

    def __exit__(self, *exc):
        fcntl.flock(self._f, fcntl.LOCK_UN)
        self._f.close()


def _append_to_spool(data: str) -> None:
    """Append records to the spool, first cutting any partial record a failed write left.

    Called under ``_Lock``. Without the repair, a record half-written when the
    disk filled would swallow the first record of the next append.
    """
    with os.fdopen(_open_private(spool_path(), os.O_RDWR), "r+b") as f:
        size = f.seek(0, os.SEEK_END)
        if size:
            f.seek(size - 1)
            if f.read(1) != b"\n":
                tail_start = max(0, size - (1 << 20))
                f.seek(tail_start)
                last_nl = f.read().rfind(b"\n")
                keep = tail_start + last_nl + 1 if last_nl >= 0 else 0
                log.warning("cutting a %d-byte partial record from the end of %s",
                            size - keep, spool_path())
                f.truncate(keep)
                f.seek(keep)
        # "replace": a lone surrogate from a JSON escape can't encode as UTF-8,
        # and one bad character mustn't block every later capture.
        f.write(data.encode("utf-8", "replace"))
        f.flush()
        os.fsync(f.fileno())


# --- domain routing ------------------------------------------------------------


def resolve_domain(cwd: str, routes: dict[str, str], default: str = "shared") -> str:
    """Pick the domain for a working directory: the longest matching route prefix wins.

    ``routes`` maps directory prefixes (``~`` expanded) to domains, e.g.
    ``{"~/dev": "dev"}``. A prefix matches whole path components only, so
    ``~/dev`` doesn't match ``~/devices``.
    """
    if not cwd or "\0" in cwd:
        return default
    try:
        path = os.path.normpath(os.path.expanduser(cwd))
    except (ValueError, UnicodeError):  # a malformed record's cwd
        return default
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


SPOOL_BATCH = 500
SPOOL_BATCH_CHARS = 4 << 20
HOOK_LOCK_TIMEOUT = 5.0  # seconds a hook waits for another capture before giving up
HOOK_TIME_BUDGET = 10.0  # seconds a hook spends reading before leaving the rest for next time


def capture_transcript(
    transcript: Path, *, session_id: str, cwd: str = "", client: str = "claude-code",
    config: CaptureConfig | None = None, lock_timeout: float | None = None,
    time_budget: float | None = None,
) -> int:
    """Append the turns added to ``transcript`` since the last capture to the spool.

    ``lock_timeout`` bounds the wait for another capture to finish (the hook
    passes one, so it never stalls the agent); on timeout nothing is spooled
    and the next capture picks the turns up. ``time_budget`` bounds the
    reading: when it runs out, what was read is spooled and the next capture
    resumes from the last turn, so a huge backlog is caught up over several
    runs instead of stalling one.

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
    with _Lock(lock_timeout), open_transcript(transcript) as f:
        offset = _read_cursor(client, session_id, transcript, f)
        deadline = None if time_budget is None else time.monotonic() + time_budget
        reader = TranscriptReader(f, offset, str(transcript), deadline)
        def records() -> Iterator[dict]:
            # Each record carries its start offset, which the adapter copies
            # onto the turn it opens, to place the cursor.
            for start, rec in reader:
                rec[OFFSET_KEY] = start
                yield rec

        # Spooled in batches, so a big backlog isn't held in memory. A failure
        # part-way leaves the cursor unmoved and the next run re-spools those
        # turns under the same ids, which the flush deduplicates.
        count, resume, lines, size = 0, None, [], 0
        for t in adapter(records(), session_id=session_id, cwd=cwd, tags=config.strip_tags):
            t.domain = resolve_domain(t.cwd, config.routes, config.default_domain)
            lines.append(json.dumps(t.spool_record(), ensure_ascii=False) + "\n")
            size += len(lines[-1])
            if t.offset is None:
                raise RuntimeError(f"the {client} adapter left a turn without its offset")
            count, resume = count + 1, t.offset
            if len(lines) >= SPOOL_BATCH or size >= SPOOL_BATCH_CHARS:
                _append_to_spool("".join(lines))
                lines, size = [], 0
        if lines:
            _append_to_spool("".join(lines))
        new_offset = resume if resume is not None else reader.end
        if reader.timed_out and new_offset == offset:
            # One turn outran the whole budget: resuming at its start would
            # never get further. Move on; the rest of it is captured next run
            # as a turn of its own.
            new_offset = reader.end
        if new_offset != offset:
            _write_cursor(client, session_id, transcript, f, new_offset)
    return count


# --- entry point -----------------------------------------------------------------


def _setup_logging(hook: bool) -> None:
    handler: logging.Handler = logging.StreamHandler(sys.stderr)
    if hook:
        # A hook's stderr isn't shown to the operator, so errors go to a file,
        # falling back to stderr if the state directory isn't writable.
        try:
            log_fd = _open_private(ensure_state_dir() / "capture.log", os.O_WRONLY | os.O_APPEND)
            handler = logging.StreamHandler(os.fdopen(log_fd, "a", encoding="utf-8"))
        except (OSError, ValueError) as exc:  # unwritable, or a refused symlink
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
    if not _usable_id(transcript, MAX_META_CHARS) or not _usable_id(session_id):
        log.error("hook payload missing a usable transcript_path or session_id: keys=%s",
                  sorted(payload))
        return 0
    n = capture_transcript(Path(transcript), session_id=session_id, cwd=payload.get("cwd", ""),
                           lock_timeout=HOOK_LOCK_TIMEOUT, time_budget=HOOK_TIME_BUDGET)
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
