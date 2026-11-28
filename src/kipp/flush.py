"""Flush the capture spool into the events collections (issue #71).

``kipp-capture`` appends one record per conversation turn to a local spool;
``kipp capture flush`` embeds them and upserts them as events
(``source="capture"``), each in its turn's domain. From there they wait for a
consolidation pass to turn them into entities, or expire.

The spool is moved aside (``spool.flushing.jsonl``) under the capture lock, so
hooks keep appending to a fresh spool while the slow part (the embedder) runs
unlocked. The moved file is deleted only once every record is written; a
flush that fails part-way leaves it, and the next flush finishes it first.

Each hook run re-spools the session's last turn, which may still be growing,
under the same turn id, so the last copy of a turn in the spool wins. Each
distinct text of a turn is its own event, ``uuid5(turn id, sha256(text))``:

- an unchanged turn maps to an existing event and is skipped, so a turn
  consolidation already handled isn't queued again;
- a changed turn is a new event, and older versions still unextracted are
  marked extracted with ``superseded_by``. Events are never rewritten in
  place, so resolving an event id acknowledges exactly the text that was
  read, even if a newer version landed meanwhile;
- a retired version that becomes current again (a rewound transcript) is
  requeued, unless a consolidator resolved it;
- every other unextracted version of a turn in the spool is retired, so a
  flush that failed part-way is finished by the next one.

Where flushes and consolidation race, the cost is a turn consolidated twice
(which merge deduplicates), never a turn left unconsolidated.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from kipp import capture
from kipp.client import (
    retired_events,
    mark_events_superseded,
    requeue_events,
    unextracted_event_ids_by,
    upsert_event_points,
)
from kipp.config import DOMAINS, EVENT_TTL_DAYS

log = logging.getLogger("kipp.flush")

SOURCE = "capture"
FLUSH_BATCH = 64
# Seconds to wait for a running hook (at most HOOK_TIME_BUDGET of reading,
# then its spool write) before moving the spool aside.
SPOOL_LOCK_TIMEOUT = 30.0

_FIELDS = ("id", "text", "domain", "session_id", "client", "cwd", "timestamp")


@dataclass
class FlushResult:
    written: int = 0  # event versions stored
    unchanged: int = 0  # turns whose current text was already stored
    superseded: int = 0  # older unextracted versions retired
    requeued: int = 0  # retired versions current again
    malformed: int = 0
    domains: Counter = field(default_factory=Counter)  # written per domain


def version_id(turn_id: str, text: str) -> str:
    """The event id for one text of a turn; a changed text is a new event."""
    digest = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()
    return str(uuid.uuid5(uuid.UUID(turn_id), digest))


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def naive_utc(timestamp: str) -> str | None:
    """A transcript timestamp as naive UTC ISO, the form every other event uses.

    Event queries compare ``timestamp`` against naive UTC datetimes, so an
    offset-aware value would raise there. ``None`` if it doesn't parse.
    """
    try:
        dt = datetime.fromisoformat(timestamp)
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt.isoformat()


def parse_record(line: bytes) -> dict | None:
    """One spool line as a record, or ``None`` if it's malformed or incomplete."""
    try:
        rec = json.loads(line)
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(rec, dict) or not all(isinstance(rec.get(k), str) for k in _FIELDS):
        return None
    if not rec["text"] or rec["domain"] not in DOMAINS:
        return None
    try:
        if str(uuid.UUID(rec["id"])) != rec["id"]:
            return None
    except ValueError:
        return None
    return rec


def _payload(rec: dict, now: datetime) -> dict:
    return {
        "text": rec["text"],
        "timestamp": naive_utc(rec["timestamp"]) or now.isoformat(),
        "source": SOURCE,
        "agent": rec["client"],
        "extracted": False,
        "expires": (now + timedelta(days=EVENT_TTL_DAYS)).isoformat(),
        "session_id": rec["session_id"],
        "client": rec["client"],
        "cwd": rec["cwd"],
        "turn_id": rec["id"],
    }


class _Writer:
    """Batches records per domain and writes the ones whose text changed."""

    def __init__(self, client: QdrantClient, get_embedder: Callable, result: FlushResult,
                 now: datetime):
        self._client = client
        self._get_embedder = get_embedder
        self._embedder = None
        self._result = result
        self._now = now
        self._pending: dict[str, list[dict]] = {d: [] for d in DOMAINS}

    def add(self, rec: dict) -> None:
        batch = self._pending[rec["domain"]]
        batch.append(rec)
        if len(batch) >= FLUSH_BATCH:
            self._write(rec["domain"])

    def finish(self) -> None:
        for domain in DOMAINS:
            if self._pending[domain]:
                self._write(domain)

    def _write(self, domain: str) -> None:
        batch, self._pending[domain] = self._pending[domain], []
        ids = {r["id"]: version_id(r["id"], r["text"]) for r in batch}
        stored = retired_events(self._client, list(ids.values()), domain=domain)
        todo = [r for r in batch if ids[r["id"]] not in stored]
        revived = [ids[r["id"]] for r in batch if stored.get(ids[r["id"]])]
        self._result.unchanged += len(batch) - len(todo) - len(revived)
        if todo:
            if self._embedder is None:  # loaded only when something needs embedding
                self._embedder = self._get_embedder()
            vectors = self._embedder.embed_batch([r["text"] for r in todo])
            points = [PointStruct(id=ids[r["id"]], vector=v, payload=_payload(r, self._now))
                      for r, v in zip(todo, vectors, strict=True)]
            upsert_event_points(self._client, points, domain=domain)
            self._result.written += len(todo)
            self._result.domains[domain] += len(todo)
        requeue_events(self._client, revived, domain=domain)
        self._result.requeued += len(revived)
        # The current versions are queued first, then the others retired: a
        # failure in between leaves both queued (a duplicate to consolidate),
        # never neither.
        current = set(ids.values())
        older = unextracted_event_ids_by(self._client, "turn_id", list(ids), domain=domain)
        superseded = {eid: ids[tid] for eid, tid in older if eid not in current and tid in ids}
        mark_events_superseded(self._client, superseded, domain=domain)
        self._result.superseded += len(superseded)


def _flush_file(path: Path, writer: _Writer, result: FlushResult) -> None:
    """Write the last copy of each id in ``path``. Two passes keep memory to the ids."""
    with os.fdopen(capture._open_private(path, os.O_RDONLY), "rb") as f:
        last: dict[str, int] = {}
        for n, line in enumerate(f):
            rec = parse_record(line)
            if rec is None:
                if line.strip():
                    result.malformed += 1
                    log.warning("skipping malformed record on line %d of %s", n + 1, path)
                continue
            last[rec["id"]] = n
        f.seek(0)
        for n, line in enumerate(f):
            rec = parse_record(line)
            if rec is not None and last.get(rec["id"]) == n:
                writer.add(rec)
    writer.finish()


def flush_spool(client: QdrantClient, get_embedder: Callable, *,
                now: datetime | None = None) -> FlushResult:
    """Embed spooled turns into their domains' events; return the counts.

    ``get_embedder`` is called only if a turn needs embedding, so an empty or
    unchanged spool never loads the model. The events collections must exist.
    Raises ``TimeoutError`` if another flush is running or a hook holds the
    spool too long.
    """
    result = FlushResult()
    writer = _Writer(client, get_embedder, result, now or _utcnow())
    pending, spool = capture.flushing_path(), capture.spool_path()
    with capture._Lock(0, name="flush.lock"):
        if os.path.lexists(pending):  # left by a flush that failed part-way
            _flush_file(pending, writer, result)
            os.unlink(pending)
        with capture._Lock(SPOOL_LOCK_TIMEOUT):
            if not os.path.lexists(spool):
                return result
            os.rename(spool, pending)
        _flush_file(pending, writer, result)
        os.unlink(pending)
    return result
