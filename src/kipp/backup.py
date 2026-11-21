"""Snapshot every Qdrant collection to local disk (#38).

``kipp export`` covers only the shared domain's entities, so it is not a
backup. This takes a native Qdrant snapshot of EVERY collection, downloads it,
verifies it against the checksum Qdrant reports, and deletes the server-side
copy (``/qdrant/snapshots`` is not on the data volume, so leftovers pile up in
the container's writable layer).

A run writes into ``<dest>/.partial-<stamp>/`` and is renamed to
``<dest>/<stamp>/`` only once every collection verified, so a directory named
like a stamp is always a complete backup. ``manifest.json`` records each
collection's exact point count and sha256 at backup time.

Same disk as the data: this covers a bad migration, a buggy repair or an
accidental delete — not drive failure.

Restore one collection (replaces it; Qdrant reads the file from the upload):
    curl -X POST "$QDRANT_URL/collections/<name>/snapshots/upload?priority=snapshot" \\
         -H "api-key: $QDRANT_API_KEY" -F snapshot=@<dest>/<stamp>/<name>.snapshot
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol

import httpx

STAMP_RE = re.compile(r"^\d{8}T\d{6}Z$")
PARTIAL_PREFIX = ".partial-"
DEFAULT_KEEP = 7  # ~150 MB each; a week of nightly rollback
CHUNK = 1 << 20
STALE_PARTIAL_SECONDS = 24 * 3600


class SnapshotClient(Protocol):
    """The slice of ``QdrantClient`` a backup needs."""

    def get_collections(self): ...
    def count(self, collection_name: str, exact: bool = True): ...
    def create_snapshot(self, collection_name: str, wait: bool = True): ...
    def delete_snapshot(self, collection_name: str, snapshot_name: str, wait: bool = True): ...


@dataclass
class BackupResult:
    path: Path
    collections: dict[str, dict] = field(default_factory=dict)  # name -> manifest entry
    pruned: list[Path] = field(default_factory=list)


def backup_all(
    client: SnapshotClient,
    http: httpx.Client,
    dest: Path,
    now: datetime,
    keep: int = DEFAULT_KEEP,
) -> BackupResult:
    """Snapshot every collection into ``dest/<stamp>/``, then prune to ``keep``.

    ``http`` must target the same Qdrant as ``client`` (base_url + api-key
    header); snapshots are downloaded through it because qdrant-client's
    generated download call decodes the binary body as text.
    """
    if keep < 1:
        raise ValueError(f"keep must be >= 1, got {keep}")
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    final = dest / stamp
    if final.exists():
        raise FileExistsError(f"backup already exists: {final}")
    partial = dest / f"{PARTIAL_PREFIX}{stamp}"
    dest.mkdir(parents=True, exist_ok=True)
    partial.mkdir(mode=0o700)

    result = BackupResult(path=final)
    try:
        names = sorted(c.name for c in client.get_collections().collections)
        for name in names:
            points = client.count(name, exact=True).count
            snap = client.create_snapshot(name, wait=True)
            if snap is None:
                raise RuntimeError(f"Qdrant returned no snapshot for {name}")
            try:
                digest, size = _download(http, name, snap.name, partial / f"{name}.snapshot")
            finally:
                client.delete_snapshot(name, snap.name, wait=True)
            if snap.checksum and digest != snap.checksum:
                raise RuntimeError(
                    f"{name}: downloaded sha256 {digest} != Qdrant's {snap.checksum}"
                )
            result.collections[name] = {"points": points, "bytes": size, "sha256": digest}
        _write_private(partial / "manifest.json", json.dumps(
            {"created": now.isoformat(), "collections": result.collections}, indent=2,
        ).encode())
        partial.rename(final)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    result.pruned = prune(dest, keep)
    return result


def _download(http: httpx.Client, collection: str, snapshot: str, path: Path) -> tuple[str, int]:
    sha, size = hashlib.sha256(), 0
    with http.stream("GET", f"/collections/{collection}/snapshots/{snapshot}") as resp:
        resp.raise_for_status()
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as fh:
            for chunk in resp.iter_bytes(CHUNK):
                fh.write(chunk)
                sha.update(chunk)
                size += len(chunk)
    if size == 0:
        raise RuntimeError(f"{collection}: empty snapshot download")
    return sha.hexdigest(), size


def _write_private(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)


def prune(dest: Path, keep: int) -> list[Path]:
    """Delete all but the newest ``keep`` complete backups, plus stale partials.

    Only directories this module names are touched: ``<stamp>`` and
    ``.partial-<stamp>``. Stamps sort chronologically as strings. A partial is
    left by a killed run (SIGKILL skips the cleanup); it is removed once it
    is a day old, so a concurrent run's in-flight directory is never touched.
    """
    removed: list[Path] = []
    complete = sorted(p for p in dest.iterdir() if p.is_dir() and STAMP_RE.match(p.name))
    for old in complete[:-keep]:
        shutil.rmtree(old)
        removed.append(old)
    cutoff = time.time() - STALE_PARTIAL_SECONDS
    for p in dest.iterdir():
        if (
            p.is_dir()
            and p.name.startswith(PARTIAL_PREFIX)
            and STAMP_RE.match(p.name[len(PARTIAL_PREFIX):])
            and p.stat().st_mtime < cutoff  # never a run still in flight
        ):
            shutil.rmtree(p)
            removed.append(p)
    return removed
