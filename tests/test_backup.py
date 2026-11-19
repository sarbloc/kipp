"""Unit tests for backup.py (#38). No Qdrant needed: a fake snapshot client
plus an httpx.MockTransport serving the snapshot bytes."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from kipp.backup import PARTIAL_PREFIX, STALE_PARTIAL_SECONDS, backup_all, prune

NOW = datetime(2026, 11, 19, 16, 0, 0)
STAMP = "20261119T160000Z"


class FakeClient:
    def __init__(self, blobs: dict[str, bytes], checksums: dict[str, str] | None = None):
        self.blobs = blobs
        self.checksums = checksums or {n: hashlib.sha256(b).hexdigest() for n, b in blobs.items()}
        self.live: set[tuple[str, str]] = set()
        self.deleted: list[tuple[str, str]] = []

    def get_collections(self):
        return SimpleNamespace(collections=[SimpleNamespace(name=n) for n in self.blobs])

    def count(self, collection_name, exact=True):
        return SimpleNamespace(count=len(collection_name))

    def create_snapshot(self, collection_name, wait=True):
        name = f"{collection_name}-snap.snapshot"
        self.live.add((collection_name, name))
        return SimpleNamespace(name=name, checksum=self.checksums[collection_name])

    def delete_snapshot(self, collection_name, snapshot_name, wait=True):
        self.live.discard((collection_name, snapshot_name))
        self.deleted.append((collection_name, snapshot_name))


def http_for(client: FakeClient, status: int = 200) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        _, _, collection, _, _ = request.url.path.split("/")
        return httpx.Response(status, content=client.blobs[collection])

    return httpx.Client(base_url="http://qdrant", transport=httpx.MockTransport(handler))


def test_backup_writes_verified_snapshots_and_manifest(tmp_path):
    client = FakeClient({"entities": b"E" * 10, "dev_events": b"D" * 3})

    result = backup_all(client, http_for(client), tmp_path, NOW)

    final = tmp_path / STAMP
    assert result.path == final
    assert (final / "entities.snapshot").read_bytes() == b"E" * 10
    assert (final / "dev_events.snapshot").read_bytes() == b"D" * 3
    manifest = json.loads((final / "manifest.json").read_text())
    assert manifest["collections"]["entities"] == {
        "points": len("entities"), "bytes": 10,
        "sha256": hashlib.sha256(b"E" * 10).hexdigest(),
    }
    assert set(manifest["collections"]) == {"entities", "dev_events"}
    assert os.stat(final / "entities.snapshot").st_mode & 0o777 == 0o600
    assert os.stat(final).st_mode & 0o777 == 0o700
    assert not client.live  # every server-side snapshot cleaned up
    assert not list(tmp_path.glob(f"{PARTIAL_PREFIX}*"))


def test_checksum_mismatch_leaves_no_backup(tmp_path):
    client = FakeClient({"entities": b"E", "events": b"V"}, checksums={"entities": "0" * 64, "events": "0" * 64})

    with pytest.raises(RuntimeError, match="sha256"):
        backup_all(client, http_for(client), tmp_path, NOW)

    assert list(tmp_path.iterdir()) == []
    assert not client.live


def test_failed_download_still_deletes_server_snapshot(tmp_path):
    client = FakeClient({"entities": b"E"})

    with pytest.raises(httpx.HTTPStatusError):
        backup_all(client, http_for(client, status=500), tmp_path, NOW)

    assert client.deleted == [("entities", "entities-snap.snapshot")]
    assert not client.live
    assert list(tmp_path.iterdir()) == []


def test_empty_download_is_an_error(tmp_path):
    client = FakeClient({"entities": b""}, checksums={"entities": ""})

    with pytest.raises(RuntimeError, match="empty"):
        backup_all(client, http_for(client), tmp_path, NOW)

    assert list(tmp_path.iterdir()) == []


def test_existing_stamp_is_not_overwritten(tmp_path):
    (tmp_path / STAMP).mkdir()
    client = FakeClient({"entities": b"E"})

    with pytest.raises(FileExistsError):
        backup_all(client, http_for(client), tmp_path, NOW)

    assert not client.deleted  # never reached Qdrant


def test_keep_must_be_positive(tmp_path):
    client = FakeClient({"entities": b"E"})
    with pytest.raises(ValueError):
        backup_all(client, http_for(client), tmp_path, NOW, keep=0)


def test_backup_prunes_to_keep(tmp_path):
    for stamp in ("20261101T000000Z", "20261102T000000Z", "20261103T000000Z"):
        (tmp_path / stamp).mkdir()
    client = FakeClient({"entities": b"E"})

    result = backup_all(client, http_for(client), tmp_path, NOW, keep=2)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["20261103T000000Z", STAMP]
    assert sorted(p.name for p in result.pruned) == ["20261101T000000Z", "20261102T000000Z"]


def test_prune_only_touches_its_own_names(tmp_path):
    for name in ("20261101T000000Z", "20261102T000000Z", "notes", "2026-11-01"):
        (tmp_path / name).mkdir()
    (tmp_path / "20261103T000000Z.tar").write_text("x")  # a file, not a backup dir

    removed = prune(tmp_path, keep=1)

    assert [p.name for p in removed] == ["20261101T000000Z"]
    assert {p.name for p in tmp_path.iterdir()} == {
        "20261102T000000Z", "notes", "2026-11-01", "20261103T000000Z.tar",
    }


def test_prune_removes_only_stale_partials(tmp_path):
    stale = tmp_path / f"{PARTIAL_PREFIX}20261101T000000Z"
    fresh = tmp_path / f"{PARTIAL_PREFIX}20261119T155900Z"
    stale.mkdir()
    fresh.mkdir()
    old = time.time() - STALE_PARTIAL_SECONDS - 60
    os.utime(stale, (old, old))

    removed = prune(tmp_path, keep=14)

    assert removed == [stale]
    assert fresh.exists()  # a concurrent run's in-flight directory


def test_prune_keeps_everything_when_under_limit(tmp_path):
    for stamp in ("20261101T000000Z", "20261102T000000Z"):
        (tmp_path / stamp).mkdir()
    assert prune(tmp_path, keep=14) == []
    assert len(list(Path(tmp_path).iterdir())) == 2
