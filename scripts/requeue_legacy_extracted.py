"""One-off: hand legacy cosine-extracted facts back to the LLM extraction step.

Before #37, untagged events were cosine-matched to entities at 35% precision,
so most ``source="extract"`` facts sit on the wrong entity (#33). Nothing
records where they belong, so ``memory repair`` can't move them. This script
detaches them and re-logs each as an unextracted event in the same domain; the
nightly extractor's LLM step then places it (or dismisses it).

What counts as legacy: a current ``extract`` fact that is either
  - untagged AND was already untagged on the same entity in the PRE-REPAIR
    snapshot (facts ``memory repair`` moved also lose their tag; they are
    tag-verified, so they must stay), or
  - still tagged for a different entity (repair reported the target missing).
Run it once, after ``memory repair --apply`` and BEFORE tag-routed extraction
goes live (Endurance restart): later ``extract`` facts would not be in the
snapshot's legacy set anyway, but keep the ordering simple.

Requeued events:
  - get a deterministic id, so a re-run after a crash overwrites instead of
    duplicating; each event is written before its fact is detached, so a
    crash can duplicate a fact but never lose one;
  - are timestamped at the fact's ``added`` date, capped at 25h ago, so they
    queue in the extractor's backlog phase, not its last-24h phase;
  - live REQUEUE_TTL_DAYS (90), since the nightly step drains ~40 per domain.

Dry run by default. Usage:
  python scripts/requeue_legacy_extracted.py ~/backups/qdrant/<pre-repair-dir>
  python scripts/requeue_legacy_extracted.py ~/backups/qdrant/<dir> --apply
"""

from __future__ import annotations

import argparse
import sys
import uuid
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import warnings

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SnapshotPriority

from kipp.client import (
    collection_name,
    ensure_collections,
    get_client,
    point_to_entity,
    scroll_entities,
)
from kipp.embedder import Embedder
from kipp.extract import parse_lead_tag
from kipp.facts import write_entity
from kipp.models import Entity, Fact
from kipp.repair import apply_repair, plan_repair

DOMAINS = ("shared", "dev", "personal")
KINDS = ("entities", "decisions")
REQUEUE_TTL_DAYS = 90
NS = uuid.UUID("8a6f3c1e-2b1d-4c55-9a0e-6f1f7a0d4e21")
TMP_PREFIX = "tmp_requeue_"


def load_snapshot_entities(client, snap_dir: Path, domain: str) -> list[Entity]:
    """Restore the domain's entity snapshots into temp collections and read them.

    Uploads go through ``client`` so they hit the same server with the same
    credentials as every other read and write here.
    """
    entities: list[Entity] = []
    for kind in KINDS:
        coll = collection_name(domain, kind)
        snap = snap_dir / f"{coll}.snapshot"
        if not snap.exists():
            sys.exit(f"missing snapshot: {snap}")
        tmp = TMP_PREFIX + coll
        try:
            with snap.open("rb") as fh:
                client.http.snapshots_api.recover_from_uploaded_snapshot(
                    tmp, wait=True, priority=SnapshotPriority.SNAPSHOT, snapshot=fh,
                )
            offset = None
            while True:
                pts, offset = client.scroll(tmp, limit=200, offset=offset, with_payload=True)
                entities.extend(point_to_entity(p) for p in pts)
                if offset is None:
                    break
        finally:
            if client.collection_exists(tmp):
                client.delete_collection(tmp)
    return entities


def repair_absorbers(entities: list[Entity], embedder, now: datetime) -> set[tuple[str, str, str]]:
    """Keys of pre-repair facts that a repair move deduplicated into.

    Replays ``apply_repair`` on the snapshot entities against a throwaway
    in-memory store: a fact whose hit_count grew absorbed a tag-verified move,
    so it is verified too and must not be requeued.
    """
    before = {id(f): (e.id, f.text, f.added, f.hit_count) for e in entities for f in e.facts}
    scratch = QdrantClient(":memory:")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensure_collections(scratch)
    by_id = {e.id: e for e in entities}
    plan = plan_repair(entities)
    apply_repair(scratch, embedder, "shared", entities, plan, now)
    absorbed = set()
    for tid in {m.target_id for m in plan.moves}:
        for f in by_id[tid].facts:
            if id(f) in before and f.hit_count > before[id(f)][3]:
                absorbed.add(before[id(f)][:3])
    return absorbed


def legacy_keys(entities: list[Entity]) -> set[tuple[str, str, str]]:
    return {
        (e.id, f.text, f.added)
        for e in entities for f in e.facts
        if f.source == "extract" and f.is_current and parse_lead_tag(f.text)[0] is None
    }


def is_legacy(entity: Entity, fact: Fact, pre: set[tuple[str, str, str]]) -> bool:
    if fact.source != "extract" or not fact.is_current:
        return False
    tag = parse_lead_tag(fact.text)[0]
    if tag is None:
        return (entity.id, fact.text, fact.added) in pre
    return tag != entity.id


def has_body(fact: Fact) -> bool:
    """A bare tag ("[tool:gh]") carries no content: drop it, don't requeue."""
    return bool(parse_lead_tag(fact.text)[1].strip())


def event_point(
    domain: str, entity: Entity, fact: Fact, vector: list[float], now: datetime,
) -> PointStruct:
    latest = now - timedelta(hours=25)
    ts = min(datetime.fromisoformat(fact.added), latest)
    eid = str(uuid.uuid5(NS, f"{domain}|{entity.id}|{fact.added}|{fact.text}"))
    return PointStruct(id=eid, vector=vector, payload={
        "text": fact.text,
        "timestamp": ts.isoformat(),
        "source": "requeue",
        "agent": "requeue_legacy_extracted",
        "extracted": False,
        "expires": (now + timedelta(days=REQUEUE_TTL_DAYS)).isoformat(),
        "requeued_from": entity.id,
        "original_added": fact.added,
    })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("snapshot_dir", type=Path, help="PRE-repair snapshot directory")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    client = get_client()
    embedder = Embedder()  # repair replay must dedup exactly as the real run did
    now = datetime.utcnow()

    for domain in DOMAINS:
        snapshot = load_snapshot_entities(client, args.snapshot_dir, domain)
        pre = legacy_keys(snapshot)  # before the replay mutates the snapshot
        absorbed = repair_absorbers(snapshot, embedder, now)
        pre -= absorbed
        live = scroll_entities(client, domain=domain)
        plan = [(e, [f for f in e.facts if is_legacy(e, f, pre)]) for e in live]
        plan = [(e, out) for e, out in plan if out]
        total = sum(len(out) for _, out in plan)
        bare = sum(1 for _, out in plan for f in out if not has_body(f))
        emptied = [e.id for e, out in plan if len(out) == len(e.facts)]
        print(f"[{domain}] {total - bare} facts to requeue, {bare} bare tags to drop, "
              f"from {len(plan)} entities ({len(emptied)} left with no facts); "
              f"legacy set {len(pre)}, {len(absorbed)} repair-verified.")
        for e, out in sorted(plan, key=lambda p: -len(p[1]))[:8]:
            print(f"  {len(out):5d}  {e.id}  ({len(e.facts) - len(out)} kept)")
        for eid in emptied:
            print(f"  empty  {eid}")
        if not args.apply:
            continue

        events = collection_name(domain, "events")
        for entity, out in plan:
            keep = [f for f in out if has_body(f)]
            vectors = embedder.embed_batch([f.text for f in keep]) if keep else []
            points = [event_point(domain, entity, f, v, now) for f, v in zip(keep, vectors)]
            for i in range(0, len(points), 100):
                client.upsert(events, points=points[i:i + 100])
            leaving = {id(f) for f in out}
            entity.facts = [f for f in entity.facts if id(f) not in leaving]
            entity.last_updated = now.isoformat()
            write_entity(client, entity, embedder, domain=domain, now=now)
        print(f"  applied: {total - bare} events written, {bare} bare tags dropped, "
              f"{len(plan)} entities rewritten")

    if not args.apply:
        print("DRY-RUN: nothing written. Pass --apply to requeue.")


if __name__ == "__main__":
    main()
