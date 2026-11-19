"""CLI entry point for the memory command."""

import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import click
import httpx
from qdrant_client.models import FieldCondition, Filter, MatchValue

from kipp.backup import DEFAULT_KEEP, backup_all
from kipp.client import (
    DOMAINS,
    collection_name,
    collection_stats,
    delete_entity,
    ensure_collections,
    get_client,
    get_entity,
    qdrant_connection,
    scroll_entities,
    store_event,
    upsert_entity,
)
from kipp.extract import MATCH_THRESHOLD
from kipp.maintenance import compact_domain, expire_events, expire_facts
from kipp.merge import build_search_text, mark_superseded, merge
from kipp.models import Entity, Fact
from kipp.pipeline import DEFAULT_BATCH_SIZE, run_extraction


def _get_embedder():
    """Lazy-load the embedder (avoids slow import on every CLI call)."""
    from kipp.embedder import Embedder

    return Embedder()


@click.group()
def main():
    """KIPP: entity memory for AI agents."""
    pass


@main.command()
def init():
    """Create collections and indexes if they don't exist."""
    client = get_client()
    created = ensure_collections(client)
    if created:
        click.echo(f"Created collections: {', '.join(created)}")
    else:
        click.echo("All collections already exist.")


@main.command()
def stats():
    """Show collection sizes and health."""
    client = get_client()
    st = collection_stats(client)
    for name, info in st.items():
        if info.get("exists") is False:
            click.echo(f"{name}: not created")
        else:
            click.echo(f"{name}: {info['points']} points")


@main.command()
@click.option("--type", "entity_type", required=True, help="Entity type: person|project|tool|preference|decision")
@click.option("--id", "entity_id", required=True, help="Entity key (e.g. alice, dashboard)")
@click.option("--content", required=True, help="Fact text to store")
@click.option(
    "--supersedes", default=None,
    help="Text of an existing current fact this replaces. "
         "Marks it superseded (kept for --as-of queries, hidden from search).",
)
@click.option(
    "--valid-from", "valid_from", default=None,
    help="ISO date the new fact became true in the world, if not today.",
)
def store(entity_type: str, entity_id: str, content: str, supersedes: str | None, valid_from: str | None):
    """Directly upsert an entity with a fact."""
    client = get_client()
    ensure_collections(client)
    embedder = _get_embedder()

    full_id = f"{entity_type}:{entity_id}" if ":" not in entity_id else entity_id
    now = datetime.utcnow()
    today = now.date().isoformat()

    # Future-effective dating isn't supported yet (issue #24) — see memory_store.
    if valid_from is not None and valid_from[:10] > today:
        raise click.ClickException(
            f"future valid_from {valid_from!r} is not supported yet (today is "
            f"{today}); future-effective dating is tracked in issue #24"
        )

    existing = get_entity(client, full_id)
    if existing is None:
        existing = Entity(id=full_id, type=entity_type)

    superseded = None
    if supersedes:
        # Close the old fact exactly when the replacement starts (its valid_from,
        # falling back to today): a backdated replacement must end the old fact at
        # the same date, or an as-of query in the gap sees both as valid (#21).
        marked = mark_superseded(
            existing, supersedes, by=content, on_date=valid_from or today,
        )
        superseded = marked.text if marked is not None else None

    new_fact = Fact(text=content, added=today, source="cli:store", valid_from=valid_from)
    merged = merge(existing, [new_fact], embedder, now=now)

    search_text = build_search_text(merged)
    vector = embedder.embed(search_text)
    upsert_entity(client, merged, vector)

    current_facts = sum(1 for f in merged.facts if f.is_current)
    click.echo(f"Stored: {full_id} ({current_facts} current facts)")
    if supersedes:
        if superseded is not None:
            click.echo(f"  Superseded: {superseded!r}")
        else:
            click.echo(f"  No current fact matched --supersedes {supersedes!r}")


@main.command()
@click.argument("entity_id")
def get(entity_id: str):
    """Retrieve a single entity's full detail."""
    client = get_client()
    entity = get_entity(client, entity_id)
    if entity is None:
        click.echo(f"Not found: {entity_id}")
        raise SystemExit(1)

    click.echo(f"{entity.type}:{entity.id}")
    click.echo(f"Updated: {entity.last_updated}")
    click.echo(f"Facts ({len(entity.facts)}):")
    for f in entity.facts:
        expires = f" [expires {f.expires}]" if f.expires else ""
        superseded = f" [superseded {f.superseded_at}]" if f.superseded_at else ""
        click.echo(f"  [{f.hit_count}x] {f.text} (since {f.added}, last seen {f.last_seen}){expires}{superseded}")


@main.command()
@click.argument("entity_id")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
def delete(entity_id: str, yes: bool):
    """Delete an entity permanently."""
    if not yes:
        click.confirm(f"Delete {entity_id}?", abort=True)

    client = get_client()
    if delete_entity(client, entity_id):
        click.echo(f"Deleted: {entity_id}")
    else:
        click.echo(f"Not found: {entity_id}")
        raise SystemExit(1)


@main.command(name="list")
@click.option("--type", "entity_type", default=None, help="Filter by type")
def list_entities(entity_type: str | None):
    """List all known entities."""
    client = get_client()
    entities = scroll_entities(client, entity_type)
    if not entities:
        click.echo("No entities found.")
        return
    for e in entities:
        updated = e.last_updated[:10] if e.last_updated else "unknown"
        click.echo(f"{e.id} ({len(e.facts)} facts, updated {updated})")


@main.command()
@click.argument("query")
@click.option("--type", "entity_type", default=None, help="Filter by entity type")
@click.option("--limit", default=5, help="Max results to return")
@click.option(
    "--as-of", "as_of", default=None,
    help="ISO date: show each entity as it was then (superseded facts in effect "
         "at that date are shown). Default hides superseded facts.",
)
def search(query: str, entity_type: str | None, limit: int, as_of: str | None):
    """Semantic search across all entities."""
    from kipp.search import search_entities, format_results

    client = get_client()
    embedder = _get_embedder()
    results = search_entities(
        client, query, embedder, entity_type=entity_type, limit=limit, as_of=as_of,
    )
    click.echo(format_results(results))


EVENT_TTL_DAYS = 30


@main.command()
@click.argument("text")
@click.option("--source", default="conversation", help="Event source: conversation|task|cron")
@click.option("--agent", default="main", help="Agent name")
def event(text: str, source: str, agent: str):
    """Log a raw observation. No entity extraction happens immediately."""
    client = get_client()
    ensure_collections(client)
    embedder = _get_embedder()

    now = datetime.utcnow()
    event_id = str(uuid.uuid4())
    vector = embedder.embed(text)
    expires = (now + timedelta(days=EVENT_TTL_DAYS)).isoformat()

    store_event(
        client, event_id=event_id, text=text, vector=vector,
        timestamp=now.isoformat(), source=source, agent=agent, expires=expires,
    )
    click.echo(f"Logged event: {event_id[:8]}...")


@main.command()
@click.option("--since", default=None, help="Process events from last N minutes (e.g. 55m)")
@click.option("--all", "process_all", is_flag=True, help="Process all unextracted events")
@click.option(
    "--domain", type=click.Choice(DOMAINS), default="shared",
    help="Domain to extract within: shared|dev|personal",
)
@click.option(
    "--batch", "batch_size", default=DEFAULT_BATCH_SIZE, show_default=True,
    help="Events per batch; each batch commits and reports progress before the next.",
)
@click.option(
    "--threshold", default=MATCH_THRESHOLD, show_default=True, type=float,
    help="Cosine gate for a sentence→fact match (with --match-untagged only).",
)
@click.option(
    "--match-untagged", is_flag=True,
    help="Also cosine-match events with no leading [type:id] tag. Off by "
         "default: it misattributes most facts (issue #33).",
)
def extract(
    since: str | None, process_all: bool, domain: str, batch_size: int,
    threshold: float, match_untagged: bool,
):
    """Process unextracted events into entity upserts.

    Events opening with a [type:id] tag attach to that entity when it exists.
    Events are processed in batches so a large backlog makes visible,
    kill-safe progress. Only events that enriched an existing entity are
    marked extracted; fully-unmatched events stay as new-entity candidates.
    """
    client = get_client()
    embedder = _get_embedder()

    since_dt = None
    if since and not process_all:
        minutes = int(since.rstrip("m"))
        since_dt = datetime.utcnow() - timedelta(minutes=minutes)

    def _progress(info: dict) -> None:
        click.echo(
            f"  batch {info['batch']}/{info['total_batches']}: "
            f"{info['events_in_batch']} events, "
            f"{info['matched_in_batch']} enriched "
            f"({info['cumulative_events']}/{info['total_events']} done)"
        )

    summary = run_extraction(
        client, embedder, domain=domain, since=since_dt,
        batch_size=batch_size, threshold=threshold, progress=_progress,
        match_untagged=match_untagged,
    )

    if summary["events_processed"] == 0:
        click.echo("No unextracted events found.")
        return

    click.echo(f"Processed {summary['events_processed']} events:")
    click.echo(f"  {summary['matched']} sentences matched to entities")
    click.echo(f"  {summary['unmatched']} unmatched sentences")
    for s in summary["unmatched_samples"]:
        click.echo(f"  UNMATCHED: \"{s}\"")


@main.command(name="dismiss")
@click.option(
    "--source", "src", required=True,
    help="Only dismiss unextracted events whose 'source' field equals this "
         "(e.g. claude_ai_import_conversation).",
)
@click.option("--domain", default="shared", help="Domain: shared|dev|personal")
@click.option(
    "--keep", "keep", multiple=True,
    help="Event ID to exclude from dismissal. Repeatable.",
)
@click.option(
    "--apply", "apply_changes", is_flag=True,
    help="Actually mark matched events extracted. Without this flag the command "
         "is a dry-run (prints count + sample, writes nothing).",
)
def dismiss(src: str, domain: str, keep: tuple[str, ...], apply_changes: bool):
    """Bulk-dismiss unextracted events by source, marking them extracted.

    Built for one-time cleanup of bulk imports (e.g. the Claude.ai conversation
    import) whose events are not memory-worthy: it marks them extracted=True
    with no entity created, so the extractor stops reconsidering them. Events
    are NOT deleted — their existing 30-day TTL still applies. Dry-run by
    default; pass --apply to commit.
    """
    client = get_client()
    coll = collection_name(domain, "events")
    if not client.collection_exists(coll):
        click.echo(f"No events collection for domain {domain!r}.")
        raise SystemExit(1)

    keep_set = set(keep)
    scroll_filter = Filter(
        must=[
            FieldCondition(key="extracted", match=MatchValue(value=False)),
            FieldCondition(key="source", match=MatchValue(value=src)),
        ]
    )

    matched_ids: list = []
    sample: list[tuple[str, str]] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=coll, scroll_filter=scroll_filter,
            limit=256, offset=offset, with_payload=True,
        )
        for p in points:
            if str(p.id) in keep_set:
                continue
            matched_ids.append(p.id)
            if len(sample) < 5:
                text = (p.payload or {}).get("text", "")
                sample.append((str(p.id), text.replace("\n", " ")[:100]))
        if offset is None:
            break

    click.echo(
        f"Matched {len(matched_ids)} unextracted events "
        f"(source={src!r}, domain={domain!r}, {len(keep_set)} excluded)."
    )
    for eid, snippet in sample:
        click.echo(f"  {eid}  {snippet}")
    if len(matched_ids) > len(sample):
        click.echo(f"  ... and {len(matched_ids) - len(sample)} more")

    if not apply_changes:
        click.echo("DRY-RUN: nothing written. Pass --apply to mark these extracted.")
        return

    if not matched_ids:
        click.echo("Nothing to dismiss.")
        return

    batch = 500
    for i in range(0, len(matched_ids), batch):
        chunk = matched_ids[i : i + batch]
        client.set_payload(
            collection_name=coll, payload={"extracted": True}, points=chunk,
            wait=True,
        )
    click.echo(f"Dismissed {len(matched_ids)} events (extracted=True).")


def _resolve_domains(domain: str) -> list[str]:
    return list(DOMAINS) if domain == "all" else [domain]


# Maintenance commands default to `shared` (the pre-#32 behaviour) so existing
# callers keep their blast radius; `--domain all` opts in to every domain.
_maintenance_domain = click.option(
    "--domain", type=click.Choice([*DOMAINS, "all"]), default="shared",
    show_default=True, help="Domain to operate on, or 'all'.",
)


@main.command(name="compact")
@click.option("--max-facts", default=20, help="Max facts per entity")
@_maintenance_domain
def compact_cmd(max_facts: int, domain: str):
    """Run compaction on all entities exceeding the facts limit."""
    client = get_client()
    embedder = _get_embedder()
    now = datetime.utcnow()

    for d in _resolve_domains(domain):
        compacted = compact_domain(client, embedder, d, now, max_facts=max_facts)
        click.echo(f"[{d}] Compacted {compacted} entities (max_facts={max_facts}).")


@main.command()
@_maintenance_domain
def expire(domain: str):
    """Garbage-collect expired events and entity facts."""
    client = get_client()
    embedder = _get_embedder()
    now = datetime.utcnow()

    for d in _resolve_domains(domain):
        expired_events = expire_events(client, d, now)
        updated_entities = expire_facts(client, embedder, d, now)
        click.echo(
            f"[{d}] Expired {expired_events} events, updated {updated_entities} entities."
        )


@main.command(name="export")
@click.option("--format", "fmt", type=click.Choice(["json", "md"]), default="json", help="Output format")
def export_cmd(fmt: str):
    """Export all entities for backup or inspection."""
    import sys
    from kipp.export import export_json, export_markdown

    client = get_client()
    entities = scroll_entities(client)

    if fmt == "json":
        export_json(entities, sys.stdout)
    else:
        export_markdown(entities, sys.stdout)


@main.command(name="import")
@click.argument("file", type=click.Path(exists=True))
def import_cmd(file: str):
    """Import entities from a JSON export. Runs merge logic (won't duplicate)."""
    import json as json_mod
    from kipp.export import import_json, reject_future_dated_facts

    client = get_client()
    ensure_collections(client)
    embedder = _get_embedder()
    now = datetime.utcnow()

    with open(file) as f:
        data = json_mod.load(f)

    imported = import_json(data)
    # Validate the whole backup before writing anything, so a bad fact aborts the
    # import atomically rather than half-applying it (issue #24).
    try:
        reject_future_dated_facts(imported, now.date().isoformat())
    except ValueError as exc:
        raise click.ClickException(str(exc))

    for entity in imported:
        existing = get_entity(client, entity.id)
        if existing is None:
            existing = Entity(id=entity.id, type=entity.type)

        merged = merge(existing, entity.facts, embedder, now=now)
        vector = embedder.embed(build_search_text(merged))
        upsert_entity(client, merged, vector)

    click.echo(f"Imported {len(imported)} entities.")


@main.command()
@click.option(
    "--domain", type=click.Choice([*DOMAINS, "all"]), default="all",
    show_default=True, help="Domain to repair, or 'all'.",
)
@click.option("--apply", "apply_changes", is_flag=True, help="Write the moves. Default is a dry run.")
def repair(domain: str, apply_changes: bool):
    """Move extracted facts whose [type:id] tag names a different entity.

    Dry run by default: prints the planned moves, tags whose entity doesn't
    exist (left in place), and per-entity counts of untagged extracted facts
    (no ground truth, so not moved). Take a backup before --apply.
    """
    from kipp.repair import apply_repair, plan_repair

    client = get_client()
    embedder = _get_embedder() if apply_changes else None
    now = datetime.utcnow()

    for d in DOMAINS if domain == "all" else [domain]:
        entities = scroll_entities(client, domain=d)
        plan = plan_repair(entities)
        moves_by_pair = Counter((m.source_id, m.target_id) for m in plan.moves)
        click.echo(
            f"[{d}] {len(plan.moves)} facts to move, "
            f"{sum(plan.missing_targets.values())} with a missing target, "
            f"{sum(plan.untagged_extracted.values())} untagged extracted."
        )
        for (src, dst), n in moves_by_pair.most_common():
            click.echo(f"  move {n:>4}  {src} -> {dst}")
        for tag, n in plan.missing_targets.most_common():
            click.echo(f"  missing {n:>4}  {tag}")
        for eid, n in plan.untagged_extracted.most_common(10):
            click.echo(f"  untagged {n:>4}  {eid}")
        if apply_changes and plan.moves:
            rewritten = apply_repair(client, embedder, d, entities, plan, now)
            click.echo(f"  applied: {rewritten} entities rewritten")

    if not apply_changes:
        click.echo("DRY-RUN: nothing written. Pass --apply to move facts.")


@main.command()
def mcp():
    """Run the MCP stdio server (used by Endurance and other MCP clients)."""
    from kipp.mcp_server import run

    run()


BACKUP_TIMEOUT = 600  # seconds, for snapshot create + download


@main.command(name="backup")
@click.option(
    "--dest", type=click.Path(file_okay=False, path_type=Path),
    default=Path.home() / "backups" / "qdrant", show_default=True,
    help="Directory holding one subdirectory per backup",
)
@click.option("--keep", default=DEFAULT_KEEP, show_default=True, help="Complete backups to retain")
def backup_cmd(dest: Path, keep: int):
    """Snapshot every Qdrant collection to DEST/<stamp>/ (verified, pruned)."""
    # Snapshot creation of a large collection can outlast the 5s default.
    client = get_client(timeout=BACKUP_TIMEOUT)
    _, api_key = qdrant_connection()
    headers = {"api-key": api_key} if api_key else {}
    # client.http.client.host is the endpoint qdrant-client resolved (default
    # port filled in), so downloads hit the same server snapshots were made on.
    with httpx.Client(base_url=client.http.client.host, headers=headers, timeout=BACKUP_TIMEOUT) as http:
        result = backup_all(client, http, dest, datetime.now(timezone.utc), keep=keep)
    for name, entry in result.collections.items():
        click.echo(f"  {name:20s} {entry['points']:6d} points  {entry['bytes'] / 1e6:7.1f} MB")
    click.echo(f"Backed up {len(result.collections)} collections to {result.path}"
               f" ({len(result.pruned)} old backups pruned).")


if __name__ == "__main__":
    main()
