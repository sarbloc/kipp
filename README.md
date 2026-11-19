<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/brand/kipp-logo-dark.svg">
    <img src="docs/brand/kipp-logo.svg" alt="KIPP" width="360">
  </picture>
</h1>

Memory for AI agents that keeps one current record per thing, not an
ever-growing log. Local, deterministic, no LLM in the loop.

Knowledge is stored as discrete entities upserted in place, so storage grows
with the number of distinct things, not with time.

> Renamed from `entity-memory`. The `memory` command still works as an alias,
> and the old config paths and `ENTITY_MEMORY_CONFIG` are still read.

## Quick start

```bash
# Start Qdrant
docker compose up -d

# Install
pip install -e .

# Initialize collections
kipp init

# Store a fact
kipp store --type person --id alice --content "Manages the auth team"

# Search
kipp search "who handles authentication"

# Log an observation
kipp event "Task completed: built auth API. Alice reviewed."

# Process events into entities
kipp extract --all
```

## Upgrading from entity-memory

The pip distribution changed name, so pipx sees a new app. Swap them, or
`memory` keeps pointing at the old install and fails to import:

```bash
pipx uninstall entity-memory
pipx install -e .            # provides `kipp` and the `memory` alias
```

`scripts/install.sh` does this for you. Existing `entity-memory-*` systemd
units keep working, because they call `memory`, which is now the alias. To
move to the new unit names, disable the old timers, remove their files from
`~/.config/systemd/user/`, then install the `kipp-*` units as described under
Maintenance.

Qdrant data and collection names are unchanged.

## Architecture

- **Qdrant** (Docker) — vector database for semantic search
- **all-MiniLM-L6-v2** — local embeddings, 384 dims, no API calls
- **Three collections**: `entities` (long-term knowledge), `events` (30-day rolling window), `decisions` (permanent)

All operations are deterministic (no LLM calls). Deduplication, merging, and compaction use cosine similarity and scoring algorithms.

## Configuration

By default the CLI talks to a local Qdrant at `http://127.0.0.1:6333`. Override
without any file via environment variables:

- `QDRANT_URL` — Qdrant endpoint (highest precedence; no config file needed)
- `QDRANT_API_KEY` — API key, if your Qdrant requires one
- `KIPP_CONFIG` — path to a config JSON (overrides the default location)

Or drop a config file at `~/.config/kipp/config.json` (honors
`$XDG_CONFIG_HOME` when set to an absolute path):

```json
{
  "qdrant": {
    "url": "http://127.0.0.1:6333",
    "api_key_env": "QDRANT_API_KEY"
  }
}
```

The pre-rename `~/.config/entity-memory/config.json` and the legacy
`~/.openclaw/memory.json` are still read, in that order, after it.

## Maintenance

`kipp expire` deletes events past their 30-day TTL and facts past their
`expires` date. `kipp compact` trims entities over the 20-fact cap. Both
default to the `shared` domain; pass `--domain dev|personal|all` to widen.

To run expiry nightly, install the systemd user units in `contrib/systemd/`:

```bash
cp contrib/systemd/kipp-expire.* ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now kipp-expire.timer
```

Put any Qdrant env vars (`QDRANT_URL`, `QDRANT_API_KEY`) in
`~/.config/kipp/env`.

### Backup and restore

`kipp backup` takes a native Qdrant snapshot of every collection (all
domains), downloads it to `~/backups/qdrant/<UTC stamp>/`, checks each file
against the sha256 Qdrant reports, writes a `manifest.json` of point counts
and digests, and deletes the server-side copy. Only the newest 3 backups are
kept (`--keep`, `--dest` to change). A run that fails leaves nothing behind
under a stamp name, so every stamp directory is a complete backup.

Same disk as the data: this covers a bad migration or an accidental delete,
not drive failure. Copy the directory elsewhere if you need that.

Install the daily timer (04:45, after expiry) the same way:

```bash
cp contrib/systemd/kipp-backup.* ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now kipp-backup.timer
```

Restore one collection (replaces its contents):

```bash
curl -X POST "http://127.0.0.1:6333/collections/<name>/snapshots/upload?priority=snapshot" \
     -H "api-key: $QDRANT_API_KEY" \
     -F snapshot=@$HOME/backups/qdrant/<stamp>/<name>.snapshot
```

## OpenClaw skill

Install as a skill:
```bash
bash scripts/install.sh
```

See `skill/SKILL.md` for agent integration instructions.

## Development

```bash
pip install -e .
python3 -m pytest tests/ -v
```

See `MEMORY-SYSTEM-SPEC.md` for the full specification.
