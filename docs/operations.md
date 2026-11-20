# Running KIPP

Configuration, housekeeping and backups for a KIPP install. Finish the
[quick start](../README.md#quick-start) first.

## Configuration

With no configuration, KIPP talks to Qdrant at `http://127.0.0.1:6333` without
an API key, which is what the bundled `docker-compose.yml` runs.

Environment variables override that, no file needed:

| Variable | Meaning |
|----------|---------|
| `QDRANT_URL` | Qdrant endpoint. Takes precedence over any config file. |
| `QDRANT_API_KEY` | API key, if your Qdrant requires one. |
| `KIPP_CONFIG` | Path to a config file, replacing the default location. |

Or write `~/.config/kipp/config.json` (under `$XDG_CONFIG_HOME` if that's set
to an absolute path):

```json
{
  "qdrant": {
    "url": "http://127.0.0.1:6333",
    "api_key_env": "QDRANT_API_KEY"
  }
}
```

`api_key_env` names the environment variable that holds the key, so the key
itself never sits in the file. The pre-rename paths,
`~/.config/entity-memory/config.json` then `~/.openclaw/memory.json`, are
still read, in that order, if the new one is missing.

The CLI, the MCP server and the systemd units all read the same settings, so
configure it once.

### Requiring an API key

The bundled compose file leaves Qdrant open on localhost. To require a key,
start it with one set:

```bash
export QDRANT_API_KEY="$(openssl rand -hex 32)"
docker compose up -d --wait
```

Then make the same `QDRANT_API_KEY` visible to every KIPP process: your shell,
your MCP client's `env` block, and `~/.config/kipp/env` for the timers.

## Housekeeping

```bash
kipp expire                  # delete events past 30 days and facts past their expiry date
kipp oversized               # list entities over 30 current facts, all domains
kipp split-plan ID -o p.json # group an entity's facts into topic entities
kipp split-apply p.json      # move them (dry run; --apply to write)
kipp compact                 # lossy: delete all but each entity's top 20 facts
kipp stats                   # point count for every collection
kipp reindex                 # rebuild search vectors and fact points from stored facts
```

`expire`, `compact` and `reindex` default to the `shared` domain. Pass
`--domain dev`, `--domain personal` or `--domain all`.

An entity's vector is only rebuilt when the entity is written. After upgrading
to a release that changes how vectors are built, take a backup and run
`kipp reindex --domain all` once, so existing entities pick up the change.
`reindex` never modifies facts. Since issue #49, each entity's vector reads
the first 16 words of each of its top 10 facts. The full text is still stored,
returned and matched by keyword search.

Each fact also has its own point in its domain's `facts` collection
(`facts`, `dev_facts`, `personal_facts`). Search scores each entity by the
better of its entity vector and its best-matching fact, so one fact among many
still finds its entity. Fact points are derived from the entity and rebuilt on
every write. `kipp reindex` backfills them for entities written before they
existed and deletes points whose entity is gone. Until a domain has been
reindexed, its search uses entity vectors alone.

A long-running process (an MCP server, for example) writes entities with the
code it started with. Restart it on the new release before reindexing, or
reindex again after it restarts, so entities it writes in the meantime aren't
left without fact points.

Run `expire` daily (see [systemd timers](#systemd-timers)).

There is no fact cap (issue #52). Since every fact is searched on its own, a
big entity doesn't hurt recall much, but it does bloat `memory_get` and every
agent's context. Once an entity passes 30 current facts, `memory_store`
returns an `oversized` note and `kipp store` prints one: split it into topic
entities named `<id>-<topic>` (`person:alice-health`, `person:alice-career`).
Superseded facts don't count towards the 30. `kipp oversized` lists every
entity over the line.

### Splitting an entity

```bash
kipp backup
kipp split-plan person:alice --domain personal -o alice.json
# edit alice.json
kipp split-apply alice.json            # dry run: what would move where
kipp split-apply alice.json --apply
```

`split-plan` only reads. It groups the entity's current facts by topic
(Ward clustering of their embeddings, about 12 facts per group; `--groups N`
overrides) and writes JSON:

```json
{
  "source": "person:alice",
  "domain": "personal",
  "groups": [
    {"target": "person:alice-knee", "keywords": ["knee", "physio"], "facts": ["..."]}
  ],
  "keep": ["..."]
}
```

Rename each `target`, move facts between groups, delete groups you don't
want. Facts in `keep`, or in no group, stay on the source: leave its core
identity there. Groups of fewer than 3 facts start out in `keep`. Each fact
may appear only once.

`split-apply` moves each group with the same code as the `memory_move` MCP
tool:

- Facts match by exact text and move unchanged: dates, source and hit count.
- Superseded facts follow the fact that replaced them (by `superseded_by`).
  Unlinked history stays on the source.
- A target that doesn't exist is created with the type from its id and
  `parent` set to the source. `kipp get` on the source lists its children.
- The target is written before the facts leave the source. If a run is
  interrupted, a fact can be on both, never on neither; run the same plan
  again to finish. Facts already moved are reported as such.
- Facts landing on an existing target are not similarity-merged, so a near
  duplicate stays a separate fact.

Deleting a parent leaves its children's `parent` pointing at nothing. Nothing
follows parent links except `children` in `kipp get` and `memory_get`.

`compact` still exists but deletes facts, so nothing schedules it. Use it
only after a backup, and only when losing the lowest-scored facts is fine.

## Recall evaluation

`kipp eval` scores search against questions whose answer you already know.
Run it before and after any change to search, merge or how facts are written,
and compare the numbers. It only reads.

```bash
kipp eval                          # uses ~/.config/kipp/eval.jsonl
kipp eval path/to/cases.jsonl --k 10
kipp eval --json                   # per-case ranks for scripting
```

A case file is JSON Lines, one question per line. Blank lines and lines
starting with `//` are skipped.

```json
{"query": "who leads the billing team", "expected": "person:alice"}
{"query": "...", "expected": ["project:a", "project:b"], "domain": "dev", "fact": "retries back off"}
```

- `expected`: the entity id that should answer, or a list where any id counts.
- `domain`: `shared` (default), `dev` or `personal`.
- `fact`: optional text from the fact that answers the question. The report
  then says whether that fact starts inside the part of the entity's search
  text the embedding model reads (`window:in`) or past it (`window:out`).
  Facts past the window can't influence the entity's vector.

The report gives hit@1, hit@5 and mean reciprocal rank overall, per domain and
per window group, then lists every question that missed the top 5.
[`contrib/eval.example.jsonl`](../contrib/eval.example.jsonl) matches the
README quick start.

Write questions the way you'd ask them, not by copying fact text, or the
scores will flatter search. A case file built from real memory holds personal
data, so keep it out of version control.

## Backup and restore

```bash
kipp backup
```

This takes a native Qdrant snapshot of every collection in every domain and:

- downloads it to `~/backups/qdrant/<UTC stamp>/`;
- checks each file against the sha256 that Qdrant reports;
- writes a `manifest.json` of point counts and digests;
- deletes the server-side copy.

The newest 3 backups are kept. `--keep N` changes that and `--dest DIR` moves
them. A failed run leaves nothing under a stamp name, so every stamp directory
is a complete backup.

The backups sit on the same disk as the data. They cover a bad migration or an
accidental delete, not a dead drive. Copy the directory elsewhere if you need
that.

To restore one collection, replacing its contents:

```bash
curl -X POST "http://127.0.0.1:6333/collections/<name>/snapshots/upload?priority=snapshot" \
     -H "api-key: $QDRANT_API_KEY" \
     -F snapshot=@$HOME/backups/qdrant/<stamp>/<name>.snapshot
```

Collection names are listed in the backup's `manifest.json`.

## systemd timers

`contrib/systemd/` has two user timers:

| Timer | Runs |
|-------|------|
| `kipp-expire.timer` | `kipp expire --domain all`, daily at 03:30 |
| `kipp-backup.timer` | `kipp backup`, daily at 04:45, after expiry |

Install them from the repo checkout:

```bash
cp contrib/systemd/kipp-*.service contrib/systemd/kipp-*.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now kipp-expire.timer kipp-backup.timer
systemctl --user list-timers 'kipp-*'
```

- **Binary path.** The units run `~/.local/bin/kipp`, where pipx puts it. If
  yours is elsewhere, edit `ExecStart` in both `.service` files.
- **Environment.** If you set `QDRANT_URL` or `QDRANT_API_KEY`, put them in
  `~/.config/kipp/env` as `KEY=value` lines. The units read that file.
- **Without a login session.** User timers only run while you're logged in,
  unless you enable lingering: `loginctl enable-linger "$USER"`.

On macOS, run the same two commands from `launchd` or `cron`.

### Moving from entity-memory units

Older installs have `entity-memory-expire` and `entity-memory-backup` units.
They keep working, because they call `memory`, which is now an alias for
`kipp`. To switch to the new names:

```bash
systemctl --user disable --now entity-memory-expire.timer entity-memory-backup.timer
rm ~/.config/systemd/user/entity-memory-{expire,backup}.{service,timer}
```

Then install the `kipp-*` units as above. They also read the old
`~/.config/entity-memory/env`, so existing settings carry over.
