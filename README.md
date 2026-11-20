<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/brand/kipp-logo-dark.svg">
    <img src="docs/brand/kipp-logo.svg" alt="KIPP" width="360">
  </picture>
</h1>

Memory for AI agents that keeps one current record per thing, not an
ever-growing log. Local, deterministic, no LLM in the loop.

- **One record per thing.** Knowledge lives in entities (`person:alice`,
  `project:billing`) whose facts are updated in place. Storage grows with the
  number of distinct things your agent knows about, not with time.
- **Bounded.** Compaction trims each entity to its 20 best current facts,
  ranked by how often, how recently and how permanently they were asserted.
- **History kept.** A replaced fact is marked superseded, not deleted, so you
  can ask what was true on a past date. Compaction drops superseded facts
  first, so an entity that hits the cap loses its oldest history.
- **Local and deterministic.** Qdrant in Docker plus a small local embedding
  model. Merge, dedup, compaction and expiry are plain maths: no API calls, no
  LLM, same input gives the same result.
- **Works with any MCP client.** Claude Code, Claude Desktop, Cursor, Codex
  CLI, or anything else that speaks MCP over stdio.

## How it works

Your agent does two things with KIPP during a conversation:

1. **Recalls** with `memory_search` before answering questions about people,
   projects or past decisions.
2. **Records** what it learns, either straight onto an entity with
   `memory_store`, or as a raw observation with `memory_event`.

Events are a 30-day inbox. A periodic **extraction** pass files them onto
entities. Events that open with a `[type:id]` tag are filed automatically. The
rest are reviewed by an agent, which creates the entities they describe. KIPP
never invents entities on its own; that judgement stays with the agent.

[`docs/agent-guide.md`](docs/agent-guide.md) has the prompt that teaches an
agent this loop.

## Requirements

- Docker (for Qdrant)
- Python 3.11+ and [pipx](https://pipx.pypa.io/)
- About 1 GB of disk to start: the Qdrant image plus the embedding model
  (`all-MiniLM-L6-v2`, about 90 MB, downloaded from Hugging Face on first use)

Developed and run on Linux. macOS should work. On Windows, use WSL.

## Quick start

```bash
git clone https://github.com/sarbloc/kipp.git
cd kipp

# 1. Start Qdrant (localhost only, data in a named Docker volume)
docker compose up -d --wait

# 2. Install the CLI and MCP server
#    (if `kipp` is then "command not found", run `pipx ensurepath` and open a new shell)
pipx install .

# 3. Create the collections
kipp init

# 4. Smoke test. The first command downloads the embedding model.
kipp store --type person --id alice --content "Leads the billing team"
kipp search "who runs billing"
```

`kipp search` should print `person:alice` as the top hit. A warning about
unauthenticated requests to the HF Hub is harmless: the model is public, and
once it's cached nothing is downloaded again.

Then:

5. **Connect your AI client:** [`docs/mcp-clients.md`](docs/mcp-clients.md)
6. **Teach your agent to use it:** [`docs/agent-guide.md`](docs/agent-guide.md)
7. **Set up expiry and backups:** [`docs/operations.md`](docs/operations.md)

`scripts/install.sh` does steps 1–3 in one go, reusing a Qdrant that is
already running on port 6333.

## Docs

| Page | What it covers |
|------|----------------|
| [mcp-clients.md](docs/mcp-clients.md) | Registering `kipp mcp` in Claude Code, Claude Desktop, Cursor, Codex CLI |
| [agent-guide.md](docs/agent-guide.md) | System-prompt section, tools, tags, domains, the extraction loop |
| [operations.md](docs/operations.md) | Configuration, expiry, compaction, backup and restore, systemd timers |
| [MEMORY-SYSTEM-SPEC.md](MEMORY-SYSTEM-SPEC.md) | The original design spec |

## Architecture

- **Qdrant** stores nine collections: three **domains** (`shared`, `dev`,
  `personal`) × three **kinds**:
  - `entities`: long-term knowledge, upserted in place;
  - `events`: raw observations, expire after 30 days;
  - `decisions`: like entities, but never expire.
- **all-MiniLM-L6-v2** embeds text locally (384 dimensions).
- **Merge:** a new fact within cosine 0.9 of an existing one is a duplicate. The
  existing fact's hit count goes up and the longer wording wins. Anything else
  is appended.
- **No fact cap:** every fact is searched on its own, so entities can grow.
  Past 30 current facts, writes return an `oversized` note suggesting a split
  into topic entities (`kipp oversized` lists them; `kipp split-plan` and
  `kipp split-apply` do the split, and children point at their `parent`). `kipp compact` exists to
  trim to the top 20 by frequency × recency × permanence, but it deletes
  facts and nothing runs it.

The `shared` domain uses the bare collection names (`entities`, `events`,
`decisions`); the others are prefixed (`dev_entities`, ...).

## Upgrading from entity-memory

KIPP was called `entity-memory`. The pip distribution changed name, so pipx
treats it as a new app:

```bash
pipx uninstall entity-memory
pipx install -e .            # provides `kipp` and the `memory` alias
```

`scripts/install.sh` does this for you. The `memory` command, the old config
paths and `ENTITY_MEMORY_CONFIG` all still work. Qdrant data and collection
names are unchanged. To move installed systemd timers to the new unit names,
see [`docs/operations.md`](docs/operations.md#systemd-timers).

## OpenClaw

`skill/SKILL.md` is the OpenClaw skill manifest. Install with
`bash scripts/install.sh`.

## Development

```bash
pip install -e .
docker compose up -d --wait   # the integration tests need Qdrant
python3 -m pytest -q
```
