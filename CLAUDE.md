# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KIPP (renamed from `entity-memory`; repo was `memory-system-skill`): entity-based persistent memory for AI agents, used by Endurance and OpenClaw. Stores knowledge as discrete entities (upserted in place) rather than appending documents. Growth is proportional to unique concepts, not time. Published as a ClawHub community skill.

The full spec is in `MEMORY-SYSTEM-SPEC.md`.

## Tech Stack

- **Language:** Python
- **Vector DB:** Qdrant (Docker container, ports 6333/6334, localhost-only)
- **Embeddings:** `all-MiniLM-L6-v2` via sentence-transformers (384 dims, local, no API calls)
- **CLI framework:** Click
- **Package manager:** pip (pyproject.toml)

## Build & Run Commands

```bash
# Install in dev mode
pip install -e .

# Run CLI
kipp <command>

# Start Qdrant
docker compose up -d

# Initialize collections
kipp init

# Run tests
pytest
pytest tests/test_merge.py           # unit tests (no Qdrant needed)
pytest tests/test_search.py          # search tests (in-memory Qdrant)
pytest tests/test_merge.py::test_name  # single test
```

## Architecture

Collections are organised on two axes — **domain** (`shared`, `dev`, `personal`) × **kind** (`entities`, `events`, `decisions`, `facts`). Twelve collections in total. The `shared` domain reuses the legacy bare names (`entities`, `events`, `decisions`, `facts`) so OpenClaw and any other existing caller continue to work unchanged. The `dev` and `personal` domains use namespaced names (`dev_entities`, `personal_events`, etc.).

The four kinds:
- **`entities`** — long-term knowledge (person, project, tool, preference, decision). Upserted in place, never appended.
- **`events`** — short-term observations (rolling 30-day window, auto-expires). Source material for entity extraction.
- **`decisions`** — architectural/strategic choices. Same structure as entities but permanent, never expire.
- **`facts`** — search index: one point per fact of every entity/decision, vector from `"[type] id: fact"`. Derived, never the record of truth; superseded facts stay indexed with `current: false`. Write entities through `facts.write_entity` (not `upsert_entity`) so the index stays in sync; `kipp reindex` backfills it.

### Key modules (`src/kipp/`)

| File | Purpose |
|------|---------|
| `cli.py` | Click CLI entry point for all commands |
| `config.py` | Config resolution and `DOMAINS`, standard library only (re-exported by `client.py`) |
| `client.py` | Qdrant client wrapper, collection initialization, domain-aware CRUD |
| `embedder.py` | sentence-transformers wrapper |
| `models.py` | Dataclasses: Fact, Entity, Event |
| `merge.py` | Merge, compact, dedup, search text (keyword payload) and vector text (clipped, issue #49) |
| `extract.py` | Event → entity extraction (sentence splitting, matching) |
| `facts.py` | Per-fact search points: `write_entity` (the one write path), sync, orphan prune, fact search |
| `search.py` | Entity vector + fact vector (max) + text + filter search fusion |
| `maintenance.py` | expire, compact, `reindex` (vectors, `search_text`, fact points) |
| `export.py` | Export/import (JSON and markdown formats) |
| `evaluate.py` | `kipp eval`: recall scoring against a JSONL known-answer case file |
| `split.py` | Split oversized entities: `plan_split` (Ward clustering), `move_facts` (target first, never loses a fact), `apply_plan` |
| `mcp_server.py` | MCP stdio server (`kipp mcp`) exposing operations as tools |
| `capture.py` | `kipp-capture`: transcript → per-turn spool records (hook path, standard library only, never imports Qdrant or the model; `KIPP_CAPTURE=off` disables the hook). Setup: `docs/capture.md` |
| `flush.py` | `kipp capture flush`: spool → events (`source="capture"`), one event per text of a turn (`uuid5(turn id, sha256(text))`), older unconsolidated versions retired |

### MCP entrypoint

`kipp mcp` runs an MCP stdio server using the official Python SDK (`mcp` package). Tools mirror the CLI verbs and add a `domain` parameter for routing:

| Tool | Maps to |
|------|---------|
| `memory_store` | `kipp store` |
| `memory_get` | `kipp get` |
| `memory_list` | `kipp list` |
| `memory_search` | `kipp search` |
| `memory_event` | `kipp event` (also accepts `run_id`/`session_id`/`profile`/`trigger_source` for provenance) |
| `memory_extract` | `kipp extract` |
| `memory_move` | `split.move_facts` (what `kipp split-apply` runs per group) |
| `memory_embed` | direct embedder access (lets MCP clients reuse the same MiniLM model) |
| `memory_stats` | `kipp stats` |

CLI commands default to `domain="shared"`, so OpenClaw keeps working without modification. `store`, `get`, `delete`, `list`, `search`, `event`, `extract`, `split-plan` and `dismiss` take `--domain`; `expire`, `compact`, `reindex`, `oversized` and `export` also accept `all`. `import` restores each record to its exported `domain` (records without one go to `shared`); its `--domain` filters, default `all`.

### Core algorithms (all deterministic, no LLM)

- **Merge:** New facts compared by cosine similarity. >= 0.9 = duplicate (increment hit_count, keep longer text). < 0.9 = new fact appended.
- **Size:** No cap (#52). Past `SPLIT_THRESHOLD` (30) current facts, `memory_store` returns an `oversized` note (separate from `warning`) suggesting topic entities named `<id>-<topic>`; `kipp oversized` lists them; `kipp split-plan`/`split-apply` split one, and topic entities carry `parent` (`memory_get` lists `children`). `kipp compact` (keep top `MAX_FACTS` by `frequency × recency × permanence`) is manual and lossy; nothing schedules it.
- **Extraction:** An event opening with a `[type:id]` tag routes all its sentences to that entity if it exists; otherwise it stays unextracted for the LLM agent to create. Untagged events are cosine-matched (per fact, 0.32) only with `match_untagged` / `--match-untagged` — off by default because it misattributed most facts (#33). Nothing is auto-created as an entity.

### Constants

```python
DUPE_THRESHOLD = 0.9
MAX_FACTS = 20          # kipp compact default only
SPLIT_THRESHOLD = 30
EVENT_TTL_DAYS = 30     # config.py
HALF_LIFE_DAYS = 30
```

## Testing

- Unit tests use `MockEmbedder` (deterministic hash-based vectors, 48 dims). Tests merge mechanics, not semantic quality.
- Every test runs against an in-memory Qdrant (`QdrantClient(":memory:")`, `conftest.py` fixtures); no container needed.
- Config lives at `~/.config/kipp/config.json` (falls back to `~/.config/entity-memory/config.json`, then `~/.openclaw/memory.json`), or env vars (`QDRANT_URL`, `KIPP_CONFIG`). Qdrant API key via `QDRANT_API_KEY` env var.
