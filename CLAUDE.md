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
kipp <command>      # `memory` is a pre-rename alias; Endurance's MEMORY_BIN still uses it

# Start Qdrant
docker compose up -d

# Initialize collections
kipp init

# Run tests
pytest
pytest tests/test_merge.py           # unit tests (no Qdrant needed)
pytest tests/test_search.py          # integration tests (Qdrant must be running)
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
| `mcp_server.py` | MCP stdio server (`kipp mcp`) exposing operations as tools |

### MCP entrypoint

`kipp mcp` runs an MCP stdio server using the official Python SDK (`mcp` package). Tools mirror the CLI verbs and add a `domain` parameter for routing:

| Tool | Maps to |
|------|---------|
| `memory_store` | `memory store` |
| `memory_get` | `memory get` |
| `memory_list` | `memory list` |
| `memory_search` | `memory search` |
| `memory_event` | `memory event` (also accepts `run_id`/`session_id`/`profile`/`trigger_source` for provenance) |
| `memory_extract` | `memory extract` |
| `memory_embed` | direct embedder access (lets MCP clients reuse the same MiniLM model) |
| `memory_stats` | `memory stats` |

The CLI surface is unchanged: existing commands default to `domain="shared"` via the new kwargs in `client.py`. OpenClaw keeps working without modification.

### Core algorithms (all deterministic, no LLM)

- **Merge:** New facts compared by cosine similarity. >= 0.9 = duplicate (increment hit_count, keep longer text). < 0.9 = new fact appended.
- **Compaction:** Triggered when facts > 20. Score = `frequency × recency × permanence`. Keep top 20.
- **Extraction:** An event opening with a `[type:id]` tag routes all its sentences to that entity if it exists; otherwise it stays unextracted for the LLM agent to create. Untagged events are cosine-matched (per fact, 0.32) only with `match_untagged` / `--match-untagged` — off by default because it misattributed most facts (#33). Nothing is auto-created as an entity.

### Constants

```python
DUPE_THRESHOLD = 0.9
MAX_FACTS = 20
EVENT_TTL_DAYS = 30
HALF_LIFE_DAYS = 30
```

## Testing

- Unit tests use `MockEmbedder` (deterministic hash-based vectors, 48 dims). Tests merge mechanics, not semantic quality.
- Integration tests require a running Qdrant container. Use `conftest.py` fixtures for temporary collections.
- Config lives at `~/.config/kipp/config.json` (falls back to `~/.config/entity-memory/config.json`, then `~/.openclaw/memory.json`), or env vars (`QDRANT_URL`, `KIPP_CONFIG`). Qdrant API key via `QDRANT_API_KEY` env var.
