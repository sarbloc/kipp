---
name: kipp
description: >
  Entity-based persistent memory with semantic search. Store facts about people,
  projects, tools, decisions. Search by meaning. Sub-linear growth over time.
  Use when the agent needs to remember, recall, or log observations.
metadata: {"openclaw":{"emoji":"🧠","requires":{"bins":["kipp","docker"]}}}
---

# KIPP memory

You have persistent entity memory backed by Qdrant and local embeddings.
It stores facts about people, projects, tools, preferences, and decisions.
Facts are updated in place (upserted), not appended — the system stays lean over time.

## When to use

**After conversations or task completions**, log what happened:
```
kipp event "<what happened>"
```

**When you learn a clear, structured fact**, store it directly:
```
kipp store --type <person|project|tool|preference|decision> --id <key> --content "<fact>"
```

**Before answering questions about the past** or gathering context:
```
kipp search "<query>"
```

## Entity types

- `person` — people and their roles, preferences, contacts
- `project` — codebases, services, architecture details
- `tool` — tools, libraries, configurations
- `preference` — user/team preferences and conventions
- `decision` — architectural and strategic choices (permanent, never expire)

## Rules

- Never store secrets (API keys, tokens, passwords) in memory
- Prefer `event` for raw observations; let extraction handle entity creation
- Start event text with its subject tag, e.g. `[person:alice] Moved to Berlin.` — extraction routes tagged events to that entity deterministically; untagged events wait for an agent to file them
- Use `store` when you're confident about the entity type and key
- Use `search` before spawning sub-agents to provide relevant context

## All commands

| Command | Purpose |
|---------|---------|
| `kipp event "<text>"` | Log raw observation |
| `kipp store --type T --id K --content "..."` | Upsert entity with fact |
| `kipp search "<query>" [--type T] [--limit N]` | Semantic search |
| `kipp extract [--since 55m \| --all]` | Process events into entities |
| `kipp get <entity_id>` | Show entity detail |
| `kipp list [--type T]` | List all entities |
| `kipp delete <entity_id>` | Delete entity |
| `kipp oversized [--domain D] [--threshold N]` | List entities due a split into topic entities |
| `kipp compact [--max-facts N]` | Delete all but each entity's top facts (lossy, manual) |
| `kipp expire` | GC expired events and facts |
| `kipp export [--format json\|md]` | Backup entities |
| `kipp import <file>` | Restore from backup |
| `kipp stats` | Collection sizes |
| `kipp init` | Create collections |
