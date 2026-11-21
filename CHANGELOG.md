# Changelog

All notable changes to KIPP are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0]

The first release on PyPI, as `kipp-memory`. It installs the `kipp` command.

### Storage
- Entities hold facts and are updated in place. A new fact that repeats an
  existing one (cosine similarity ≥ 0.9) merges into it instead of being added.
- Data lives in three domains (`shared`, `dev`, `personal`), each with four
  kinds of collection (entities, events, decisions, facts). The `shared` domain
  keeps the original collection names, so existing callers keep working.
- Facts have dates. Each records when it became true; `supersedes` marks a fact
  as replaced while keeping it for `as_of` searches.
- An entity with over 30 current facts is flagged as oversized.
  `kipp split-plan` and `kipp split-apply` split it into topic entities, which
  record the entity they came from in `parent`. `memory_move` moves facts
  between entities and never loses one if it fails part way.

### Search
- Every fact is indexed on its own. Search scores each entity by its best match:
  the entity's own vector, its closest fact, or the keyword text. A detail buried
  in a large entity still surfaces.
- Each fact contributes only its first 16 words to the entity's vector, so every
  top fact reaches it.
- `memory_search` returns at least 10 facts per hit, best match first, then adds
  more while the hit stays under about 400 words.

### Extraction
- Events are raw observations, kept for 30 days. An event that starts with a
  `[type:id]` tag is filed on that entity. Cosine matching of untagged events is
  opt-in (`--match-untagged`).
- Events no existing entity matches are left for the agent to turn into new
  entities. Nothing is created automatically.

### Interfaces
- `kipp mcp` is an MCP stdio server. Its tools cover store, get, list, search,
  move, event, extract, listing and resolving unextracted events, embed, and
  stats, each taking a domain.
- On the command line, `store`, `get`, `delete`, `list`, `search`, `event`,
  `extract`, `split-plan` and `dismiss` take `--domain`. `memory` is an alias for
  `kipp`.
- `kipp eval` scores recall (hit@k and MRR) against a JSONL file of questions
  with known answers.

### Operations
- `kipp backup` snapshots every collection, checks each file, and keeps the
  newest 7 backups.
- `kipp expire`, `compact`, `reindex` and `oversized` work on one domain or all
  of them. Example systemd units are in `contrib/systemd/`.

[Unreleased]: https://github.com/sarbloc/kipp/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/sarbloc/kipp/releases/tag/v0.1.0
