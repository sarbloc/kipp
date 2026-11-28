# Teaching your agent to use KIPP

Connecting the MCP server gives your agent the tools. It still has to be told
when to use them, and something has to turn its raw observations into
entities. This page covers both:

1. [A system-prompt section](#system-prompt-section) for the agent you talk to.
2. [The extraction pass](#the-extraction-pass), a periodic job that files
   events onto entities.

## System-prompt section

Paste this into your agent's instructions: `CLAUDE.md` for Claude Code, a
project's custom instructions in Claude Desktop, `.cursor/rules` in Cursor, or
`AGENTS.md` for Codex. Edit the examples to suit you.

```markdown
## Memory (KIPP)

You have persistent memory through the KIPP tools (memory_search,
memory_store, memory_event, memory_get). It holds one record per person,
project, tool, preference or decision, and outlives this conversation.

**Recall first.** Before answering anything that depends on the past, such as
a person, a project's state, an earlier decision or a stated preference, call
memory_search with a short query. Treat hits as things you were told before,
not as verified current truth. If memory and what you can see now disagree,
trust what you can see and update memory.

**Record what's durable.** Write to memory, without asking, when you learn:
- a preference ("prefers short answers", "uses pnpm, not npm");
- a decision and its reason;
- a lasting fact about a person, project or tool;
- a name, relationship or identifier worth keeping.

Don't record small talk, status updates, today-only state, brainstorms that
didn't land, or anything already in memory (search first).

**How to write.**
- If you know the subject and the fact is clear, call memory_store with
  entity_type (person | project | tool | preference | decision), a short
  lowercase entity_id (`alice`, `billing-api`), and the fact as ONE short,
  self-contained sentence.
- If reality changed, pass `supersedes` with the old fact's exact text, so the
  old fact becomes history instead of contradicting the new one.
- If you're unsure where a fact belongs, log it with memory_event instead and
  leave it for the extraction pass. Open the text with a `[type:id]` tag when
  you know the subject: `[person:alice] Speaks fluent Portuguese.`
- Tagged events can only add facts. A fact that replaces an old one always
  goes through memory_store with `supersedes`.

Don't announce memory writes. Just make them.
```

### Store or event?

- **`memory_store`** writes straight onto an entity. Use it when the agent is
  sure of the subject. It's immediate and needs no extraction pass.
- **`memory_event`** drops an observation in a 30-day inbox. Use it for things
  noticed mid-task, facts whose subject is unclear, or anything you'd rather
  have reviewed. Nothing reaches an entity until the extraction pass runs.

An agent that only uses `memory_store` works fine without an extraction pass.
Events are for when you want capture to be cheap and filing to happen later.

Agents still forget to write. To record conversations without relying on
them, see [Capturing conversations](capture.md): a Claude Code hook turns
every exchange into an event for the extraction pass.

### Tags

An event that **opens** with `[type:id]`, with no spaces inside the brackets,
is filed onto that entity automatically by `memory_extract`:

```
[project:billing-api] Retries now back off exponentially, capped at 30s.
```

- **Entity must exist.** If it doesn't yet, the event waits for the extraction
  pass, which creates it.
- **Tag first.** Brackets later in the text are ignored. Provenance brackets
  (`[run_id:...]`, `[session_id:...]`, `[profile:...]`) are skipped, but the
  subject tag should still come first.
- **No tag.** Untagged events always wait for the extraction pass.
- **Only tag what you're sure of.** A wrong tag files the fact on the wrong
  entity.
- **Additions only.** Tag filing merges a fact in, but never marks an old fact
  superseded. "Moved to the payments team" filed by tag leaves her old team
  current too. Use `memory_store` with `supersedes` for changes.

### Writing good facts

- **One sentence, one fact, key words first.** Each fact is searched on its
  own, so a fact that mixes topics matches none of them well. The entity's
  own vector reads only the first 16 words of each of its top 10 facts.
  `memory_store` returns a `warning` for any fact over 40 words.
- **Narrow entities.** Past 30 current facts, `memory_store` returns an
  `oversized` note. File new facts on a topic entity named `<id>-<topic>`
  instead (`person:alice-health`, `person:alice-career`), passing
  `parent: "person:alice"` when you create it. Keep using the same topic ids
  once they exist: `memory_get` on the parent lists them as `children`.
- **Moving facts.** `memory_move` moves facts, by exact text, to another
  entity in the same domain, creating it as a child if it doesn't exist. Use
  it to split an entity or to fix a fact filed in the wrong place.
- **Self-contained.** Write "Alice leads the billing team", not "She leads
  it".
- **Stable ids.** `project:billing-api` every time. A second id for the same
  thing splits its history.

### Domains

KIPP has three domains: `shared`, `dev` and `personal`. Each is a separate set
of collections. Every tool takes a `domain` argument that defaults to
`shared`.

- **One agent, one person:** use `shared` for everything and ignore domains.
  The CLI's `store`, `search`, `get`, `list` and `event` commands only use
  `shared`.
- **Several agents or contexts:** a coding agent and a household agent, say.
  Give each its own domain so its subject matter stays separate, and keep
  people and standing preferences in `shared`. Tell each agent its domain in
  its prompt, and pass it on reads as well as writes. A search in the wrong
  domain misses the entity you meant to update, and you end up with a
  duplicate.

## The extraction pass

Events need a periodic pass to become entities. It has two steps:

1. **`memory_extract`** files tagged events onto existing entities. It's
   deterministic: no model, no judgement.
2. **An agent** reviews what's left (`memory_events_unextracted`), creates or
   updates entities with `memory_store`, and marks each event handled with
   `memory_event_resolve`.

The second step needs judgement: what's durable, which entity a fact belongs
to, whether it's a duplicate. So KIPP leaves it to an LLM agent rather than
guessing with similarity scores.

[`contrib/extraction-prompt.md`](../contrib/extraction-prompt.md) is a
ready-made prompt for that agent. It processes up to 40 events per domain per
run, so a large backlog drains over a few runs. Edit its `Domains to process:`
line if you use more than `shared`.

### Running it nightly with Claude Code

[`contrib/kipp-extract.sh`](../contrib/kipp-extract.sh) flushes any captured
turns into events ([capture](capture.md)), then runs the prompt with
`claude -p` and limits the agent to the KIPP tools:

- `--strict-mcp-config` with its own `--mcp-config`, so none of the other MCP
  servers in your Claude Code config are loaded;
- `--tools ""`, so there's no shell, file or web access;
- `--allowedTools mcp__kipp`, so the KIPP calls run without prompting;
- `--settings '{"disableAllHooks": true}'`, so none of your hooks run,
  including a capture hook that would record the pass itself.

That matters because the agent reads raw event text, and an event can contain
anything your agent captured, including text from web pages. The script
resolves `kipp` and `claude` to absolute paths itself, so it also works under
cron's minimal `PATH`. Set `KIPP_BIN` or `CLAUDE_BIN` if they aren't in
`~/.local/bin`, `/usr/local/bin` or `/opt/homebrew/bin`.

Run it once by hand from your KIPP checkout (the examples assume `~/kipp`) and
read the summary:

```bash
~/kipp/contrib/kipp-extract.sh
```

Then schedule it, for example with cron:

```bash
mkdir -p ~/.local/state      # for the log; cron won't create it
crontab -e
```

```cron
# m h dom mon dow
30 3 * * * $HOME/kipp/contrib/kipp-extract.sh >> $HOME/.local/state/kipp-extract.log 2>&1
```

Any other agent runner that can call MCP tools works the same way: give it the
prompt and only the KIPP tools.

### If you skip it

Events expire after 30 days. If nothing extracts them, untagged observations
are simply lost. That's fine if your agent mostly uses `memory_store`. Check
the backlog with `kipp stats`; `events` is the shared domain's inbox.

## Tool reference

| Tool | Use |
|------|-----|
| `memory_search` | Semantic search over entities in a domain. `as_of` shows the state on a past date. |
| `memory_get` | One entity, every fact, including superseded history, plus its `parent` and `children`. |
| `memory_list` | All entity ids in a domain, optionally filtered by type. |
| `memory_store` | Add a fact to an entity, creating it if needed. Dedups automatically. `parent` links a topic entity. |
| `memory_move` | Move facts to another entity in the same domain. Superseded history follows. |
| `memory_event` | Log a raw observation to the domain's 30-day inbox. |
| `memory_extract` | File tagged events onto existing entities. |
| `memory_events_unextracted` | Events still waiting to be filed. |
| `memory_event_resolve` | Mark an event handled, with the entity it fed. |
| `memory_stats` | Point counts for every collection. |
| `memory_embed` | The MiniLM vector for a string, for clients that want the same embedder. |
