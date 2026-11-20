# Connecting an MCP client

`kipp mcp` is a stdio MCP server. Any client that can launch a local command
and talk MCP over stdin/stdout can use it. Finish the
[quick start](../README.md#quick-start) first, so Qdrant is running and
`kipp search` works from your shell.

**The server name is up to you.** It sets the tool prefix your client shows.
These examples use `kipp`, so Claude Code sees `mcp__kipp__memory_search`. If
you name it something else, use that name wherever a prompt or a permission
rule refers to the tools.

**Use an absolute path for GUI apps.** pipx puts `kipp` in `~/.local/bin`,
which desktop apps often don't have on their `PATH`. Find the full path with:

```bash
command -v kipp        # e.g. /home/you/.local/bin/kipp
```

**Environment.** With the default local Qdrant, `kipp mcp` needs no
environment. If your Qdrant is elsewhere, pass `QDRANT_URL` in the client's
`env` block, or set the URL once in `~/.config/kipp/config.json` (see
[operations.md](operations.md#configuration)). An API key is only ever read
from the environment, so if Qdrant needs one, every client's `env` block must
include `QDRANT_API_KEY`.

**First call is slower.** The server starts in about a second. The first tool
call loads the embedding model, about 5 seconds when it's cached, longer if it
still has to download. Later calls take milliseconds. Running the quick start's
smoke test downloads the model in advance.

## Claude Code

```bash
claude mcp add --scope user kipp -- kipp mcp
claude mcp list          # kipp should show as connected
```

`--scope user` makes it available in every project. Use `--scope project` to
write a `.mcp.json` you can commit for a team instead. Pass environment with
`-e`, for example
`claude mcp add --scope user -e QDRANT_URL=http://qdrant.lan:6333 kipp -- kipp mcp`.

Claude Code asks before each tool call until you allow the tools. To allow
them all, add `"mcp__kipp"` to `permissions.allow` in
`~/.claude/settings.json`.

## Claude Desktop

Open **Settings → Developer → Edit Config**. That opens or creates
`claude_desktop_config.json`:

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "kipp": {
      "command": "/home/you/.local/bin/kipp",
      "args": ["mcp"]
    }
  }
}
```

Use the absolute path from `command -v kipp`. Add an `"env": { ... }` object
next to `args` if you need `QDRANT_URL` or `QDRANT_API_KEY`. Quit Claude
Desktop completely and restart it. If the server doesn't show up, check
`mcp-server-kipp.log` in Claude's log folder (`~/Library/Logs/Claude` on macOS,
`%APPDATA%\Claude\logs` on Windows, `~/.config/Claude/logs` on Linux).

## Cursor

Global: `~/.cursor/mcp.json`. Per project: `.cursor/mcp.json` in the project.

```json
{
  "mcpServers": {
    "kipp": {
      "type": "stdio",
      "command": "/home/you/.local/bin/kipp",
      "args": ["mcp"]
    }
  }
}
```

Cursor also takes `env` and `envFile`, and expands `${userHome}`, so
`"command": "${userHome}/.local/bin/kipp"` works on any machine. Restart Cursor
after editing. Server logs are in the Output panel under **MCP Logs**.

## Codex CLI

```bash
codex mcp add kipp -- kipp mcp
codex mcp list
```

That writes this to `~/.codex/config.toml`, which you can also edit by hand:

```toml
[mcp_servers.kipp]
command = "kipp"
args = ["mcp"]
```

Use `codex mcp add --env QDRANT_URL=... kipp -- kipp mcp` for environment.

## Anything else

Any stdio MCP client can launch it:

- **command:** `kipp` (absolute path recommended)
- **args:** `["mcp"]`
- **env:** optional `QDRANT_URL`, `QDRANT_API_KEY`, `KIPP_CONFIG`

## Checking it works

Ask your agent to "store a fact that my favourite editor is Helix, then search
memory for my editor". You should see a `memory_store` call followed by a
`memory_search` that returns `preference:editor` (or whichever id the agent
picked). From a shell, `kipp list` shows the same entity, because the CLI and
the MCP server share the `shared` domain.

Next: [teach your agent when to use these tools](agent-guide.md).
