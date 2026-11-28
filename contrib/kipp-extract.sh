#!/usr/bin/env bash
# Run the KIPP extraction pass headless with Claude Code.
#
# The agent gets the KIPP MCP tools and nothing else: no built-in tools, no
# other MCP servers from your Claude Code config, and no hooks (so a capture
# hook doesn't record this session as conversation). Safe for cron, which has
# a minimal PATH: every executable is resolved to an absolute path here.
#
# Captured turns (kipp-capture) are flushed into events first. With no spool
# that's a no-op that doesn't load the embedding model.
#
# Usage: contrib/kipp-extract.sh            (prompt: contrib/extraction-prompt.md)
# Env:   KIPP_BIN, CLAUDE_BIN override the executables; QDRANT_URL and
#        QDRANT_API_KEY are passed through to the MCP server if set.
set -euo pipefail

export PATH="$HOME/.local/bin:/usr/local/bin:/opt/homebrew/bin:$PATH"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

kipp_bin="${KIPP_BIN:-$(command -v kipp || true)}"
claude_bin="${CLAUDE_BIN:-$(command -v claude || true)}"
[[ -x "$kipp_bin" ]] || { echo "kipp-extract: kipp not found; set KIPP_BIN" >&2; exit 1; }
[[ -x "$claude_bin" ]] || { echo "kipp-extract: claude not found; set CLAUDE_BIN" >&2; exit 1; }

mcp_config="$(python3 -c '
import json, os, sys
env = {k: os.environ[k] for k in ("QDRANT_URL", "QDRANT_API_KEY", "KIPP_CONFIG") if os.environ.get(k)}
print(json.dumps({"mcpServers": {"kipp": {"command": sys.argv[1], "args": ["mcp"], "env": env}}}))
' "$kipp_bin")"

# A failed flush shouldn't block extracting what's already in events.
"$kipp_bin" capture flush || echo "kipp-extract: capture flush failed; extracting existing events" >&2

export KIPP_CAPTURE=off  # belt and braces: hooks are disabled below too
exec "$claude_bin" -p "$(cat "$here/extraction-prompt.md")" \
  --settings '{"disableAllHooks": true}' \
  --strict-mcp-config --mcp-config "$mcp_config" \
  --tools "" \
  --allowedTools mcp__kipp
