#!/bin/bash
set -e

echo "Installing KIPP..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Pre-rename installs are a separate pipx app ("entity-memory") that owns a
# `memory` command importing the removed entity_memory package. Remove it so
# nothing calls the stale code; the command is now `kipp`.
if pipx list --short 2>/dev/null | grep -q '^entity-memory '; then
    echo "Removing pre-rename entity-memory install..."
    pipx uninstall entity-memory
fi
pipx install --force -e "$SCRIPT_DIR/.."

# Reuse any Qdrant already answering (an explicit QDRANT_URL, or a local one
# on 6333 under whatever container name) rather than starting a second one
# that would fight it for the port.
if [ -n "$QDRANT_URL" ]; then
    echo "Using Qdrant at $QDRANT_URL"
elif curl -sf http://127.0.0.1:6333/healthz >/dev/null 2>&1; then
    echo "Qdrant already running on 127.0.0.1:6333"
else
    echo "Starting Qdrant (docker compose)..."
    # --wait blocks until the healthcheck passes, so `kipp init` can't race it.
    docker compose -f "$SCRIPT_DIR/../docker-compose.yml" up -d --wait
fi

echo "Initializing collections..."
kipp init

echo "Done. KIPP is ready."
