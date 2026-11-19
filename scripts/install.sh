#!/bin/bash
set -e

echo "Installing KIPP..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Pre-rename installs are a separate pipx app ("entity-memory") that owns the
# `memory` symlink and imports the removed entity_memory package. Remove it so
# kipp-memory can take over `memory` as an alias.
if pipx list --short 2>/dev/null | grep -q '^entity-memory '; then
    echo "Removing pre-rename entity-memory install..."
    pipx uninstall entity-memory
fi
pipx install --force -e "$SCRIPT_DIR/.."

echo "Pulling Qdrant Docker image..."
docker pull qdrant/qdrant:latest

# Start Qdrant if not already running. Match either the current or the legacy
# container name, so upgrading a host that still runs the old openclaw-memory
# container doesn't try to start a second Qdrant on the same port.
if ! docker ps --format '{{.Names}}' | grep -qE '^(kipp-qdrant|entity-memory|openclaw-memory)$'; then
    echo "Starting Qdrant container..."
    docker compose -f "$SCRIPT_DIR/../docker-compose.yml" up -d
fi

echo "Initializing collections..."
kipp init

echo "Done. KIPP is ready."
