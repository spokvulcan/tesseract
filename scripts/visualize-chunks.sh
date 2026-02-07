#!/usr/bin/env bash
# Run TTS chunk boundary visualization on a debug dump directory.
# Usage: scripts/visualize-chunks.sh /tmp/tesseract-debug/<timestamp>/
#
# Automatically sets up the venv if it doesn't exist yet.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found. Setting up ..."
    "$SCRIPT_DIR/setup-debug-venv.sh"
    echo ""
fi

exec "$VENV_DIR/bin/python3" "$SCRIPT_DIR/visualize_chunks.py" "$@"
