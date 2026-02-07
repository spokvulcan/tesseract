#!/usr/bin/env bash
# Create a Python virtual environment for TTS debug visualization.
# Usage: scripts/setup-debug-venv.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
    echo "To recreate, remove it first: rm -rf $VENV_DIR"
    exit 0
fi

echo "Creating virtual environment at $VENV_DIR ..."
python3 -m venv "$VENV_DIR"

echo "Installing dependencies ..."
"$VENV_DIR/bin/pip" install --quiet numpy matplotlib scipy

echo "Done. Run visualizations with:"
echo "  scripts/visualize-chunks.sh /tmp/tesseract-debug/<timestamp>/"
