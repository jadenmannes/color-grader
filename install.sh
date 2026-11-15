#!/usr/bin/env bash
# Color Grader: one-command install for macOS and Linux.   ./install.sh
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv (Python manager)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "Installing Python packages..."
uv sync --python 3.12

echo "Checking ffmpeg..."
uv run python -m colorgrader.setup_ffmpeg

echo
echo "Done. Start the app with ./start.sh"
