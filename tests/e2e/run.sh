#!/usr/bin/env bash
# Browser test of the add-camera form. Needs Node with Playwright and a Chromium.
#   tests/e2e/run.sh [screenshot_dir]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
export COLORGRADER_HOME="$(mktemp -d)"
PORT=8799
CLIP="$COLORGRADER_HOME/sony_slog3.mov"
cd "$ROOT"
PYTHONPATH="$ROOT/tests:$ROOT" "$PY" -W ignore -c "
from pathlib import Path
from colorgrader import cameras
from helpers import slog3_clip
cameras.seed()
slog3_clip(Path('$CLIP'), frames_each=30, size=(640, 360))
"
"$PY" -W ignore -m colorgrader --no-browser --port $PORT >"$COLORGRADER_HOME/server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
for i in $(seq 1 40); do curl -s -o /dev/null "http://127.0.0.1:$PORT/api/state" && break; sleep 0.5; done
node "$ROOT/tests/e2e/add_camera_form.js" "http://127.0.0.1:$PORT/" "$CLIP" "${1:-}"
