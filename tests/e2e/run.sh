#!/usr/bin/env bash
# Browser tests: the add-camera form, the Color tab, the assistant cursor, the full-quality still and the Export panel. Needs Node with Playwright and a Chromium.
#   tests/e2e/run.sh [screenshot_dir]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
cd "$ROOT"
SERVERS=()
trap 'for s in "${SERVERS[@]}"; do kill "$s" 2>/dev/null || true; done; rm -rf "${HOME1:-}" "${HOME2:-}"' EXIT

serve() {   # serve <home> <port>
  COLORGRADER_HOME="$1" "$PY" -W ignore -m colorgrader --no-browser --port "$2" >"$1/server.log" 2>&1 &
  SERVERS+=($!)
  for i in $(seq 1 40); do curl -s -o /dev/null "http://127.0.0.1:$2/api/state" && break; sleep 0.5; done
}

# 1. add a camera through the form
HOME1="$(mktemp -d)"
CLIP="$HOME1/sony_slog3.mov"
COLORGRADER_HOME="$HOME1" PYTHONPATH="$ROOT/tests:$ROOT" "$PY" -W ignore -c "
from pathlib import Path
from colorgrader import cameras
from helpers import slog3_clip
cameras.seed()
slog3_clip(Path('$CLIP'), frames_each=30, size=(640, 360))
"
serve "$HOME1" 8799
node "$ROOT/tests/e2e/add_camera_form.js" "http://127.0.0.1:8799/" "$CLIP" "${1:-}"

# 2. the Color tab on a graded project
HOME2="$(mktemp -d)"
COLORGRADER_HOME="$HOME2" PYTHONPATH="$ROOT/tests:$ROOT" "$PY" -W ignore -c "
from pathlib import Path
import numpy as np
from colorgrader import cameras, project, settings, synth
from colorgrader.grade import auto
from helpers import flog2_clip, load_clip
cameras.seed()
settings.save({'auto_detail': False})
scenes = [synth.procedural_scene(640, 360), synth.procedural_scene(640, 360, seed=5)[:, ::-1] * np.float32(0.8)]
load_clip(project.get(), flog2_clip(Path('$HOME2/fuji_flog2.mov'), scenes=scenes, frames_each=20, size=(640, 360)))
auto.build(project.get(), refit=False)
"
serve "$HOME2" 8798
node "$ROOT/tests/e2e/color_panel.js" "http://127.0.0.1:8798/" "${1:-}"
node "$ROOT/tests/e2e/assistant_cursor.js" "http://127.0.0.1:8798/" "${1:-}"
node "$ROOT/tests/e2e/still_view.js" "http://127.0.0.1:8798/" "${1:-}"
node "$ROOT/tests/e2e/export_panel.js" "http://127.0.0.1:8798/" "${1:-}"
