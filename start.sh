#!/usr/bin/env bash
# Color Grader: start the app.   ./start.sh
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"
exec uv run python -m colorgrader "$@"
