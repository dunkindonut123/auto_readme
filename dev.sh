#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

source "$ROOT_DIR/../.venv/bin/activate"

cleanup() {
  if [[ -n "${API_PID:-}" ]]; then
    kill "$API_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

python "$ROOT_DIR/server.py" &
API_PID=$!

npm --prefix "$ROOT_DIR/web" run dev