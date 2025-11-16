#!/usr/bin/env bash
set -euo pipefail

# Start FastAPI backend for CCD (IST endpoints)
# Usage: bash scripts/server/run_api.sh [--port 8000]

PORT="${1:-8000}"

ROOT_DIR="$(cd "$(dirname "${BUNDLE:-$0}")/../.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

echo "[INFO] Starting CCD API on :${PORT}"
python -m uvicorn ccd.server.main:app --host 0.0.0.0 --port "${PORT}" --reload


