#!/usr/bin/env bash
# --------------------------------------------------------------------------- #
# Start the ImToLego Enterprise API
# --------------------------------------------------------------------------- #
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Defaults
HOST="${UVICORN_HOST:-0.0.0.0}"
PORT="${UVICORN_PORT:-8000}"
WORKERS="${UVICORN_WORKERS:-1}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

echo "========================================"
echo "  ImToLego Enterprise API"
echo "  Host:   $HOST"
echo "  Port:   $PORT"
echo "  Workers: $WORKERS"
echo "========================================"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

exec python -m uvicorn app.api.server:app \
  --host "$HOST" \
  --port "$PORT" \
  --workers "$WORKERS" \
  --loop uvloop \
  --log-level "$LOG_LEVEL"
