#!/usr/bin/env bash
# Starts the full demo system: backend (FastAPI + AdvisoryPipeline, real LLM
# included) on :8000, and the static frontend on :5173. Ctrl+C stops both.
#
# Usage: ./start.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_PORT=8000
FRONTEND_PORT=5173

cleanup() {
  echo
  echo "Stopping backend (pid $BACKEND_PID) and frontend (pid $FRONTEND_PID)..."
  kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Starting backend on :$BACKEND_PORT (loading models, incl. the LLM -- this takes ~15-20s)..."
(
  cd "$ROOT_DIR/implementation"
  source .venv/bin/activate
  exec uvicorn src.server:app --host 0.0.0.0 --port "$BACKEND_PORT"
) &
BACKEND_PID=$!

echo "Starting frontend on :$FRONTEND_PORT..."
(
  cd "$ROOT_DIR/frontend"
  exec python3 -m http.server "$FRONTEND_PORT"
) &
FRONTEND_PID=$!

echo -n "Waiting for backend to become healthy"
until curl -s -o /dev/null "http://localhost:$BACKEND_PORT/health"; do
  echo -n "."
  sleep 1
done
echo
curl -s "http://localhost:$BACKEND_PORT/health"
echo
echo "=================================================================="
echo "  Backend  : http://localhost:$BACKEND_PORT  (see /health, /query)"
echo "  Frontend : http://localhost:$FRONTEND_PORT"
echo "=================================================================="
echo "Press Ctrl+C to stop both."

wait
