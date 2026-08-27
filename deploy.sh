#!/usr/bin/env bash
# Sets up a venv, installs backend deps, and starts the backend (FastAPI/uvicorn) and
# frontend (static file server) as background processes. Run ./stop.sh to stop them.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"
VENV_DIR="$BACKEND_DIR/venv"
BACKEND_PORT="${BACKEND_PORT:-8010}"
FRONTEND_PORT="${FRONTEND_PORT:-8081}"
API_KEY_FILE="$SCRIPT_DIR/.api_key"

# Reuse a previously generated key across restarts unless API_KEY is explicitly set.
if [ -z "${API_KEY:-}" ]; then
  if [ -f "$API_KEY_FILE" ]; then
    API_KEY="$(cat "$API_KEY_FILE")"
  else
    API_KEY="$(openssl rand -hex 32)"
    echo "$API_KEY" > "$API_KEY_FILE"
    chmod 600 "$API_KEY_FILE"
  fi
fi
export API_KEY

echo "Setting up backend virtualenv at $VENV_DIR..."
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi
"$VENV_DIR/bin/pip" install --quiet --upgrade pip
"$VENV_DIR/bin/pip" install --quiet -r "$BACKEND_DIR/requirements.txt"

echo "Starting backend on port $BACKEND_PORT..."
cd "$BACKEND_DIR"
# PYTHONUNBUFFERED: without it Python buffers stdout when it's not a TTY, so
# backend.log stays empty for the whole ~30s model-load window instead of streaming.
PYTHONUNBUFFERED=1 nohup "$VENV_DIR/bin/uvicorn" api:app --host 0.0.0.0 --port "$BACKEND_PORT" \
  > "$SCRIPT_DIR/backend.log" 2>&1 &
echo $! > "$SCRIPT_DIR/backend.pid"

echo "Starting frontend on port $FRONTEND_PORT..."
cd "$FRONTEND_DIR"
nohup python3 -m http.server "$FRONTEND_PORT" \
  > "$SCRIPT_DIR/frontend.log" 2>&1 &
echo $! > "$SCRIPT_DIR/frontend.pid"

echo ""
echo "Backend:  http://0.0.0.0:$BACKEND_PORT  (PID $(cat "$SCRIPT_DIR/backend.pid"), log: backend.log)"
echo "Frontend: http://0.0.0.0:$FRONTEND_PORT (PID $(cat "$SCRIPT_DIR/frontend.pid"), log: frontend.log)"
echo "API key: $API_KEY  (saved to .api_key -- callers must send it as the X-API-Key header)"
echo "Backend model load takes a few seconds -- tail backend.log to watch startup."
echo "To stop: ./stop.sh"
