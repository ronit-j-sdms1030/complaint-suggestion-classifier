#!/usr/bin/env bash
# Stops the backend/frontend processes started by deploy.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for name in backend frontend; do
  pidfile="$SCRIPT_DIR/$name.pid"
  if [ -f "$pidfile" ]; then
    pid="$(cat "$pidfile")"
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid"
      echo "Stopped $name (PID $pid)"
    else
      echo "$name (PID $pid) was not running"
    fi
    rm -f "$pidfile"
  else
    echo "No $name.pid found -- not running (or not started via deploy.sh)"
  fi
done
