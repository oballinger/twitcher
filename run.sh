#!/bin/bash

# ── TfL JamCam Detection Stack ───────────────────────────────────────────────
# Usage: ./run.sh
# Starts the Python detection server and Node API server concurrently.
# Ctrl+C kills both.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

log()  { echo -e "${CYAN}[run]${RESET} $*"; }
ok()   { echo -e "${GREEN}[ok]${RESET}  $*"; }
warn() { echo -e "${YELLOW}[warn]${RESET} $*"; }
err()  { echo -e "${RED}[err]${RESET}  $*"; }

# ── Env check ─────────────────────────────────────────────────────────────────
if [ -z "$TFL_KEY" ]; then
  warn "TFL_KEY is not set. Export it before running:"
  warn "  export TFL_KEY=your_key_here"
  exit 1
fi

# ── Dependency checks ─────────────────────────────────────────────────────────
command -v python3 &>/dev/null || { err "python3 not found"; exit 1; }
command -v node    &>/dev/null || { err "node not found";    exit 1; }
command -v ffmpeg  &>/dev/null || warn "ffmpeg not found — video streams may not work"

python3 -c "import ultralytics, cv2, flask, pytesseract, requests" 2>/dev/null \
  || { err "Missing Python deps. Run: pip install ultralytics opencv-python flask requests pytesseract"; exit 1; }

(cd "$SCRIPT_DIR" && node -e "require('express'); require('cors'); require('node-fetch')") 2>/dev/null \
  || { err "Missing Node deps. Run: npm install express cors node-fetch"; exit 1; }

# ── Kill any existing processes on our ports ──────────────────────────────────
for PORT in 3000 5000; do
  PID=$(lsof -ti tcp:$PORT 2>/dev/null || true)
  if [ -n "$PID" ]; then
    warn "Port $PORT in use (PID $PID) — killing"
    kill -9 $PID 2>/dev/null || true
    sleep 0.3
  fi
done

# ── Cleanup on exit ───────────────────────────────────────────────────────────
cleanup() {
  echo ""
  log "Shutting down…"
  kill "$PY_PID" "$NODE_PID" 2>/dev/null || true
  wait "$PY_PID" "$NODE_PID" 2>/dev/null || true
  ok "Done."
}
trap cleanup INT TERM

# ── Start Python detection server ─────────────────────────────────────────────
log "Starting Python detection server on :5000"
python3 "$SCRIPT_DIR/detect.py" 2>&1 | sed "s/^/${YELLOW}[python]${RESET} /" &
PY_PID=$!

# Wait for Flask to be ready
for i in $(seq 1 20); do
  sleep 0.5
  curl -sf http://localhost:5000/ &>/dev/null && break
  if [ $i -eq 20 ]; then
    err "Python server failed to start"
    kill "$PY_PID" 2>/dev/null
    exit 1
  fi
done
ok "Python server ready"

# ── Start Node API server ──────────────────────────────────────────────────────
log "Starting Node API server on :3000"
node "$SCRIPT_DIR/server.js" 2>&1 | sed "s/^/${GREEN}[node]${RESET} /" &
NODE_PID=$!

sleep 1
if ! kill -0 "$NODE_PID" 2>/dev/null; then
  err "Node server failed to start"
  kill "$PY_PID" 2>/dev/null
  exit 1
fi
ok "Node server ready"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${BOLD}  TfL JamCam Detection Stack running${RESET}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "  ${CYAN}Frontend${RESET}  →  open frontend.html in browser"
echo -e "  ${CYAN}API${RESET}       →  http://localhost:3000"
echo -e "  ${CYAN}Detection${RESET} →  http://localhost:5000"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "  Press ${RED}Ctrl+C${RESET} to stop

  Collect data  →  python collect.py --workers 8 --fps 5"
echo ""

wait
