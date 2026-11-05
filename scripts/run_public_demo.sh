#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT_DIR"

API_HOST="${FINRAG_API_HOST:-127.0.0.1}"
API_PORT="${FINRAG_API_PORT:-8000}"
STREAMLIT_HOST="${FINRAG_STREAMLIT_HOST:-127.0.0.1}"
STREAMLIT_PORT="${FINRAG_STREAMLIT_PORT:-8501}"
LOG_DIR="${FINRAG_DEMO_LOG_DIR:-$ROOT_DIR/.demo-logs}"
mkdir -p "$LOG_DIR"

api_pid=
streamlit_pid=
tunnel_pid=

cleanup() {
    status=$?
    trap - EXIT INT TERM
    for pid in "$tunnel_pid" "$streamlit_pid" "$api_pid"; do
        if [ -n "$pid" ]; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    exit "$status"
}
trap cleanup EXIT INT TERM

echo "Starting FastAPI on ${API_HOST}:${API_PORT}..."
if curl -fsS --max-time 2 "http://${API_HOST}:${API_PORT}/health" >/dev/null 2>&1; then
    echo "Using the existing healthy FastAPI process on ${API_HOST}:${API_PORT}."
else
    FINRAG_API_HOST="$API_HOST" FINRAG_API_PORT="$API_PORT" \
        uv run uvicorn finrag_indic.api.app:app \
        --host "$API_HOST" --port "$API_PORT" \
        >"$LOG_DIR/fastapi.log" 2>&1 &
    api_pid=$!
fi

uv run python - "$API_HOST" "$API_PORT" "${api_pid:-}" <<'PY'
import sys
import time
import os
from urllib.request import urlopen

host, port, pid = sys.argv[1:]
url = f"http://{host}:{port}/health"
for _ in range(60):
    try:
        with urlopen(url, timeout=2):
            break
    except Exception:
        if pid:
            try:
                os.kill(int(pid), 0)
            except OSError:
                raise SystemExit(
                    f"FastAPI process exited before becoming ready at {url}"
                )
        time.sleep(1)
else:
    raise SystemExit(f"FastAPI did not become ready at {url}")
PY

echo "Starting Streamlit on ${STREAMLIT_HOST}:${STREAMLIT_PORT}..."
if curl -fsS --max-time 2 \
    "http://${STREAMLIT_HOST}:${STREAMLIT_PORT}/_stcore/health" >/dev/null 2>&1; then
    echo "Using the existing healthy Streamlit process on ${STREAMLIT_HOST}:${STREAMLIT_PORT}."
else
    FINRAG_FRONTEND_API_URL="http://${API_HOST}:${API_PORT}" \
        uv run streamlit run frontend/streamlit_app.py \
        --server.address "$STREAMLIT_HOST" \
        --server.port "$STREAMLIT_PORT" \
        --server.headless true \
        >"$LOG_DIR/streamlit.log" 2>&1 &
    streamlit_pid=$!
fi

uv run python - "$STREAMLIT_HOST" "$STREAMLIT_PORT" "${streamlit_pid:-}" <<'PY'
import sys
import time
import os
from urllib.request import urlopen

host, port, pid = sys.argv[1:]
url = f"http://{host}:{port}/_stcore/health"
for _ in range(60):
    try:
        with urlopen(url, timeout=2):
            break
    except Exception:
        if pid:
            try:
                os.kill(int(pid), 0)
            except OSError:
                raise SystemExit(
                    f"Streamlit process exited before becoming ready at {url}"
                )
        time.sleep(1)
else:
    raise SystemExit(f"Streamlit did not become ready at {url}")
PY

echo "Starting Cloudflare Quick Tunnel for Streamlit only..."
cloudflared tunnel --url "http://${STREAMLIT_HOST}:${STREAMLIT_PORT}" \
    >"$LOG_DIR/cloudflared.log" 2>&1 &
tunnel_pid=$!

public_url=
for _ in $(seq 1 30); do
    public_url=$(grep -Eo 'https://[-a-z0-9]+\.trycloudflare\.com' \
        "$LOG_DIR/cloudflared.log" 2>/dev/null | head -1 || true)
    if [ -n "$public_url" ]; then
        break
    fi
    sleep 1
done

if [ -z "$public_url" ]; then
    echo "Cloudflare Quick Tunnel did not provide a public URL." >&2
    echo "Inspect $LOG_DIR/cloudflared.log for details." >&2
    exit 1
fi

printf '\nPublic demo URL: %s\n' "$public_url"
printf 'Local Streamlit: http://%s:%s\n' "$STREAMLIT_HOST" "$STREAMLIT_PORT"
printf 'Local FastAPI health: http://%s:%s/health\n' "$API_HOST" "$API_PORT"
printf 'Logs: %s\n' "$LOG_DIR"
printf 'Press Ctrl-C to stop FastAPI, Streamlit, and the tunnel.\n\n'

wait "$tunnel_pid"
