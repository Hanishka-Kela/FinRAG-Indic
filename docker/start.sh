#!/bin/sh
set -eu

index_path="${FINRAG_RETRIEVAL_INDEX_PATH:-data/indexes/multilingual-genuine-v2}"
for artifact in vectors.faiss metadata.json chunks.jsonl; do
    if [ ! -f "${index_path}/${artifact}" ]; then
        echo "ERROR: missing retrieval artifact: ${index_path}/${artifact}" >&2
        exit 1
    fi
done

if [ "${FINRAG_LLM_PROVIDER:-mock}" = "openai" ]; then
    if [ -z "${FINRAG_LLM_MODEL:-}" ] || [ -z "${FINRAG_LLM_API_KEY:-}" ]; then
        echo "WARNING: hosted LLM is not fully configured; the UI will show degraded status." >&2
    else
        echo "Hosted LLM configured for model ${FINRAG_LLM_MODEL}."
    fi
else
    echo "Using deterministic mock provider; set FINRAG_LLM_PROVIDER=openai for Gemini."
fi

echo "Starting internal FastAPI on 127.0.0.1:${FINRAG_API_PORT:-8000}."
uv run uvicorn finrag_indic.api.app:app \
    --host "${FINRAG_API_HOST:-127.0.0.1}" \
    --port "${FINRAG_API_PORT:-8000}" &
api_pid=$!

cleanup() {
    kill "$api_pid" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

python - <<'PY'
import os
import time
from urllib.request import urlopen

port = os.environ.get("FINRAG_API_PORT", "8000")
for _ in range(30):
    try:
        with urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
            break
    except Exception:
        time.sleep(1)
else:
    raise SystemExit("ERROR: internal FastAPI did not become ready")
PY

echo "Starting public Streamlit on 0.0.0.0:7860."
exec uv run streamlit run frontend/streamlit_app.py \
    --server.address "${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}" \
    --server.port "${STREAMLIT_SERVER_PORT:-7860}" \
    --server.headless true
