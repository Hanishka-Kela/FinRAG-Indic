FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    FINRAG_API_HOST=127.0.0.1 \
    FINRAG_API_PORT=8000 \
    FINRAG_FRONTEND_API_URL=http://127.0.0.1:8000 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_HEADLESS=true \
    HF_HOME=/app/data/models/huggingface \
    TRANSFORMERS_CACHE=/app/data/models/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/data/models/sentence-transformers

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /uvx /bin/
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY frontend ./frontend
COPY data/indexes/multilingual-genuine-v2 ./data/indexes/multilingual-genuine-v2
COPY docker/start.sh ./docker/start.sh
RUN uv sync --frozen --no-dev \
    && chmod +x ./docker/start.sh \
    && mkdir -p data/models

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health', timeout=3).read()"

ENTRYPOINT ["./docker/start.sh"]
