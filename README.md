# FinRAG-Indic

FinRAG-Indic is a research implementation for an evidence-grounded financial
literacy assistant supporting English, Hindi, and Tamil. The project is being
built in reviewed phases so that implemented features, design proposals, and
experimentally validated behavior remain distinct.

## Current status

**Implemented in Phase 2:**

- Python 3.13 and `uv` project foundation.
- Environment-based Pydantic Settings configuration.
- FastAPI application with `GET /health`.
- Streamlit interface shell with a backend status check.
- JSON structured logging and request IDs.
- Replaceable text-provider boundary and deterministic test provider.
- Unit and integration test infrastructure.

**Approved but not yet implemented:**

- Curated RBI, NPCI, and SEBI document ingestion.
- BGE-M3 embeddings and FAISS retrieval.
- LLM-backed answer generation and citations.
- FABRIC-derived, human-reviewed evaluation records.
- Experiments A-D and evaluation metrics.

Nothing in the second list should be interpreted as completed or experimentally
validated.

## Requirements

- Python 3.13
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync --dev
cp .env.example .env
```

The `.env` file is local and ignored by Git. Do not place secrets in
`.env.example` or commit API keys.

## Run the backend

```bash
uv run uvicorn finrag_indic.api.app:app --reload --host 127.0.0.1 --port 8000
```

The health endpoint is available at <http://127.0.0.1:8000/health> and the
interactive API documentation at <http://127.0.0.1:8000/docs>.

## Run the frontend

In a second terminal:

```bash
uv run streamlit run frontend/streamlit_app.py
```

The frontend reads `FINRAG_FRONTEND_API_URL`, which defaults to
`http://127.0.0.1:8000`.

## Run tests

```bash
uv run pytest
```

Tests requiring live APIs must use the `live` marker and are never part of the
default deterministic test suite unless explicitly selected.

## Configuration

All application settings use the `FINRAG_` prefix. See `.env.example` for the
supported Phase 2 settings. The hosted LLM provider and model are intentionally
unset; `mock` is the safe default and performs no inference.

## Research integrity

FABRIC is a candidate source for evaluation questions, not independently
verified ground truth. Future records will preserve upstream provenance,
translation status, evidence mapping, and human-review status. No evaluation
score will be reported without actual experiment output.

