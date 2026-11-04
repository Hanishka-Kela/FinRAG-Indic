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

**Implemented in Phase 3:**

- Controlled source manifest with explicit URL-approval states.
- Official-domain acquisition guard with SHA-256 recording.
- PDF and HTML extraction, Unicode normalization, section/page-aware chunks,
  and duplicate-content removal.
- Pinned FABRIC acquisition and schema validation with a separate unreviewed
  human-review record.
- Replaceable embedding interface, deterministic mock, pinned BGE-M3 adapter,
  embedding cache, and local benchmark recorder.
- Persistent FAISS `IndexFlatIP` indexes with checked chunk mappings and model
  compatibility metadata.
- Separate English and multilingual corpus views and exploratory multilingual
  retrieval-pilot tooling.

**Remaining or proposed:**

- NPCI acquisition: the approved exact endpoint returned HTTP 403 and was not
  replaced or bypassed. RBI and SEBI PDFs were acquired and verified.
- A genuine Hindi/Tamil regulator corpus; the current multilingual view contains
  only genuine English source material.
- LLM-backed answer generation and citations.
- FABRIC-derived, human-reviewed evaluation records.
- Full experiments A-C and formal evaluation metrics. Experiment D currently
  has only an exploratory, ungraded direct-retrieval pilot.

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

Run the optional real-model test only after downloading the pinned weights:

```bash
FINRAG_RUN_LIVE_BGE=1 uv run pytest -m live tests/live/test_bge_m3.py
```

## Phase 3 data workflow

The three approved regulator URLs and their observed acquisition states are in
`configs/sources.json`. Acquisition never crawls linked pages or substitutes
failed sources:

```bash
uv run python scripts/acquire_sources.py
uv run python scripts/build_indexes.py
uv run python scripts/run_retrieval_pilot.py
```

FABRIC is independently pinned and can be prepared with:

```bash
uv run python scripts/prepare_fabric.py
```

This creates an unreviewed review template. It does not verify FABRIC answers,
translations, regulations, or citations.

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

See `docs/reproducibility.md` for pinned revisions, artifact locations, and
integrity checks.

