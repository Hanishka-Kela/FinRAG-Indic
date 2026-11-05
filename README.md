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
- A checksum-locked 107-chunk English baseline and a separate 262-vector genuine
  multilingual extension index (112 English, 98 Hindi, 52 Tamil).
- A 15-question/45-variant review benchmark and human-verification-gated
  Recall@1/3/5 evaluation script.

**Implemented in Phase 3.5:**

- CFF glyph-level Hindi extraction repair and row-bounded RBI failed-transaction
  table extraction.
- A separate 269-vector `multilingual-genuine-v2` index while preserving the
  immutable English baseline and v1 index.
- Reviewer-ready 45-variant workbook and evaluation breakdowns that remain
  gated to human-approved evidence.

**Implemented in Phase 4:**

- Multilingual RAG pipeline using the pinned BGE-M3 model and
  `multilingual-genuine-v2` by default.
- Configurable asynchronous hosted provider through the official OpenAI SDK,
  plus the deterministic mock used by default for local development and tests.
- Grounded prompting, citation-marker validation, conservative abstention, and
  sensitive financial-information rejection.
- `POST /chat`, `GET /health`, and `GET /system/status` FastAPI endpoints.
- Functional Streamlit chat, manual English/Hindi/Tamil selection, history,
  source expanders, backend status, loading state, and clear-chat control.

**Remaining or not experimentally validated:**

- NPCI acquisition: the approved exact endpoint returned HTTP 403 and was not
  replaced or bypassed. RBI and SEBI PDFs were acquired and verified.
- Human review of repaired Hindi text, drafted Hindi/Tamil questions, and all
  candidate evidence mappings.
- Hosted-model quality, latency, and cost. No hosted model is assumed available
  and no live provider result is reported without configured access.
- FABRIC-derived, human-reviewed evaluation records.
- Full experiments A-C and formal evaluation metrics. Experiment D currently
  has only an exploratory, ungraded direct-retrieval pilot.

Items listed as remaining are not implemented or experimentally validated.

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
interactive API documentation at <http://127.0.0.1:8000/docs>. Runtime readiness
is available at <http://127.0.0.1:8000/system/status>.

## Run the frontend

In a second terminal:

```bash
uv run streamlit run frontend/streamlit_app.py
```

The frontend reads `FINRAG_FRONTEND_API_URL`, which defaults to
`http://127.0.0.1:8000`. The first chat request loads BGE-M3 and may be slower
than subsequent requests.

## Run tests

```bash
uv run pytest
```

Run the optional real-model test only after downloading the pinned weights:

```bash
FINRAG_RUN_LIVE_BGE=1 uv run pytest -m live tests/live/test_bge_m3.py
```

Run the hosted-provider smoke test only with an explicitly configured model and
credential:

```bash
FINRAG_LLM_PROVIDER=openai \
FINRAG_LLM_MODEL=<account-available-model> \
FINRAG_LLM_API_KEY=<secret> \
FINRAG_RUN_LIVE_LLM=1 \
uv run pytest -m live tests/live/test_live_openai_provider.py
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

## Phase 5 exploratory evaluation

The Phase 5 runner compares LLM-only, immutable English-baseline RAG, and
`multilingual-genuine-v2` RAG. It expands common financial abbreviations for
retrieval, preserves the original question for generation, records retrieval
observations and citation-reference validity, appends results incrementally,
and supports resuming completed cases:

```bash
uv run python scripts/run_phase5_experiment.py --dry-run
uv run python scripts/run_phase5_experiment.py --limit 3 \
  --output results/raw/phase5_pilot.jsonl
uv run python scripts/run_phase5_experiment.py --resume
uv run python scripts/summarize_phase5.py --input results/raw/phase5_pilot.jsonl
```

The full benchmark is exploratory: translations and evidence mappings are not
human-verified, and reviewer score columns remain blank. The separate
`configs/benchmarks/phase5_unsupported_probes.json` file contains additional
unsupported probes for abstention analysis. The runner prints the planned
request count before making provider calls; do not start the full run without
reviewing quota and cost implications.

## Hosted LLM configuration

All application settings use the `FINRAG_` prefix. The hosted model remains
intentionally unset because model availability is account-specific. To enable a
configured provider, set:

```dotenv
FINRAG_LLM_PROVIDER=openai
FINRAG_LLM_MODEL=<model-available-to-your-account>
FINRAG_LLM_API_KEY=<secret>
```

Optional base URL, temperature, output-token, timeout, and retry settings are
listed in `.env.example`. Never commit `.env` or credentials. `mock` remains the
safe local default and performs no model inference.

## Phase 4 demonstration

The deterministic demonstration exercises real v2 retrieval and citations for
five topics in all three languages:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run python scripts/run_phase4_demo.py
```

Add `--use-configured-provider` only when hosted access is configured. Mock
answers are engineering fixtures and must not be presented as hosted-model
quality results.

## Research integrity

FABRIC is a candidate source for evaluation questions, not independently
verified ground truth. Future records will preserve upstream provenance,
translation status, evidence mapping, and human-review status. No evaluation
score will be reported without actual experiment output.

See `docs/reproducibility.md` for pinned revisions, artifact locations, and
integrity checks.
