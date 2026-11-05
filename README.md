---
title: FinRAG-Indic
emoji: 💰
sdk: docker
app_port: 7860
---

# FinRAG-Indic

FinRAG-Indic is a research implementation for an evidence-grounded financial
literacy assistant supporting English, Hindi, and Tamil. The project is being
built in reviewed phases so that implemented features, design proposals, and
experimentally validated behavior remain distinct.

## Project overview

### Problem statement

Financial-literacy information is distributed across regulator documents, often
in English and difficult to navigate in regional languages. FinRAG-Indic
provides accessible explanations grounded in retrieved RBI and SEBI passages
while showing the source metadata used for factual claims.

### Objectives

- Support manual English, Hindi, and Tamil response selection.
- Retrieve multilingual evidence with pinned BGE-M3 and FAISS.
- Generate grounded explanations through an account-configured Gemini model.
- Abstain when evidence or provider responses are insufficient.
- Preserve provenance, citations, review flags, and reproducible experiments.

### Technology stack

Python 3.13, `uv`, FastAPI, Streamlit, Pydantic Settings, FAISS
`IndexFlatIP`, BGE-M3, the asynchronous OpenAI-compatible SDK, Gemini
Chat Completions, pytest, and Pillow for the Phase 5 PNG summary.

The full component flow and evaluation conditions are in
[docs/architecture.md](docs/architecture.md).

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
- Helpful abstention for personalized loan and investment decisions, including
  missing-information guidance and one localized clarification question.
- Response latency breakdown for retrieval, initial generation, citation
  correction, and total request time.
- `POST /chat`, `GET /health`, and `GET /system/status` FastAPI endpoints.
- Functional Streamlit chat, manual English/Hindi/Tamil selection, history,
  source expanders, backend status, loading state, and clear-chat control.

**Implemented in Phase 5:**

- Retrieval-only financial abbreviation normalization in production chat and
  evaluation runs.
- Separate LLM-only, immutable English RAG, and multilingual RAG conditions.
- Resumable incremental pilot runner with bounded retries and dry-run request
  estimates.
- Citation-reference, latency, success, abstention, language-signal, and
  retrieval-observation reporting.
- Reviewer CSV template and generated summary tables/figure.

**Phase 6 handoff:**

- Final integration audit and startup smoke checks completed.
- Architecture diagram: [docs/architecture.md](docs/architecture.md).
- Demonstration guide: [docs/demo_guide.md](docs/demo_guide.md).
- Research-paper handoff: [docs/research_handoff.md](docs/research_handoff.md).

**Known limitations:**

- NPCI acquisition: the approved exact endpoint returned HTTP 403 and was not
  replaced or bypassed. RBI and SEBI PDFs were acquired and verified.
- Human review of repaired Hindi text, drafted Hindi/Tamil questions, and all
  candidate evidence mappings.
- Hosted-model quality, latency, and cost depend on the configured Gemini
  account and quota.
- FABRIC-derived, human-reviewed evaluation records.
- The Phase 5 pilot is exploratory; the complete 135-request benchmark was not
  run and no human accuracy score is reported.

Items listed as remaining are not implemented or experimentally validated.

## Example questions

- English: `What is simple interest?`
- Hindi: `अगर UPI लेन-देन असफल हो जाए तो मुझे क्या करना चाहिए?`
- Tamil: `முதலீட்டில் பல்வகைப்படுத்தல் என்றால் என்ன?`
- Abbreviation handling: `What is SI?`

See [docs/demo_guide.md](docs/demo_guide.md) for a complete walkthrough,
including source expanders, abbreviation handling, helpful personalized
abstention, and an unsupported-question abstention example.

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

## Docker and Hugging Face Spaces deployment preparation

The repository includes a single-container deployment for Hugging Face Spaces:

- Public Streamlit listens on port `7860`, the Spaces default.
- FastAPI listens only on `127.0.0.1:8000` inside the container.
- `docker/start.sh` validates the v2 index, starts FastAPI, waits for its
  health endpoint, and then starts Streamlit.
- BGE-M3 is downloaded lazily at runtime into the container cache; the model
  cache is intentionally not copied into the image.

Build and run locally from a checkout that contains the existing v2 index:

```bash
docker build -t finrag-indic .
docker run --rm -p 7860:7860 \
  -e FINRAG_LLM_PROVIDER=openai \
  -e FINRAG_LLM_MODEL=gemini-flash-lite-latest \
  -e FINRAG_LLM_API_KEY=<space-secret> \
  -e FINRAG_LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/ \
  finrag-indic
```

For Hugging Face Spaces, create a Docker Space and configure these as Space
secrets or variables (never in the repository): `FINRAG_LLM_PROVIDER=openai`,
`FINRAG_LLM_MODEL`, `FINRAG_LLM_API_KEY`, and
`FINRAG_LLM_BASE_URL`. The Space must expose port `7860`. The
`data/indexes/multilingual-genuine-v2` directory must be present in the Space
build context; the Dockerfile does not rebuild or download an index.

Exact Space preparation:

1. Create a new Hugging Face Space, choose **Docker**, and set the exposed
   application port to `7860`.
2. Upload `Dockerfile`, `.dockerignore`, `docker/start.sh`, the `src/` and
   `frontend/` directories, `pyproject.toml`, `uv.lock`, `README.md`, and only
   the three files under `data/indexes/multilingual-genuine-v2/`.
3. If using Git from this checkout, add the deployment package with:

   ```bash
   git add Dockerfile .dockerignore docker/start.sh src frontend \
     pyproject.toml uv.lock README.md data/indexes/multilingual-genuine-v2
   ```

   The repository ignore rules keep the English baseline, other indexes,
   model cache, results, and local data out of this deployment package.
4. In the Space **Settings**, add `FINRAG_LLM_PROVIDER`, `FINRAG_LLM_MODEL`,
   `FINRAG_LLM_API_KEY`, and `FINRAG_LLM_BASE_URL` as secrets or variables.
5. Wait for the Space build logs to show FastAPI health readiness followed by
   Streamlit startup, then open the generated Space URL.

### Resource expectations and blockers

The v2 index itself is approximately 2 MB, but the pinned BGE-M3 model cache
is approximately 7.2 GB in the current development environment. The
container also needs memory for Python, PyTorch, FAISS, and BGE-M3 inference;
allocate at least 8 GB RAM and preferably 16 GB for a reliable cold start.
Provide at least 10 GB writable disk for the model cache and temporary files.
The application passes `data/models/sentence-transformers` as the
SentenceTransformer `cache_folder`, so in the container the effective model
cache is `/app/data/models/sentence-transformers`; the local 9.3 GB cache is
not copied into the image.
CPU-only inference is supported but the first model load can take several
minutes and each query may be slow. A Space with less memory or disk may fail
while loading BGE-M3.

The local index artifacts are ignored by Git in this project. Therefore,
public deployment is blocked until the v2 index is deliberately included in
the Space repository or supplied through an approved artifact mechanism.
No index rebuild is performed by this deployment preparation. The Docker
build fails early if the required index files are absent.

If `FINRAG_LLM_API_KEY` or `FINRAG_LLM_MODEL` is missing, the container still
starts but reports degraded status and chat requests cannot use hosted
generation. If the index is missing, startup exits with the specific missing
artifact. BGE-M3 loading remains lazy; model-loading failures are returned by
the backend as retrieval-unavailable errors rather than exposing credentials.

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

## Research handoff

The consolidated research-paper handoff is in
[docs/research_handoff.md](docs/research_handoff.md). It records the corpus,
architecture, experimental conditions, corrected pilot metrics, artifacts,
limitations, and reproducibility commands. No credentials, human scores, or
fabricated findings are included.
