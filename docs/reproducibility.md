# Reproducibility

## Environment

```bash
uv sync --dev --locked
uv run pytest
```

Python is pinned to 3.13 by `.python-version`; the complete dependency graph is
in `uv.lock`.

## FABRIC

```bash
uv run python scripts/prepare_fabric.py
shasum -a 256 data/raw/datasets/fabric/d8476ea6a9081f7dec4852200b379bfd1213ad47/train.parquet
```

Expected SHA-256:
`bcc8341d559d6c73f023104a562dfcaf74da9ad14ddade8571c93cbfbd28f8ee`.

The generated review template defaults every verification field to
`unreviewed`.

## Sources and indexes

The original manifest contains three approved URLs. The extension manifest
contains exactly four separately approved URLs. Acquisition continues after a
per-source failure and records the error without substitution:

```bash
uv run python scripts/acquire_sources.py
uv run python scripts/build_indexes.py
uv run python scripts/run_retrieval_pilot.py

# Phase 3 extension only
uv run python scripts/acquire_sources.py \
  --manifest configs/sources_extension.json \
  --raw-root data/raw/sources-extension
HF_HUB_OFFLINE=1 uv run python scripts/build_multilingual_extension.py
HF_HUB_OFFLINE=1 uv run python scripts/prepare_retrieval_benchmark.py
HF_HUB_OFFLINE=1 uv run python scripts/evaluate_retrieval.py

# Phase 3.5: prepare chunks separately so PDF/font and FAISS/PyTorch native
# libraries do not share the extraction process on macOS.
HF_HUB_OFFLINE=1 uv run python scripts/build_quality_index.py --prepare-only
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  FINRAG_EMBEDDING_BATCH_SIZE=1 \
  uv run python scripts/embed_quality_chunks.py
HF_HUB_OFFLINE=1 uv run python scripts/prepare_retrieval_benchmark.py \
  --multilingual-index data/indexes/multilingual-genuine-v2 \
  --output configs/benchmarks/retrieval_v2_candidates.json
uv run python scripts/evaluate_retrieval.py \
  --output data/processed/retrieval_evaluation_v2.json
```

Raw sources, embeddings, FAISS indexes, and raw outputs are Git-ignored.
Manifests, code, schemas, and pinned revisions are versioned.

The initial English build produced 107 vectors (33 RBI and 74 SEBI). Its three
artifact hashes are locked in `configs/baselines/english_v1.json`; the extension
builder refuses to proceed if any differs. The extension produced 262 vectors:
112 English, 98 Hindi, and 52 Tamil. It is stored separately at
`data/indexes/multilingual-genuine-v1`.

The rejected FAQ challenge response is preserved and checksummed but marked
failed. It contributes no chunks.

The Phase 3.5 build produced 269 vectors: 120 English, 97 Hindi, and 52
Tamil. Its corpus view is `multilingual-genuine-v2`. The separate preparation
step validates the English baseline lock before writing the v2 chunk artifact.
Embedding workers use small isolated processes to avoid a reproducible macOS
native-library conflict between eagerly loaded FAISS and PyTorch/OpenMP.

## Real BGE-M3 verification

```bash
FINRAG_RUN_LIVE_BGE=1 uv run pytest -m live tests/live/test_bge_m3.py
```

Model ID: `BAAI/bge-m3`  
Revision: `5617a9f61b028005a4858fdac845db406aefb181`

Default device: `cpu`. FAISS and PyTorch are loaded lazily after two observed
segmentation faults with eager native imports. Subsequent isolated and combined
CPU smoke tests passed; MPS is not validated.

The index build writes observed elapsed time and process RSS delta to
`results/raw/embedding_benchmark.json`. RSS delta is not claimed to be a
hardware-wide peak measurement.

The latest eight-chunk build sample took 4.415 seconds and observed a process
RSS increase of 2,253,848,576 bytes on the local Mac. This is a run observation,
not a portable latency or peak-memory guarantee.

For a standalone three-language local benchmark:

```bash
HF_HUB_OFFLINE=1 uv run python scripts/benchmark_bge.py
```

## Retrieval metrics

The exploratory pilot writes raw scores and hits but sets `recall_at_k` to
null. Recall@K must remain unset until reviewers map expected evidence to exact
chunk IDs.

The durable abridged report is in `docs/retrieval_pilot.md`; the complete raw
passages, scores, source IDs, URLs, and modes are in the ignored JSONL artifact.

The extension benchmark has 15 canonical questions and 45 variants. Phase 3.5
candidate passages are in `configs/benchmarks/retrieval_v2_candidates.json`.
The evaluator currently reports zero evaluated variants and null Recall@1/3/5
for both indexes, every language, and every topic because all evidence mappings
are unverified. See `docs/methodology.md` for the review gate.

## Phase 4 chatbot

Deterministic tests require no network or API credential:

```bash
uv run pytest
```

Run the real-index, deterministic-provider demonstration with:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  uv run python scripts/run_phase4_demo.py
```

This writes 15 raw API responses to the Git-ignored
`results/raw/phase4_demo.json`. The answer text is a mock fixture; retrieved
passages, scores, IDs, pages, and URLs come from the actual v2 index.

Hosted execution requires explicit account configuration. No default hosted
model is assumed:

```bash
FINRAG_LLM_PROVIDER=openai \
FINRAG_LLM_MODEL=<account-available-model> \
FINRAG_LLM_API_KEY=<secret> \
FINRAG_RUN_LIVE_LLM=1 \
  uv run pytest -m live tests/live/test_live_openai_provider.py
```

