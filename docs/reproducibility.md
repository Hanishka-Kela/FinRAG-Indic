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

The manifest contains only the three explicitly approved URLs. Acquisition
continues after a per-source failure and records the error without substitution:

```bash
uv run python scripts/acquire_sources.py
uv run python scripts/build_indexes.py
uv run python scripts/run_retrieval_pilot.py
```

Raw sources, embeddings, FAISS indexes, and raw outputs are Git-ignored.
Manifests, code, schemas, and pinned revisions are versioned.

The observed build produced 107 vectors in each index (33 RBI and 74 SEBI).
`data/indexes/*/metadata.json` pins the embedding metadata and chunk-list hash.
The multilingual view is English-only until genuine approved Hindi/Tamil source
versions are acquired.

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

