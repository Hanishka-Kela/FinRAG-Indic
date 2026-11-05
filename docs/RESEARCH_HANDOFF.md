# FinRAG-Indic research handoff

## Scope and status

FinRAG-Indic is an evidence-grounded financial-literacy chatbot for English,
Hindi, and Tamil. Phase 6 prepares the implementation for a college
demonstration and research-paper handoff. No Phase 6 deployment or new live
experiment was performed.

## Architecture

The production path is documented in [architecture.md](./architecture.md):
Streamlit calls FastAPI; FastAPI normalizes retrieval queries, embeds with
pinned BGE-M3, searches FAISS, sends retrieved evidence to Gemini, and returns
a selected-language answer with source metadata and abstention status.
Personalized loan or investment decisions remain abstained: the service can
provide grounded general education, identify missing information, and ask one
localized clarification question. API responses separately report retrieval,
initial generation, citation-correction, and total latency.

## Corpus and provenance

The source inventory is in [source_inventory.md](./source_inventory.md).
Acquired source documents are official RBI and SEBI materials with recorded
URLs, retrieval metadata, checksums, and extraction status. The English baseline
contains 107 chunks. The genuine multilingual v2 index contains 269 chunks
across English, Hindi, and Tamil. The baseline and v2 index artifacts were not
modified during Phase 6.

Hindi and Tamil extraction repairs remain review-limited. The RBI failed-
transaction circular carries a regulatory-currency review flag. The NPCI
endpoint and an RBI FAQ acquisition were not substituted or bypassed.

## Retrieval and generation configuration

- Embeddings: `BAAI/bge-m3`, pinned revision
  `5617a9f61b028005a4858fdac845db406aefb181`.
- Index: `multilingual-genuine-v2` for production.
- Baseline comparison: immutable `english-v1`.
- Provider: Gemini through the official OpenAI SDK's asynchronous
  Chat Completions-compatible endpoint.
- Model used in live validation: `gemini-flash-lite-latest`.
- Generation parameters, timeouts, and retries are environment-configured.
- API credentials are local-only and are not included in this handoff.

## Benchmark and methodology

`configs/benchmarks/retrieval_v1.json` contains 15 canonical questions with
English, Hindi, and Tamil variants (45 variants total). Existing translations
and evidence mappings are explicitly `candidate_unverified`. They are not
ground truth and must not be reported as human-validated accuracy.

Phase 5 compared:

1. LLM-only, with no retrieved evidence.
2. RAG over the immutable English baseline.
3. RAG over `multilingual-genuine-v2`.

The pilot ran 27 actual Gemini requests: 3 canonical questions × 3 languages ×
3 conditions. No complete 135-request benchmark was run.

## Pilot results

The corrected table is recorded in [phase5_results.md](./phase5_results.md).
The preserved historical outputs are:

- [Raw pilot JSONL](../results/raw/phase5_pilot.jsonl)
- [Aggregate summary JSON](../results/aggregate/phase5_summary.json)
- [Aggregate summary CSV](../results/aggregate/phase5_summary.csv)
- [Reviewer template](../results/aggregate/phase5_reviewer_scores.csv)
- [Performance figure](../results/figures/phase5_response_performance.png)

| Condition | Successful / failed | Generated | Mean successful latency | Citation-valid / applicable |
|---|---:|---:|---:|---:|
| LLM-only | 9 / 0 | 9 | 6609 ms | Not applicable |
| English RAG | 9 / 0 | 9 | 3312 ms | 3 / 3 |
| Multilingual RAG | 8 / 1 | 8 | 1564 ms | 8 / 8 |

The multilingual failure was a retry-exhausted transient provider 503.
English RAG had six successful abstentions; those records are excluded from
the citation-valid denominator because no citation is expected from an
abstention. Citation validity checks marker-to-retrieved-passage resolution,
not factual entailment. Language consistency is a script-dominance heuristic,
not human translation assessment.

No human reviewer scores have been entered. The preserved reviewer template is
blank by design.

## Reproducibility

```bash
uv sync --dev --locked
cp .env.example .env
uv run pytest -m "not live"
uv run python scripts/run_phase4_demo.py
uv run python scripts/run_phase5_experiment.py --dry-run
uv run python scripts/run_phase5_experiment.py --limit 3 \
  --output results/raw/phase5_pilot.jsonl
uv run python scripts/summarize_phase5.py \
  --input results/raw/phase5_pilot.jsonl
```

The Phase 5 dry run makes no provider or embedding calls. A real run requires
explicit hosted-provider configuration and may incur cost; do not rerun it
merely to reproduce the historical table. The runner flushes records
incrementally and supports `--resume`. Reviewers should treat output quality as
exploratory until human scoring and claim-level evidence review are completed.

## Submission limitations

- No human accuracy, factuality, or translation score is reported.
- Similarity scores are not correctness measures.
- Citation linkage is not semantic proof.
- The documented pilot is small and incomplete; it does not support a
  statistically significant improvement claim.
- Hosted-model behavior depends on account quota and model availability.
- Source extraction and regulatory currency require human review for
  high-stakes use.

## Relevant artifacts

- [Phase 3.5 workbook](../outputs/phase3_5/retrieval_benchmark_review.xlsx)
- [Phase 3.5 report](./phase3_5_report.md)
- [Phase 5 results](./phase5_results.md)
- [Retrieval pilot report](./retrieval_pilot.md)
