# Phase 5 experimental evaluation

This document describes the reproducible exploratory evaluation. It is not a
human-validated accuracy study: all benchmark variants remain marked
`candidate_unverified`, and reviewer score columns are intentionally blank until
someone evaluates the answers.

## Conditions

- **LLM-only:** the selected Gemini model receives the question without
  retrieved evidence or citations.
- **RAG English:** BGE-M3 retrieves from the immutable `english-v1` index.
- **RAG multilingual:** BGE-M3 retrieves from `multilingual-genuine-v2`.

All conditions use the configured provider, model, temperature, output limit, and
bounded retry policy. Retrieval scores are observations, not correctness
measures. Citation-reference validity only checks that markers resolve to
retrieved passages and metadata; it does not establish factual entailment.

## Benchmark

`configs/benchmarks/retrieval_v1.json` contains 15 canonical questions with
English, Hindi, and Tamil variants. Existing translations and evidence mappings
are exploratory and unverified. The runner reports an estimated request count
before execution: `canonical questions × 3 languages × 3 conditions`.
The separate `configs/benchmarks/phase5_unsupported_probes.json` file contains
two exploratory unsupported probes in three languages. These are additional
probes, not ground truth and not part of the 15-question benchmark count.

## Reproduction

```bash
uv run python scripts/run_phase5_experiment.py --dry-run
uv run python scripts/run_phase5_experiment.py --limit 3 --output results/raw/phase5_pilot.jsonl
uv run python scripts/run_phase5_experiment.py --unsupported-benchmark configs/benchmarks/phase5_unsupported_probes.json --resume
uv run python scripts/run_phase5_experiment.py --resume
uv run python scripts/summarize_phase5.py
```

The runner appends each completed case immediately and resumes by the
`canonical_id`, language, and condition key. The reviewer CSV contains blank
1–5 fields for factual correctness, relevance, grounding, language consistency,
clarity, and appropriate abstention. No human scores are generated
automatically.
