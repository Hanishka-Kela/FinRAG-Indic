# Retrieval methodology and evidence states

## Corpus boundaries

`english-v1-107` is the immutable initial baseline: 107 English chunks and its
original FAISS index. `multilingual-genuine-v1` is a separate index containing
the baseline plus acquired regulator documents in Hindi, Tamil, and English.
The earlier English-only index directory named `multilingual` is retained only
as a historical engineering artifact and is not called a multilingual document
corpus.

Every chunk retains document ID, language, official URL, organization, page,
content checksum, extraction metadata, coverage group, and review flags.

## Three distinct result levels

1. **Engineering validation:** download validation, extraction, normalized
   embedding generation, index persistence, and deterministic tests. This shows
   the pipeline operates; it says nothing about retrieval quality.
2. **Exploratory retrieval:** ungraded top-k passages and similarity scores.
   These can expose successes and failures but are not effectiveness metrics.
3. **Formal retrieval evaluation:** Recall@1, Recall@3, and Recall@5 over only
   question variants whose evidence mappings are explicitly marked
   `human_verified`. No current mapping has that status.

## Benchmark review workflow

`configs/benchmarks/retrieval_v1.json` contains 15 canonical topics and 45
language variants. English is the canonical draft; Hindi and Tamil variants are
labelled `draft_unverified`, not verified translations.

`scripts/prepare_retrieval_benchmark.py` records top-five candidates from both
indexes, including chunk ID, passage excerpt, score, source, page, language, URL,
and flags. Candidates are review aids and never become gold automatically.

A reviewer must inspect the underlying source passage, correct the question
translation if necessary, change `evidence_status` to `human_verified`, and add
one or more exact chunk IDs. Schema validation rejects gold IDs on any other
status. `scripts/evaluate_retrieval.py` then re-runs retrieval and excludes every
unverified variant from the denominator.

## Known validity threats

- Hindi font decoding materially degrades lexical retrieval signals.
- Tamil extraction has lower-severity Unicode and layout artifacts.
- The SEBI editions cover similar chapter-level concepts but are not proven
  sentence-aligned translations.
- The 2019 TAT circular requires ongoing currency review.
- Similarity scores are not calibrated probabilities or relevance judgments.
- The current candidate mappings and drafted question variants await human review.
