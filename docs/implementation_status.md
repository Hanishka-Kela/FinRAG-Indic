# Implementation status

## Implemented and run in Phase 3

- Exact-URL, approval-gated acquisition with official-domain, redirect, HTTP
  status, content-type, PDF-signature, parseability, and visible-HTML checks.
- Original-file preservation, SHA-256 checksums, retrieval timestamps, resolved
  URLs, and explicit acquisition failures.
- PDF/HTML extraction, Unicode normalization, page/section metadata,
  configurable overlapping chunks, duplicate removal, and currency-review flags.
- FABRIC revision/schema/checksum validation and an unreviewed 204-record human-
  review template. FABRIC answers, citations, and translations are not verified.
- Replaceable embedding interface and pinned BGE-M3 model, CPU benchmark, cache,
  and deterministic mock.
- Two persisted FAISS `IndexFlatIP` views, each containing 107 normalized,
  1,024-dimensional vectors (33 RBI and 74 SEBI chunks).
- Exploratory English, Hindi, and Tamil queries, including direct Hindi/Tamil
  multilingual embeddings against the English index (Experiment D isolation).
- Raw JSONL retrieval logging with passages, scores, IDs, URLs, model metadata,
  and deliberately null Recall@K.

## Experimentally observed, with limited scope

- The pinned BGE-M3 revision loaded locally on CPU and embedded English, Hindi,
  and Tamil strings.
- The six-question pilot returned semantically relevant top results for several
  topics. The Hindi failed-UPI query did not retrieve failure-resolution evidence
  because the inaccessible NPCI FAQ left that corpus gap.
- Hindi/Tamil-to-English results are evidence that the path executes and can
  retrieve relevant examples, not evidence of general multilingual effectiveness.
- English and multilingual views returned identical results because both contain
  the exact same English chunks; this is not a comparative multilingual result.

## Pending or explicitly out of scope

- NPCI FAQ acquisition (approved endpoint returned HTTP 403).
- Genuine Hindi/Tamil regulator content and translated-corpus experiments.
- Formal Recall@K or comparative effectiveness claims: no human-verified gold
  chunk mappings exist.
- Human review of all 204 FABRIC records and all currency-review flags.
- Hosted LLM selection, answer generation, and Phase 4 experiments A–C.

No LLM generation was implemented in Phase 3.

## Phase 3 extension

Implemented:

- Immutable checksum lock for the 107-chunk English baseline.
- Separate 262-vector genuine multilingual index with original regulator
  documents in English, Hindi, and Tamil.
- Anti-bot response detection and preservation as failed acquisition evidence.
- Fifteen canonical questions with 45 explicitly unverified language variants.
- Candidate evidence review artifact covering both indexes.
- Recall@1/3/5 evaluator gated to human-verified mappings only.

Experimentally observed but not formally evaluated:

- Tamil documents reached rank 1 for 10/45 exploratory variants; Hindi documents
  reached rank 1 for 0/45. Extraction quality is a major confounder.
- No evidence mapping is verified, so formal recall values remain null.
