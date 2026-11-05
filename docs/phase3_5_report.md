# Phase 3.5 retrieval quality report

Run date: 2026-11-05. This report separates implemented engineering changes,
exploratory retrieval observations, and formal evaluation results.

## Hindi extraction

The official SEBI Hindi PDF uses subset Adobe Devanagari Type 1/CFF fonts with
custom encodings and no usable `ToUnicode` maps for several ligatures. Standard
PyMuPDF extraction therefore emitted gaps or replacement Latin characters.
The repair reads the embedded CFF glyph names (for example,
`uni0924094D0924` for `त्त`), aligns glyph IDs to the visible raw text layer,
and repairs visually encoded pre-base `ि` order. It does not translate English
content or replace the official PDF.

Actual samples:

| Page | Standard extraction | CFF-glyph extraction |
|---:|---|---|
| 1 | `िव ीय िश ा पुि का` | `वित्तीय शिक्षा पुस्तिका` |
| 5 | `िव संबंधी ¼ान` | `वित्त संबंधी ज्ञान` |
| 5 | corrupted `अ…ाय` / `बचत …ा है ?` forms | `अध्याय` / `बचत क्या है ?` |

The repaired run produced 221 blocks and 97 chunks, with one unresolved U+FFFD
glyph and no retained C0 controls. It contains 112,533 Devanagari code points,
compared with 91,228 from the standard extraction. This is a measurable
improvement, not proof of exact text recovery. Visible anomalies such as
duplicated vowel marks remain in some words, so Hindi evidence still requires
inspection against the PDF.

## RBI failed-transaction table extraction

PyMuPDF detected the circular's tables on pages 4 and 5. The v2 extractor uses
the detected description, timeline, and compensation columns and creates one
block per logical incident row. Payment-system names are repeated into each row
and page provenance is retained. It emitted 14 row blocks and 13 indexable
chunks; one short row fell below the configured 20-token minimum.

The following rows are now separate:

- UPI funds transfer: beneficiary not credited; T+1 auto-reversal rule; page 4.
- UPI merchant payment: confirmation not received; T+5 auto-reversal rule;
  page 4.
- AePS merchant confirmation: credit adjustment within T+5; page 5.
- AePS beneficiary not credited: its row is preserved without invented timeline
  or compensation values because those cells were blank in the extracted row.

This avoids the v1 failure in which UPI and adjacent AePS text could be merged.
It does not infer values from visually merged or blank cells.

## Versioned index and exploratory retrieval

`multilingual-genuine-v2` contains 269 normalized BGE-M3 vectors: 120 English,
97 Hindi, and 52 Tamil. The 120 English vectors consist of the immutable
107-chunk baseline and 13 RBI table chunks. The v1 index remains unchanged.

For the failed-UPI-transfer question, the specific UPI transfer row ranked first
in v2 for English (0.696407), Hindi (0.707450), and Tamil (0.671883). For the
merchant-payment question, the specific merchant row ranked first for English
(0.803476), Hindi (0.747880), and Tamil (0.714200). The unchanged English
baseline instead returned general UPI/security passages at rank 1. These are
ungraded exploratory observations, not Recall measurements or evidence that v2
is generally superior.

## Benchmark status

The workbook `outputs/phase3_5/retrieval_benchmark_review.xlsx` contains all 45
variants and the required reviewer fields. Every translation, answerability,
evidence-correctness, currency, and decision field starts as `Unverified`.
Candidate passages are not gold evidence.

There are zero human-approved evidence mappings. Accordingly, the evaluation
artifact reports zero evaluated variants and null Recall@1, Recall@3, and
Recall@5 for both indexes, including every language and topic breakdown.

## Limitations and Phase 4 recommendation

- Hindi text is substantially improved but not exact enough to waive source
  inspection.
- The Tamil extraction still contains 851 U+FFFD characters in the current
  corpus and was not repaired in this phase.
- The 2019 RBI circular remains flagged for currency/amendment review.
- The failed RBI Ombudsman FAQ acquisition remains excluded.
- Questions, translations, and evidence mappings are unverified; there are no
  formal effectiveness results.

The engineering foundation is suitable for human review, but the benchmark is
not ready to support Phase 4 quality claims. Complete at least the benchmark's
translation and evidence review, and resolve the regulatory-currency decisions
for affected questions, before using retrieval results to assess generated
answers.
