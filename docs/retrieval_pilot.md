# Exploratory retrieval report

Run on 2026-11-04 with `BAAI/bge-m3` revision
`5617a9f61b028005a4858fdac845db406aefb181`, normalized 1,024-dimensional
embeddings, and FAISS `IndexFlatIP`. Scores are inner products (cosine
similarities because vectors are normalized). These questions and passages have
not been mapped or graded by a human, so Recall@K is not reported.

| Query | View | Top passage (abridged) | Score | Source / page |
|---|---|---|---:|---|
| English: UPI PIN safety | English | “Unified Payments Interface or UPI is a platform…” | 0.633965 | `rbi-fame-english-c05eebf2950baa16`, p.24 |
| Hindi: failed UPI debit | English (direct cross-language) | “Unified Payments Interface or UPI is a platform…” | 0.612259 | `rbi-fame-english-c05eebf2950baa16`, p.24 |
| Tamil: do not share UPI PIN | English (direct cross-language) | “Unified Payments Interface or UPI is a platform…” | 0.675068 | `rbi-fame-english-c05eebf2950baa16`, p.24 |
| English: diversification | English | “You can minimize this risk… by diversifying your investments.” | 0.659686 | `sebi-financial-education-english-93b63cbce792ab91`, p.24 |
| Hindi: risk and return | English (direct cross-language) | “Any investment entails… risk… direct correlation between risk and return.” | 0.691482 | `rbi-fame-english-aaade536a58aa1bc`, p.11 |
| Tamil: Ponzi warning signs | English (direct cross-language) | “A Ponzi scheme is a fraudulent investment scheme…” | 0.612671 | `sebi-financial-education-english-184681bfd734d5cd`, p.55 |

For the UPI PIN questions, the more directly responsive RBI p.25 passage (“Never
share your PIN with anyone”) ranked second: 0.620801 for English and 0.651862 for
Tamil. For the Tamil Ponzi query, the p.56 warning-sign passage ranked second at
0.562563. These rank details illustrate why top-1 alone must not be treated as a
quality verdict.

The Hindi failed-transaction result is a visible failure case: the retrieved
passages describe UPI and security but do not answer how to resolve a failed
debit. The approved NPCI FAQ was inaccessible, and no replacement was used.

Source URLs:

- RBI: `https://www.rbi.org.in/commonman/images/FAME%20Booklet%20second%20edition/FAME%20Booklet%20English/FAME30072020.pdf`
- SEBI: `https://investor.sebi.gov.in/pdf/downloadable-documents/Financial%20Education%20Booklet%20-%20English.pdf`

The raw, unabridged records are written to
`results/raw/retrieval_pilot.jsonl` (Git-ignored). Hindi and Tamil were also run
against the “multilingual” view, but its source set is currently identical to the
English view, so its scores are identical and do not establish translation or
multilingual-corpus gains.

## Phase 3 extension comparison — 2026-11-05

The extension compares the unchanged 107-vector English baseline with the
separate 262-vector `multilingual-genuine-v1` index. The latter contains 112
English, 98 Hindi, and 52 Tamil chunks. All 45 question variants are exploratory
and ungraded.

Observed top-rank document languages in the multilingual index were 35 English,
10 Tamil, and 0 Hindi. This distribution is not an effectiveness score. It also
exposes the likely effect of Hindi extraction corruption.

Selected actual results:

| Query | English baseline top hit | Genuine multilingual top hit |
|---|---|---|
| English UPI merchant failure | RBI FAME p.25, score 0.571627 | RBI TAT circular p.5, score 0.670841; retrieved table continuation starts with AePS rather than the UPI row, so relevance needs review |
| Hindi failed UPI transfer | RBI FAME p.25, score 0.604020 | Same RBI FAME chunk and score; the TAT circular was only rank 3, so the specific failure evidence did not reach top 1 |
| Tamil inflation | English SEBI p.7, score 0.634369 | Tamil SEBI `sebi-financial-education-tamil-e2a2219eb5294c62`, score 0.734598; passage explains price rises and declining purchasing power |
| Hindi risk/return | RBI FAME p.11, score 0.690219 | Same English RBI passage and score; no Hindi chunk reached top 1 |
| Tamil Ponzi warning | English SEBI p.55, score 0.618615 | Tamil SEBI `sebi-financial-education-tamil-8434c27c87999b0d`, score 0.668060; passage describes Ponzi structure, with visible extraction artifacts |

The circular improves coverage of failed-payment rules, but table extraction
mixes adjacent payment-system rows. The failed-transfer query remains a failure
at top 1 in every language. Tamil same-language retrieval shows several plausible
examples; Hindi same-language retrieval is limited by text quality. None of these
observations is a relevance judgment or a multilingual effectiveness claim.

The complete review artifact is
`configs/benchmarks/retrieval_v1_candidates.json`. It contains top-five passages,
scores, chunk IDs, document IDs, language, page, URL, and flags for both indexes.
All 45 records remain `candidate_unverified`; Recall@1/3/5 are therefore null.

## Phase 3.5 table-boundary check — 2026-11-05

The v2 index contains row-bounded RBI TAT chunks. Actual rank-1 results were:

| Question | Language | v2 rank-1 score | Rank-1 chunk |
|---|---|---:|---|
| Failed UPI transfer | English | 0.696407 | `rbi-failed-transactions-tat-2019-a0b3dab3dd67679f` |
| Failed UPI transfer | Hindi | 0.707450 | `rbi-failed-transactions-tat-2019-a0b3dab3dd67679f` |
| Failed UPI transfer | Tamil | 0.671883 | `rbi-failed-transactions-tat-2019-a0b3dab3dd67679f` |
| UPI merchant payment failure | English | 0.803476 | `rbi-failed-transactions-tat-2019-bd5b8e3bcff5b28d` |
| UPI merchant payment failure | Hindi | 0.747880 | `rbi-failed-transactions-tat-2019-bd5b8e3bcff5b28d` |
| UPI merchant payment failure | Tamil | 0.714200 | `rbi-failed-transactions-tat-2019-bd5b8e3bcff5b28d` |

The first chunk is the page-4 beneficiary-not-credited transfer row (T+1); the
second is the distinct page-4 merchant-confirmation row (T+5). AePS content is
not present in either chunk. These are exploratory retrieval outputs. They are
not human relevance labels and do not establish v2 superiority.

Complete v2 candidates are in
`configs/benchmarks/retrieval_v2_candidates.json`. Formal evaluation remains
empty: 0/45 variants have human-approved evidence, so Recall@1/3/5 are null.
