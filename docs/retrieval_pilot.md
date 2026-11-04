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
