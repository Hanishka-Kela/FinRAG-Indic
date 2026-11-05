# Source acquisition report

Status as of 2026-11-05 (Asia/Kolkata). Acquisition is exact-URL and approval
gated. Redirect hosts, HTTP status, content type, signatures, and actual content
are checked; linked pages are not crawled and failures are not substituted.

## English baseline (preserved)

| Document ID | Result | SHA-256 | Pages | Chunks |
|---|---|---|---:|---:|
| `rbi-fame-english` | HTTP 200, genuine PDF | `a4054de4d1f0517126ceaeaf6aab35168957eaff087b3905e4acebca49c48538` | 40 | 33 |
| `sebi-financial-education-english` | HTTP 200, genuine PDF | `c9ae1ad2388acc8269ca643bd365467f2c3e4ad10117a16c5e5ae55bf0a15210` | 73 | 74 |
| `npci-upi-faq-english` | HTTP 403 | — | — | 0 |

The 107-chunk English index is locked by
`configs/baselines/english_v1.json`. Its FAISS, chunk-map, and metadata hashes
were unchanged after the extension build.

## Approved extension

| Document ID | Result | SHA-256 | Pages | Chunks |
|---|---|---|---:|---:|
| `sebi-financial-education-hindi` | HTTP 200, `application/pdf`, genuine PDF | `1c41627ebcd7aabf6e4b90b92a25832547a9e10ce68453a0aa6057205dfd95c0` | 65 | 98 |
| `sebi-financial-education-tamil` | HTTP 200, `application/pdf`, genuine PDF | `5fefd552df51c3b1a501a0edc4e12e2237c21655199b4e814536cc71e4417feb` | 56 | 52 |
| `rbi-failed-transactions-tat-2019` | HTTP 200, `application/pdf`, genuine PDF | `666fe280ed11486cea4e36f905318e75322ed9343a6b188008d0936f800f88f6` | 5 | 5 |
| `rbi-integrated-ombudsman-faq-2026` | HTTP 200, `text/html`, anti-bot challenge—not FAQ content | `4be854f565ab0e416cce1d0a7b34414e94a32586382c270bfbb494fd15a36a70` | — | 0 |

The rejected FAQ response is preserved as failure evidence but has
`extraction_status: failed` and is excluded from every index. No CAPTCHA was
solved or bypassed. The other three documents are verified source documents.

## Extraction and comparability review

- Hindi: v1's standard extraction produced 98 chunks with dropped ligatures and
  matras. Phase 3.5 decodes the embedded CFF glyph names into 97 versioned v2
  chunks. The title improves from `िव ीय िश ा पुि का` to
  `वित्तीय शिक्षा पुस्तिका`. One unresolved replacement glyph and word-level
  anomalies remain, so the repair is not treated as verified transcription.
- Tamil: text is substantially readable, including the inflation and Ponzi
  sections, but contains `�`, duplicated words, and spacing artifacts.
- RBI TAT: v1 produced 5 page-level chunks that could merge adjacent payment
  systems. Phase 3.5 produces 14 logical incident blocks and 13 indexable
  row-bounded chunks, preserving UPI transfer, UPI merchant, and AePS rows
  separately. Blank extracted cells remain blank; no values are inferred.
- The Hindi and Tamil tables of contents both enumerate the same broad 15-topic
  structure as the English SEBI booklet: personal finance, planning, savings,
  securities, insurance, retirement, borrowing, schemes, tax, Ponzi/unregistered
  advisers, grievance mechanisms, and SEBI information.
- This supports concept-level comparability only. Page-, paragraph-, and
  translation-level equivalence has not been established.

## Regulatory currency

The 2019 RBI circular is flagged
`currency_verification_inconclusive_check_amendments` on all five chunks. Later
RBI materials continue to cite it “as amended/updated from time to time,” but
the review did not locate a single authoritative consolidated amendment history.
It is therefore not asserted to be either unchanged or superseded.

Authoritative cross-checks used were RBI's Master Direction on PPIs, which cites
the circular “as amended from time to time,” and the 2024 BBPS Directions, which
cites it “as updated from time to time”:

- `https://www.rbi.org.in/Scripts/BS_ViewMasDirections.aspx?id=12156`
- `https://www.rbi.org.in/Scripts/BS_ViewMasDirections.aspx?id=12616`

The acquired FAQ response could not verify the 2026 scheme or effective date.
Search discovery metadata is not used as indexed evidence.
