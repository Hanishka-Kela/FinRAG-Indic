# Phase 4 multilingual RAG chatbot

## Implemented request path

`POST /chat` validates a manually selected `en`, `hi`, or `ta` language, rejects
likely submitted credentials, embeds the question with the pinned BGE-M3
revision, searches `multilingual-genuine-v2`, constructs a source-labelled
grounding prompt, and calls the configured provider asynchronously. Every
non-abstaining generated answer must contain only citation markers that resolve
to retrieved evidence. Missing or invalid markers cause a conservative
abstention response.

No similarity threshold is treated as a correctness boundary. Retrieval scores
are returned for inspection but do not prove support.

## Provider modes

- `mock` is deterministic and requires no credential. Its answer is a test/demo
  fixture, not a language-model quality result.
- `openai` uses the official asynchronous SDK and Responses API. Model, API key,
  optional base URL, generation controls, timeout, and retry count come from
  environment settings. No hosted model is selected automatically.

The SDK automatically retries its documented transient classes within the
configured retry budget. Application responses convert provider and timeout
failures to structured errors without logging the prompt, credential, or user
financial data.

## API responses

`POST /chat` returns the answer, selected language, retrieved passages,
resolvable source citations, abstention flag, provider/model identifiers,
retrieval/generation/total latency, and an optional structured error.

`GET /health` is a liveness endpoint. `GET /system/status` reports provider
configuration and local index availability without exposing credentials.

## Demonstration run

The 2026-11-05 deterministic run executed 15 requests covering UPI PIN safety,
failed UPI transfers, diversification, inflation, and Ponzi schemes in English,
Hindi, and Tamil. All used the real 269-vector v2 index and returned HTTP 200.
The outputs are mock-provider fixtures; only retrieval passages and citations
are empirical chatbot artifacts.

Selected rank-1 citations:

| Topic / language | Score | Retrieved source |
|---|---:|---|
| Failed UPI transfer / English | 0.696407 | RBI TAT circular, page 4, UPI transfer row |
| Failed UPI transfer / Hindi | 0.707450 | RBI TAT circular, page 4, UPI transfer row |
| Failed UPI transfer / Tamil | 0.671883 | RBI TAT circular, page 4, UPI transfer row |
| Diversification / English | 0.664906 | SEBI English booklet, page 24 |
| Inflation / Hindi | 0.695712 | Repaired SEBI Hindi booklet, page 6 |
| Inflation / Tamil | 0.734598 | SEBI Tamil booklet, page 7 |
| Ponzi schemes / English | 0.733083 | SEBI English booklet, page 55 |

The Tamil and Hindi sources retain the extraction limitations documented in
Phase 3.5. The 2019 RBI circular retains its regulatory-currency review flag.

## Remaining limitations

- No hosted API credential/model was configured for this run, so there are no
  actual hosted-model answers, latency measurements, or costs to report.
- The deterministic mock does not measure answer quality.
- Human evidence mappings remain unverified, so retrieval Recall remains null.
- Citation-marker validation confirms provenance linkage, not claim-level
  entailment.
- Source extraction and regulatory-currency flags remain visible and require
  human review for high-stakes use.
