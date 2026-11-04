# Source acquisition report

Status as of 2026-11-04. Acquisition was restricted to the three explicitly
approved URLs. Redirect targets were checked against the official-domain
allowlist; no linked pages were crawled and no source was substituted.

| Document ID | Result | Verified response/content | SHA-256 | Corpus state |
|---|---|---|---|---|
| `rbi-fame-english` | HTTP 200 | `application/pdf`; `%PDF-` signature; 40 parseable pages | `a4054de4d1f0517126ceaeaf6aab35168957eaff087b3905e4acebca49c48538` | Original English; extracted, 33 chunks |
| `npci-upi-faq-english` | HTTP 403 | Download rejected before a valid document was received | — | Not acquired; excluded |
| `sebi-financial-education-english` | HTTP 200 | `application/pdf`; `%PDF-` signature; 73 parseable pages | `c9ae1ad2388acc8269ca643bd365467f2c3e4ad10117a16c5e5ae55bf0a15210` | Original English; extracted, 74 chunks |

The resolved RBI and SEBI URLs and domains matched the approved URLs. Original
files are preserved under `data/raw/sources/` and intentionally Git-ignored.
The NPCI 403 is retained in `configs/sources.json`; it was not bypassed with a
browser, alternate URL, cached copy, or unofficial mirror.

No genuinely available approved Hindi or Tamil regulator document was present.
Consequently the multilingual corpus view contains the same 107 original
English chunks as the English view. It contains no machine-translated material.

## Currency-review flags

Automated flags identify passages requiring a human currency check; they do not
declare that the content is obsolete. The indexed metadata flags:

- RBI pages 32 and 36–38: potentially superseded ombudsman process.
- SEBI page 28: date-sensitive settlement-cycle details.
- SEBI pages 18, 48, 50, 53–54: date-sensitive tax information.
- SEBI pages 3, 22, 48–49: date-sensitive government/savings scheme details.
- SEBI pages 57–58: potentially superseded regulatory amendment reference.
- SEBI pages 60–62 and 65: potentially superseded SCORES/grievance process.

The flags are deliberately conservative and may include contextual or table-of-
contents mentions. Publication/effective dates and reuse terms were not inferred
and remain unverified where the documents do not provide reliable metadata.
