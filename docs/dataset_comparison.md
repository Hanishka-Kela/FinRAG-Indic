# Dataset comparison and current status

The Phase 1 comparison remains the selection basis. Phase 3 inspected the
actual FABRIC artifact rather than relying solely on its card.

## FABRIC inspection

- Dataset: `agenticclass/fabric`
- Pinned revision: `d8476ea6a9081f7dec4852200b379bfd1213ad47`
- Artifact: `data/train-00000-of-00001.parquet`
- Local artifact SHA-256:
  `bcc8341d559d6c73f023104a562dfcaf74da9ad14ddade8571c93cbfbd28f8ee`
- Rows observed: 204
- IDs inspected for schema/provenance: `adv_001`, `adv_002`
- Required fields observed: ID, category, subcategory, difficulty, English,
  Hindi and Tamil questions, English answer, and source description.
- Ten records (`ent_001` through `ent_010`) have an empty upstream `source`
  description. The empty values are preserved rather than filled or inferred.

The two inspected records were inspected only for format. Their answers,
translations, regulatory currency, suitability, and source descriptions remain
`unreviewed`. No supporting passage or gold retrieval mapping was created.

The review schema requires separate decisions for source verification, answer
correctness, Hindi translation, Tamil translation, regulatory currency, and
question suitability. A record cannot receive reviewer approval until all six
fields are explicitly verified and reviewer identity/time are recorded.

No final evaluation split has been produced.

