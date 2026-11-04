# Architecture

## Implemented through Phase 3

```text
approved source manifest
  -> official-domain downloader
  -> immutable raw file + SHA-256
  -> PDF/HTML extraction
  -> Unicode normalization
  -> page/section-aware chunks
  -> replaceable dense embedding provider
  -> normalized float32 vectors
  -> FAISS IndexFlatIP + checked chunk mapping
  -> top-k hits with scores and original source metadata
```

The manifest is the control boundary. Candidate or rejected URLs cannot be
downloaded. Acquired files retain a retrieval timestamp, checksum, local path,
and extraction state. The ingestion pipeline never substitutes an unofficial
source after a failure.

`EmbeddingProvider` is replaceable. The research candidate is BGE-M3 pinned to
revision `5617a9f61b028005a4858fdac845db406aefb181`; routine tests use a
deterministic normalized mock. Mock embeddings are never valid research output.

Each persisted index contains:

- `vectors.faiss`: exact `IndexFlatIP` vectors;
- `chunks.jsonl`: ordered vector-to-chunk mapping;
- `metadata.json`: schema version, corpus view, embedding revision, dimensions,
  counts, and mapping checksum.

Loading rejects missing files, embedding mismatches, dimensional mismatches,
count mismatches, and modified chunk mappings.

## Corpus views

- `english`: English chunks only.
- `multilingual`: all acquired language chunks.

The `coverage_group` field links official translations of the same material.
`controlled_coverage_views` includes only groups containing English, Hindi, and
Tamil, preventing a controlled comparison from silently using unequal facts.

## Not implemented

LLM answer generation, generated citations, chatbot behavior, query
translation, and full experiments remain outside Phase 3. Experiment D is
represented in the retrieval pilot as direct Hindi/Tamil embedding retrieval
against the English index, without query translation.

