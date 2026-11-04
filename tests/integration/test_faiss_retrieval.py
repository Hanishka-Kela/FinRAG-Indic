from pathlib import Path

import numpy as np
import pytest

from finrag_indic.errors import EmptyCorpusError, IndexCompatibilityError
from finrag_indic.knowledge.models import Chunk, Language
from finrag_indic.retrieval.embeddings import DeterministicHashEmbedder
from finrag_indic.retrieval.index import FaissRetriever


def chunk(index: int, text: str) -> Chunk:
    import hashlib

    digest = hashlib.sha256(text.encode()).hexdigest()
    return Chunk(
        chunk_id=f"chunk-{index}",
        document_id="npci-test",
        language=Language.ENGLISH,
        text=text,
        chunk_index=index,
        token_count=len(text.split()),
        content_sha256=digest,
        source_url="https://www.npci.org.in/test",
        title="Test",
        issuing_organization="NPCI",
    )


def test_index_save_load_and_top_k(tmp_path: Path) -> None:
    provider = DeterministicHashEmbedder(dimension=16)
    chunks = [chunk(0, "UPI safety"), chunk(1, "loan interest")]
    vectors = provider.embed([item.text for item in chunks], batch_size=2)
    retriever = FaissRetriever.build(
        chunks=chunks,
        vectors=vectors,
        embedding=provider.metadata,
        corpus_view="english",
    )

    hits = retriever.search(vectors[0], top_k=2)
    retriever.save(tmp_path)
    restored = FaissRetriever.load(tmp_path, expected_embedding=provider.metadata)
    restored_hits = restored.search(vectors[0], top_k=1)

    assert hits[0].chunk.chunk_id == "chunk-0"
    assert hits[0].score == pytest.approx(1.0)
    assert restored_hits[0].chunk == chunks[0]


def test_empty_corpus_is_rejected() -> None:
    provider = DeterministicHashEmbedder(dimension=8)
    with pytest.raises(EmptyCorpusError):
        FaissRetriever.build(
            chunks=[],
            vectors=np.empty((0, 8), dtype=np.float32),
            embedding=provider.metadata,
            corpus_view="empty",
        )


def test_duplicate_content_is_rejected() -> None:
    provider = DeterministicHashEmbedder(dimension=8)
    chunks = [chunk(0, "same"), chunk(1, "same")]
    vectors = provider.embed([item.text for item in chunks], batch_size=2)
    with pytest.raises(IndexCompatibilityError, match="duplicate chunk content"):
        FaissRetriever.build(
            chunks=chunks,
            vectors=vectors,
            embedding=provider.metadata,
            corpus_view="english",
        )

