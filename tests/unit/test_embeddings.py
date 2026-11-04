from pathlib import Path

import numpy as np
import pytest

from finrag_indic.errors import EmbeddingError
from finrag_indic.retrieval.embeddings import (
    BgeM3Embedder,
    DeterministicHashEmbedder,
    embed_with_cache,
)


def test_mock_embeddings_are_normalized_and_deterministic() -> None:
    provider = DeterministicHashEmbedder(dimension=8)

    first = provider.embed(["English", "தமிழ்"], batch_size=2)
    second = provider.embed(["English", "தமிழ்"], batch_size=1)

    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), np.ones(2), atol=1e-6)


def test_embedding_cache_round_trip(tmp_path: Path) -> None:
    provider = DeterministicHashEmbedder(dimension=8)
    cache = tmp_path / "vectors.npz"

    first = embed_with_cache(provider, ["one", "two"], batch_size=2, cache_path=cache)
    second = embed_with_cache(provider, ["one", "two"], batch_size=2, cache_path=cache)

    np.testing.assert_array_equal(first, second)


def test_model_loading_failure_is_wrapped(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    provider = BgeM3Embedder(model_id="invalid", revision="invalid")

    def fail():  # type: ignore[no-untyped-def]
        raise RuntimeError("load failed")

    monkeypatch.setattr(provider, "_load", fail)
    with pytest.raises(EmbeddingError, match="inference failed"):
        provider.embed(["text"], batch_size=1)

