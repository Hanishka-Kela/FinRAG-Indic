import os
from pathlib import Path

import numpy as np
import pytest

from finrag_indic.config import Settings
from finrag_indic.retrieval.embeddings import BgeM3Embedder


@pytest.mark.live
@pytest.mark.skipif(
    os.getenv("FINRAG_RUN_LIVE_BGE") != "1",
    reason="set FINRAG_RUN_LIVE_BGE=1 to load the pinned BGE-M3 weights",
)
def test_pinned_bge_m3_generates_normalized_multilingual_embeddings() -> None:
    settings = Settings(_env_file=None)
    provider = BgeM3Embedder(
        model_id=settings.embedding_model,
        revision=settings.embedding_revision,
        cache_folder=Path("data/models/sentence-transformers"),
        device=settings.embedding_device,
    )

    vectors = provider.embed(
        ["UPI payment safety", "यूपीआई भुगतान सुरक्षा", "UPI கட்டண பாதுகாப்பு"],
        batch_size=3,
    )

    assert vectors.shape == (3, 1024)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), np.ones(3), atol=1e-5)

