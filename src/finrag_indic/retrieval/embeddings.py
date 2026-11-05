import hashlib
import json
import time
from pathlib import Path
from typing import Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from finrag_indic.errors import EmbeddingError


class EmbeddingMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    model_id: str
    revision: str
    dimension: int = Field(gt=0)
    normalized: bool = True


class EmbeddingRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata: EmbeddingMetadata
    text_count: int = Field(ge=0)
    batch_size: int = Field(gt=0)
    elapsed_seconds: float = Field(ge=0)
    rss_delta_bytes: int | None = Field(default=None, ge=0)


class EmbeddingProvider(Protocol):
    @property
    def metadata(self) -> EmbeddingMetadata: ...

    def embed(self, texts: list[str], *, batch_size: int) -> np.ndarray: ...


def _normalize(vectors: np.ndarray) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2:
        raise EmbeddingError("embedding output must be a two-dimensional array")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise EmbeddingError("embedding output contains a zero vector")
    return array / norms


class DeterministicHashEmbedder:
    """Deterministic normalized embeddings for tests, never for research results."""

    def __init__(self, dimension: int = 32) -> None:
        self._metadata = EmbeddingMetadata(
            provider="mock",
            model_id="deterministic-hash",
            revision="v1",
            dimension=dimension,
        )

    @property
    def metadata(self) -> EmbeddingMetadata:
        return self._metadata

    def embed(self, texts: list[str], *, batch_size: int = 32) -> np.ndarray:
        del batch_size
        rows: list[np.ndarray] = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
            generator = np.random.default_rng(seed)
            rows.append(generator.standard_normal(self.metadata.dimension).astype(np.float32))
        if not rows:
            return np.empty((0, self.metadata.dimension), dtype=np.float32)
        return _normalize(np.vstack(rows))


class BgeM3Embedder:
    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        cache_folder: Path | None = None,
        device: str = "cpu",
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.cache_folder = cache_folder
        self.device = device
        self._model = None
        self._metadata = EmbeddingMetadata(
            provider="sentence-transformers",
            model_id=model_id,
            revision=revision,
            dimension=1024,
        )

    @property
    def metadata(self) -> EmbeddingMetadata:
        return self._metadata

    def _load(self):  # type: ignore[no-untyped-def]
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(
                    self.model_id,
                    revision=self.revision,
                    cache_folder=str(self.cache_folder) if self.cache_folder else None,
                    device=self.device,
                    trust_remote_code=False,
                )
                dimension = self._model.get_embedding_dimension()
                if dimension is None:
                    raise EmbeddingError("model did not report an embedding dimension")
                self._metadata = self._metadata.model_copy(update={"dimension": dimension})
            except EmbeddingError:
                raise
            except Exception as exc:
                raise EmbeddingError(f"failed to load {self.model_id}@{self.revision}: {exc}") from exc
        return self._model

    def embed(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        if not texts:
            return np.empty((0, self.metadata.dimension), dtype=np.float32)
        try:
            model = self._load()
            vectors = np.empty((len(texts), self.metadata.dimension), dtype=np.float32)
            for start in range(0, len(texts), batch_size):
                stop = min(start + batch_size, len(texts))
                vectors[start:stop] = np.asarray(
                    model.encode(
                        texts[start:stop],
                        batch_size=batch_size,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    ),
                    dtype=np.float32,
                )
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"embedding inference failed: {exc}") from exc
        normalized = _normalize(vectors)
        if normalized.shape[1] != self.metadata.dimension:
            raise EmbeddingError("embedding dimension differs from model metadata")
        return normalized


def benchmark_embeddings(
    provider: EmbeddingProvider, texts: list[str], *, batch_size: int
) -> tuple[np.ndarray, EmbeddingRun]:
    try:
        import psutil

        process = psutil.Process()
        before = process.memory_info().rss
    except (ImportError, OSError):
        process = None
        before = 0
    started = time.perf_counter()
    vectors = provider.embed(texts, batch_size=batch_size)
    elapsed = time.perf_counter() - started
    after = process.memory_info().rss if process is not None else 0
    peak_delta = max(0, after - before) if process is not None else None
    return vectors, EmbeddingRun(
        metadata=provider.metadata,
        text_count=len(texts),
        batch_size=batch_size,
        elapsed_seconds=elapsed,
        rss_delta_bytes=peak_delta,
    )


def embed_with_cache(
    provider: EmbeddingProvider,
    texts: list[str],
    *,
    batch_size: int,
    cache_path: Path,
) -> np.ndarray:
    digest_payload = {
        "model_id": provider.metadata.model_id,
        "revision": provider.metadata.revision,
        "texts": texts,
    }
    digest = hashlib.sha256(
        json.dumps(digest_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if cache_path.exists():
        try:
            cached = np.load(cache_path, allow_pickle=False)
            if str(cached["digest"].item()) == digest:
                vectors = np.asarray(cached["vectors"], dtype=np.float32)
                if vectors.shape == (len(texts), provider.metadata.dimension):
                    return vectors
        except (OSError, ValueError, KeyError):
            pass
    vectors = provider.embed(texts, batch_size=batch_size)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, digest=np.array(digest), vectors=vectors)
    return vectors

