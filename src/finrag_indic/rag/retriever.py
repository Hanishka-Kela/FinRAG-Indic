import asyncio
import threading
from pathlib import Path
from typing import Protocol

from finrag_indic.config import Settings
from finrag_indic.errors import EmbeddingError, IndexCompatibilityError, RetrievalUnavailableError
from finrag_indic.knowledge.models import RetrievalHit
from finrag_indic.retrieval.embeddings import BgeM3Embedder


class EvidenceRetriever(Protocol):
    async def retrieve(self, question: str, *, top_k: int) -> list[RetrievalHit]: ...


class LocalFaissEvidenceRetriever:
    """Lazy BGE-M3 + FAISS adapter for the versioned local corpus."""

    def __init__(self, settings: Settings) -> None:
        self.index_path = Path(settings.retrieval_index_path)
        self.embedder = BgeM3Embedder(
            model_id=settings.embedding_model,
            revision=settings.embedding_revision,
            cache_folder=Path("data/models/sentence-transformers"),
            device=settings.embedding_device,
        )
        self._retriever = None
        self._load_lock = threading.Lock()

    @property
    def available(self) -> bool:
        return all(
            (self.index_path / filename).is_file()
            for filename in ("vectors.faiss", "metadata.json", "chunks.jsonl")
        )

    def _load(self):  # type: ignore[no-untyped-def]
        if self._retriever is not None:
            return self._retriever
        with self._load_lock:
            if self._retriever is not None:
                return self._retriever
            if not self.available:
                raise RetrievalUnavailableError(
                    f"Retrieval index is missing or incomplete: {self.index_path}"
                )
            try:
                # Load PyTorch/model before importing FAISS to avoid the observed
                # macOS native-library initialization conflict.
                self.embedder.load()
                from finrag_indic.retrieval.index import FaissRetriever

                self._retriever = FaissRetriever.load(
                    self.index_path, expected_embedding=self.embedder.metadata
                )
            except (EmbeddingError, IndexCompatibilityError, OSError) as exc:
                raise RetrievalUnavailableError(
                    "The multilingual retrieval index or embedding model could not be loaded"
                ) from exc
        return self._retriever

    def _retrieve_sync(self, question: str, top_k: int) -> list[RetrievalHit]:
        retriever = self._load()
        try:
            vector = self.embedder.embed([question], batch_size=1)[0]
            return retriever.search(vector, top_k=top_k)
        except (EmbeddingError, IndexCompatibilityError) as exc:
            raise RetrievalUnavailableError("Question retrieval failed") from exc

    async def retrieve(self, question: str, *, top_k: int) -> list[RetrievalHit]:
        return await asyncio.to_thread(self._retrieve_sync, question, top_k)
