import hashlib
import json
from pathlib import Path

import faiss
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from finrag_indic.errors import EmptyCorpusError, IndexCompatibilityError
from finrag_indic.knowledge.models import Chunk, Language, RetrievalHit
from finrag_indic.retrieval.embeddings import EmbeddingMetadata


class IndexMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    index_type: str = "IndexFlatIP"
    corpus_view: str
    vector_count: int = Field(ge=0)
    dimension: int = Field(gt=0)
    embedding: EmbeddingMetadata
    chunks_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class FaissRetriever:
    def __init__(
        self,
        *,
        index: faiss.Index,
        chunks: list[Chunk],
        metadata: IndexMetadata,
    ) -> None:
        self.index = index
        self.chunks = chunks
        self.metadata = metadata
        self._validate()

    @classmethod
    def build(
        cls,
        *,
        chunks: list[Chunk],
        vectors: np.ndarray,
        embedding: EmbeddingMetadata,
        corpus_view: str,
    ) -> "FaissRetriever":
        if not chunks:
            raise EmptyCorpusError("cannot build an index from an empty corpus")
        if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
            raise IndexCompatibilityError("duplicate chunk_id values cannot be indexed")
        if len({chunk.content_sha256 for chunk in chunks}) != len(chunks):
            raise IndexCompatibilityError("duplicate chunk content cannot be indexed")
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim != 2 or array.shape != (len(chunks), embedding.dimension):
            raise IndexCompatibilityError("vector shape is incompatible with chunk metadata")
        norms = np.linalg.norm(array, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4):
            raise IndexCompatibilityError("vectors must be normalized before indexing")
        index = faiss.IndexFlatIP(embedding.dimension)
        index.add(np.ascontiguousarray(array))
        metadata = IndexMetadata(
            corpus_view=corpus_view,
            vector_count=len(chunks),
            dimension=embedding.dimension,
            embedding=embedding,
            chunks_sha256=_chunks_digest(chunks),
        )
        return cls(index=index, chunks=chunks, metadata=metadata)

    def search(self, query_vector: np.ndarray, *, top_k: int) -> list[RetrievalHit]:
        if self.index.ntotal == 0:
            raise EmptyCorpusError("cannot search an empty index")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        query = np.asarray(query_vector, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.shape != (1, self.metadata.dimension):
            raise IndexCompatibilityError("query vector has an incompatible dimension")
        norm = np.linalg.norm(query)
        if norm == 0:
            raise IndexCompatibilityError("query vector cannot be zero")
        query = np.ascontiguousarray(query / norm, dtype=np.float32)
        limit = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query, limit)
        return [
            RetrievalHit(rank=rank, score=float(score), chunk=self.chunks[int(index)])
            for rank, (score, index) in enumerate(zip(scores[0], indices[0]), start=1)
            if index >= 0
        ]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "vectors.faiss"))
        (directory / "metadata.json").write_text(
            self.metadata.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                handle.write(chunk.model_dump_json() + "\n")

    @classmethod
    def load(
        cls,
        directory: Path,
        *,
        expected_embedding: EmbeddingMetadata | None = None,
    ) -> "FaissRetriever":
        required = ["vectors.faiss", "metadata.json", "chunks.jsonl"]
        missing = [name for name in required if not (directory / name).exists()]
        if missing:
            raise IndexCompatibilityError(f"index files are missing: {', '.join(missing)}")
        try:
            index = faiss.read_index(str(directory / "vectors.faiss"))
            metadata = IndexMetadata.model_validate_json(
                (directory / "metadata.json").read_text(encoding="utf-8")
            )
            chunks = [
                Chunk.model_validate_json(line)
                for line in (directory / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, RuntimeError, ValueError) as exc:
            raise IndexCompatibilityError(f"failed to load persisted index: {exc}") from exc
        if expected_embedding and metadata.embedding != expected_embedding:
            raise IndexCompatibilityError("persisted index uses a different embedding model")
        return cls(index=index, chunks=chunks, metadata=metadata)

    def _validate(self) -> None:
        if self.metadata.index_type != "IndexFlatIP":
            raise IndexCompatibilityError("only IndexFlatIP is supported")
        if self.index.d != self.metadata.dimension:
            raise IndexCompatibilityError("FAISS dimension does not match metadata")
        if self.index.ntotal != len(self.chunks) or self.index.ntotal != self.metadata.vector_count:
            raise IndexCompatibilityError("FAISS vector count does not match chunk mapping")
        if self.metadata.embedding.dimension != self.metadata.dimension:
            raise IndexCompatibilityError("embedding dimension does not match index dimension")
        if _chunks_digest(self.chunks) != self.metadata.chunks_sha256:
            raise IndexCompatibilityError("chunk mapping checksum does not match metadata")


def corpus_view(chunks: list[Chunk], *, multilingual: bool) -> list[Chunk]:
    if multilingual:
        return list(chunks)
    return [chunk for chunk in chunks if chunk.language == Language.ENGLISH]


def controlled_coverage_views(chunks: list[Chunk]) -> tuple[list[Chunk], list[Chunk]]:
    groups: dict[str, set[Language]] = {}
    for chunk in chunks:
        if chunk.coverage_group:
            groups.setdefault(chunk.coverage_group, set()).add(chunk.language)
    paired = {
        group
        for group, languages in groups.items()
        if {Language.ENGLISH, Language.HINDI, Language.TAMIL}.issubset(languages)
    }
    multilingual = [chunk for chunk in chunks if chunk.coverage_group in paired]
    english = [chunk for chunk in multilingual if chunk.language == Language.ENGLISH]
    return english, multilingual


def _chunks_digest(chunks: list[Chunk]) -> str:
    payload = "\n".join(
        f"{chunk.chunk_id}:{chunk.content_sha256}" for chunk in chunks
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

