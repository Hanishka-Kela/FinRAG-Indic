import asyncio
import csv
import json
import re
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from finrag_indic.config import Settings
from finrag_indic.contracts import EvidenceReference, ProviderRequest, ProviderResponse
from finrag_indic.errors import (
    EmbeddingError,
    IndexCompatibilityError,
    ProviderConfigurationError,
    ProviderRequestError,
    RetrievalUnavailableError,
)
from finrag_indic.evaluation.query_normalization import normalize_financial_query
from finrag_indic.providers.base import TextProvider
from finrag_indic.providers.mock import DeterministicMockProvider
from finrag_indic.providers.openai import OpenAIProvider
from finrag_indic.rag.prompting import build_grounded_prompt
from finrag_indic.retrieval.embeddings import BgeM3Embedder

if TYPE_CHECKING:
    from finrag_indic.retrieval.index import FaissRetriever


Condition = Literal["llm_only", "rag_english", "rag_multilingual"]


class ExperimentVariant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    canonical_id: str
    topic: str
    language: Literal["en", "hi", "ta"]
    question: str = Field(min_length=1)
    translation_status: str
    evidence_status: str


def load_variants(path: Path, *, limit: int | None = None) -> list[ExperimentVariant]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    variants = [
        ExperimentVariant(
            canonical_id=item["canonical_id"],
            topic=item["topic"],
            language=language,
            question=variant["question"],
            translation_status=variant["translation_status"],
            evidence_status=variant["evidence_status"],
        )
        for item in payload
        for language, variant in item["variants"].items()
    ]
    if limit is None:
        return variants
    allowed = {item["canonical_id"] for item in _canonical_items(payload, limit)}
    return [item for item in variants if item.canonical_id in allowed]


def load_probe_variants(path: Path) -> list[ExperimentVariant]:
    return load_variants(path)


def _canonical_items(payload: list[dict[str, object]], limit: int | None) -> list[dict[str, object]]:
    return payload if limit is None else payload[:limit]


def citation_reference_validity(
    answer: str, evidence: list[EvidenceReference]
) -> tuple[bool, list[str]]:
    markers = set(re.findall(r"\[(S\d+)\]", answer))
    valid = {item.citation_id for item in evidence}
    invalid = sorted(markers - valid)
    if evidence and not markers:
        invalid.append("missing")
    return not invalid, invalid


def _language_signal(answer: str, language: str) -> bool:
    if language == "hi":
        return any("\u0900" <= char <= "\u097f" for char in answer)
    if language == "ta":
        return any("\u0b80" <= char <= "\u0bff" for char in answer)
    return any(char.isascii() and char.isalpha() for char in answer)


def _evidence_from_hits(hits) -> list[EvidenceReference]:  # type: ignore[no-untyped-def]
    return [
        EvidenceReference(
            citation_id=f"S{index}",
            rank=hit.rank,
            score=hit.score,
            chunk_id=hit.chunk.chunk_id,
            document_id=hit.chunk.document_id,
            title=hit.chunk.title,
            document_language=hit.chunk.language.value,
            page_number=hit.chunk.page_number,
            source_url=hit.chunk.source_url,
            passage=hit.chunk.text,
            review_flags=hit.chunk.review_flags,
        )
        for index, hit in enumerate(hits, start=1)
    ]


class ExperimentRunner:
    def __init__(
        self,
        *,
        settings: Settings,
        provider: TextProvider | None = None,
        retry_attempts: int = 2,
    ) -> None:
        self.settings = settings
        self.provider = provider or self._configured_provider()
        self.retry_attempts = retry_attempts
        self.embedder = BgeM3Embedder(
            model_id=settings.embedding_model,
            revision=settings.embedding_revision,
            cache_folder=Path("data/models/sentence-transformers"),
            device=settings.embedding_device,
        )
        self._retrievers: dict[Condition, "FaissRetriever"] = {}

    def _configured_provider(self) -> TextProvider:
        if self.settings.llm_provider == "mock":
            return DeterministicMockProvider()
        return OpenAIProvider(self.settings)

    def _retriever(self, condition: Condition) -> "FaissRetriever":
        if condition == "rag_english":
            path = Path("data/indexes/english")
        elif condition == "rag_multilingual":
            path = Path("data/indexes/multilingual-genuine-v2")
        else:
            raise ValueError(f"{condition} has no retriever")
        if condition not in self._retrievers:
            self.embedder.load()
            from finrag_indic.retrieval.index import FaissRetriever

            self._retrievers[condition] = FaissRetriever.load(
                path, expected_embedding=self.embedder.metadata
            )
        return self._retrievers[condition]

    async def _generate(self, request: ProviderRequest) -> ProviderResponse:
        last_error: ProviderRequestError | None = None
        for attempt in range(self.retry_attempts + 1):
            try:
                return await self.provider.generate(request)
            except ProviderRequestError as exc:
                last_error = exc
                if not exc.retryable or attempt == self.retry_attempts:
                    raise
                await asyncio.sleep(0.5 * (2**attempt))
        raise last_error or ProviderRequestError("Provider generation failed")

    async def run_case(
        self, variant: ExperimentVariant, condition: Condition
    ) -> dict[str, object]:
        started = time.perf_counter()
        normalized_question = normalize_financial_query(variant.question)
        evidence: list[EvidenceReference] = []
        retrieval_error: str | None = None
        retrieval_started = time.perf_counter()
        if condition != "llm_only":
            try:
                vector = self.embedder.embed([normalized_question], batch_size=1)[0]
                hits = self._retriever(condition).search(
                    vector, top_k=self.settings.retrieval_top_k
                )
                evidence = _evidence_from_hits(hits)
            except (
                EmbeddingError,
                IndexCompatibilityError,
                OSError,
                RetrievalUnavailableError,
                ValueError,
            ) as exc:
                retrieval_error = str(exc)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
        if condition == "llm_only":
            prompt = (
                f"Answer in the selected language ({variant.language}) using general "
                "financial-literacy knowledge. Do not invent citations or claim access "
                "to sources. If you cannot answer safely, begin with "
                "INSUFFICIENT_EVIDENCE:.\n\nQUESTION:\n" + variant.question
            )
        elif evidence:
            prompt = build_grounded_prompt(
                question=variant.question, language=variant.language, evidence=evidence
            )
        else:
            prompt = ""
        generation_ms = 0.0
        error: dict[str, object] | None = None
        generated: ProviderResponse | None = None
        if retrieval_error is None and prompt:
            generation_started = time.perf_counter()
            try:
                generated = await self._generate(
                    ProviderRequest(prompt=prompt, language=variant.language)
                )
            except (ProviderConfigurationError, ProviderRequestError) as exc:
                error = {
                    "code": "provider_error",
                    "message": str(exc),
                    "retryable": getattr(exc, "retryable", False),
                }
            generation_ms = (time.perf_counter() - generation_started) * 1000
        elif retrieval_error is not None:
            error = {"code": "retrieval_error", "message": retrieval_error}
        else:
            error = {"code": "empty_retrieval", "message": "No evidence retrieved"}
        answer = generated.text if generated else ""
        citation_valid, invalid_citations = citation_reference_validity(answer, evidence)
        return {
            "canonical_id": variant.canonical_id,
            "topic": variant.topic,
            "language": variant.language,
            "question": variant.question,
            "normalized_question": normalized_question,
            "translation_status": variant.translation_status,
            "evidence_status": variant.evidence_status,
            "condition": condition,
            "http_success": error is None,
            "answer_generated": bool(answer),
            "answer": answer,
            "abstained": generated.abstained if generated else False,
            "answer_language_signal": _language_signal(answer, variant.language),
            "citation_count": len(evidence) if generated else 0,
            "citation_reference_valid": (
                None if condition == "llm_only" else citation_valid
            ),
            "invalid_citations": invalid_citations,
            "model": generated.model if generated else getattr(self.provider, "model", None),
            "provider": generated.provider if generated else None,
            "retrieval_index": (
                "none"
                if condition == "llm_only"
                else (
                    "english-v1"
                    if condition == "rag_english"
                    else "multilingual-genuine-v2"
                )
            ),
            "retrieval_ms": round(retrieval_ms, 3),
            "generation_ms": round(generation_ms, 3),
            "total_ms": round((time.perf_counter() - started) * 1000, 3),
            "retrieved_chunks": [
                {
                    "chunk_id": item.chunk_id,
                    "document_id": item.document_id,
                    "title": item.title,
                    "document_language": item.document_language,
                    "page_number": item.page_number,
                    "score": item.score,
                    "rank": item.rank,
                    "source_url": str(item.source_url),
                }
                for item in evidence
            ],
            "error": error,
        }

    async def close(self) -> None:
        close = getattr(self.provider, "close", None)
        if close is not None:
            await close()


def summarize_results(records: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for condition in ("llm_only", "rag_english", "rag_multilingual"):
        selected = [r for r in records if r["condition"] == condition]
        latencies = [float(r["total_ms"]) for r in selected if r["http_success"]]
        source_languages = Counter(
            chunk["document_language"]
            for record in selected
            for chunk in record.get("retrieved_chunks", [])  # type: ignore[union-attr]
        )
        output.append(
            {
                "condition": condition,
                "records": len(selected),
                "success_rate": (
                    sum(bool(r["http_success"]) for r in selected) / len(selected)
                    if selected
                    else None
                ),
                "mean_latency_ms": statistics.mean(latencies) if latencies else None,
                "median_latency_ms": statistics.median(latencies) if latencies else None,
                "abstention_rate": (
                    sum(bool(r["abstained"]) for r in selected) / len(selected)
                    if selected
                    else None
                ),
                "language_signal_rate": (
                    sum(bool(r["answer_language_signal"]) for r in selected) / len(selected)
                    if selected
                    else None
                ),
                "citation_valid_rate": (
                    None
                    if condition == "llm_only"
                    else (
                    sum(
                        bool(r["citation_reference_valid"])
                        for r in selected
                        if r["citation_reference_valid"] is not None
                    )
                    / sum(
                        r["citation_reference_valid"] is not None for r in selected
                    )
                    if any(
                        r["citation_reference_valid"] is not None for r in selected
                    )
                    else None
                    )
                ),
                "retrieved_source_language_counts": dict(source_languages),
            }
        )
    return output


def write_reviewer_template(records: list[dict[str, object]], path: Path) -> None:
    fields = [
        "canonical_id", "topic", "language", "condition", "question", "answer",
        "factual_correctness_1_5", "relevance_1_5", "grounding_1_5",
        "language_consistency_1_5", "clarity_1_5", "appropriate_abstention_1_5",
        "reviewer_notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fields})
