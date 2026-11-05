from pathlib import Path

import pytest

from finrag_indic.config import Settings
from finrag_indic.contracts import ChatRequest, ProviderRequest, ProviderResponse
from finrag_indic.errors import (
    ProviderConfigurationError,
    ProviderRequestError,
    RetrievalUnavailableError,
)
from finrag_indic.knowledge.models import Chunk, Language, RetrievalHit
from finrag_indic.providers.mock import DeterministicMockProvider
from finrag_indic.rag.safety import contains_sensitive_financial_information
from finrag_indic.rag.service import ChatService


def hit(*, rank: int = 1, document_language: Language = Language.ENGLISH) -> RetrievalHit:
    chunk = Chunk(
        chunk_id=f"chunk-{rank}",
        document_id="rbi-source",
        language=document_language,
        text="Never share a UPI PIN with anyone.",
        page_number=25,
        section_heading="UPI safety",
        chunk_index=rank - 1,
        token_count=8,
        content_sha256=(str(rank) * 64)[:64],
        source_url="https://www.rbi.org.in/source.pdf",
        title="RBI Financial Awareness",
        issuing_organization="RBI",
    )
    return RetrievalHit(rank=rank, score=0.75, chunk=chunk)


class FakeRetriever:
    def __init__(self, hits: list[RetrievalHit] | None = None, *, fail: bool = False) -> None:
        self.hits = hits or []
        self.fail = fail
        self.last_top_k: int | None = None

    async def retrieve(self, question: str, *, top_k: int) -> list[RetrievalHit]:
        del question
        self.last_top_k = top_k
        if self.fail:
            raise RetrievalUnavailableError("unavailable")
        return self.hits


class RecordingProvider:
    def __init__(self, text: str = "Use the official safety guidance [S1].") -> None:
        self.text = text
        self.requests: list[ProviderRequest] = []

    async def generate(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        return ProviderResponse(text=self.text, provider="recording", model="test-model")


class FailingProvider:
    async def generate(self, request: ProviderRequest) -> ProviderResponse:
        del request
        raise ProviderRequestError("temporary failure", retryable=True)


def settings(tmp_path: Path) -> Settings:
    return Settings(
        environment="test",
        retrieval_index_path=tmp_path / "index",
        retrieval_top_k=3,
        _env_file=None,
    )


@pytest.mark.anyio
@pytest.mark.parametrize("language", ["en", "hi", "ta"])
async def test_selected_language_reaches_grounded_prompt(
    tmp_path: Path, language: str
) -> None:
    provider = RecordingProvider()
    retriever = FakeRetriever([hit()])
    service = ChatService(
        settings=settings(tmp_path), retriever=retriever, provider=provider
    )

    result = await service.chat(ChatRequest(question="UPI safety?", language=language))

    assert result.language == language
    assert not result.abstained
    assert result.citations[0].source_url.host == "www.rbi.org.in"
    assert result.evidence[0].chunk_id == "chunk-1"
    assert provider.requests[0].language == language
    assert {"en": "English", "hi": "Hindi", "ta": "Tamil"}[language] in provider.requests[
        0
    ].prompt
    assert retriever.last_top_k == 3


@pytest.mark.anyio
async def test_empty_retrieval_abstains_without_generation(tmp_path: Path) -> None:
    provider = RecordingProvider()
    service = ChatService(
        settings=settings(tmp_path), retriever=FakeRetriever(), provider=provider
    )

    result = await service.chat(ChatRequest(question="Unknown topic", language="en"))

    assert result.abstained
    assert result.error and result.error.code == "empty_retrieval"
    assert provider.requests == []


@pytest.mark.anyio
async def test_invalid_generated_citation_forces_abstention(tmp_path: Path) -> None:
    service = ChatService(
        settings=settings(tmp_path),
        retriever=FakeRetriever([hit()]),
        provider=RecordingProvider("Unsupported statement [S99]."),
    )

    result = await service.chat(ChatRequest(question="Question", language="en"))

    assert result.abstained
    assert result.error and result.error.code == "invalid_generated_citations"


@pytest.mark.anyio
async def test_provider_failure_is_structured_and_retryable(tmp_path: Path) -> None:
    service = ChatService(
        settings=settings(tmp_path),
        retriever=FakeRetriever([hit()]),
        provider=FailingProvider(),
    )

    result = await service.chat(ChatRequest(question="Question", language="en"))

    assert result.abstained
    assert result.error and result.error.code == "provider_request_failed"
    assert result.error.retryable


@pytest.mark.anyio
async def test_retrieval_failure_is_structured(tmp_path: Path) -> None:
    service = ChatService(
        settings=settings(tmp_path),
        retriever=FakeRetriever(fail=True),
        provider=DeterministicMockProvider(),
    )

    result = await service.chat(ChatRequest(question="Question", language="ta"))

    assert result.abstained
    assert result.error and result.error.code == "retrieval_unavailable"


@pytest.mark.anyio
async def test_missing_provider_configuration_fails_before_retrieval(
    tmp_path: Path,
) -> None:
    retriever = FakeRetriever([hit()])
    service = ChatService(
        settings=settings(tmp_path),
        retriever=retriever,
        provider=None,
        provider_error=ProviderConfigurationError("missing credential"),
    )

    result = await service.chat(ChatRequest(question="Question", language="en"))

    assert result.abstained
    assert result.error and result.error.code == "provider_not_configured"
    assert retriever.last_top_k is None


@pytest.mark.anyio
async def test_sensitive_information_is_not_retrieved_or_generated(tmp_path: Path) -> None:
    provider = RecordingProvider()
    retriever = FakeRetriever([hit()])
    service = ChatService(
        settings=settings(tmp_path), retriever=retriever, provider=provider
    )

    result = await service.chat(
        ChatRequest(question="My OTP is 123456, what should I do?", language="en")
    )

    assert result.abstained
    assert result.error and result.error.code == "sensitive_financial_information"
    assert retriever.last_top_k is None
    assert provider.requests == []


def test_safety_question_without_actual_secret_is_allowed() -> None:
    assert not contains_sensitive_financial_information(
        "Should I ever share my UPI PIN or OTP with another person?"
    )
