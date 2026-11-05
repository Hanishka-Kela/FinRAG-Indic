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
from finrag_indic.rag.service import (
    ChatService,
    _invalid_citation_ids,
    _normalize_citation_format,
)


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
        self.last_question: str | None = None

    async def retrieve(self, question: str, *, top_k: int) -> list[RetrievalHit]:
        self.last_question = question
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


class SequenceProvider:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.requests: list[ProviderRequest] = []

    async def generate(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        return ProviderResponse(
            text=next(self.responses), provider="recording", model="test-model"
        )


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
async def test_retrieval_expands_financial_abbreviation_but_prompt_keeps_question(
    tmp_path: Path,
) -> None:
    provider = RecordingProvider()
    retriever = FakeRetriever([hit()])
    service = ChatService(
        settings=settings(tmp_path), retriever=retriever, provider=provider
    )

    await service.chat(ChatRequest(question="What is SI?", language="en"))

    assert retriever.last_question == "What is SI (Simple Interest)?"
    assert "What is SI?" in provider.requests[0].prompt


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("language", "expected_question"),
    [
        ("en", "Does ₹1,000 refer to your monthly income or your monthly repayment capacity?"),
        ("hi", "क्या ₹1,000 आपकी मासिक आय है या आपकी मासिक पुनर्भुगतान क्षमता?"),
        ("ta", "₹1,000 என்பது உங்கள் மாத வருமானமா அல்லது மாதத் திருப்பிச் செலுத்தும் திறனா?"),
    ],
)
async def test_personalized_loan_question_gets_helpful_abstention(
    tmp_path: Path, language: str, expected_question: str
) -> None:
    provider = RecordingProvider(
        "I cannot decide this for you. Please compare your income and expenses [S1]."
    )
    service = ChatService(
        settings=settings(tmp_path),
        retriever=FakeRetriever([hit()]),
        provider=provider,
    )

    result = await service.chat(
        ChatRequest(
            question="Kya mujhe ₹1,00,000 ka loan lena chahiye? Meri monthly income 1000 repay hai.",
            language=language,
        )
    )

    assert result.abstained
    assert result.needs_clarification
    assert result.clarification_question == expected_question
    assert result.abstention_reason == "personalized_decision"
    assert "[S1]" in result.answer
    assert "personalized financial decision" in provider.requests[0].prompt


@pytest.mark.anyio
async def test_general_emi_question_is_not_personalized_abstention(tmp_path: Path) -> None:
    provider = RecordingProvider("EMI is a regular payment [S1].")
    service = ChatService(
        settings=settings(tmp_path), retriever=FakeRetriever([hit()]), provider=provider
    )

    result = await service.chat(ChatRequest(question="What is EMI?", language="en"))

    assert not result.abstained
    assert not result.needs_clarification


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


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("Claim [S1].", "Claim [S1]."),
        ("Claim [S 1, S2].", "Claim [S1] [S2]."),
        ("Claim [s1;  S 2].", "Claim [S1] [S2]."),
        ("Claim [S1, S1, S2].", "Claim [S1] [S2]."),
    ],
)
def test_citation_format_variations_are_normalized(answer: str, expected: str) -> None:
    assert _normalize_citation_format(answer) == expected


def test_citation_validation_rejects_unknown_and_mixed_ids() -> None:
    evidence = [hit()]
    from finrag_indic.rag.service import _evidence_from_hits

    references = _evidence_from_hits(evidence)
    assert _invalid_citation_ids("Claim [S99].", references) == {"S99"}
    assert _invalid_citation_ids("Claim [S1, S99].", references) == {"S99"}
    assert _invalid_citation_ids("Claim without a marker.", references) == {"missing"}


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("language", "text"),
    [
        ("en", "Grounded answer [S1]."),
        ("hi", "उत्तर [S1]।"),
        ("ta", "பதில் [S1]."),
    ],
)
async def test_invalid_citation_is_corrected_once(
    tmp_path: Path, language: str, text: str
) -> None:
    provider = SequenceProvider(["गलत उत्तर [S99].", text])
    service = ChatService(
        settings=settings(tmp_path),
        retriever=FakeRetriever([hit()]),
        provider=provider,
    )

    result = await service.chat(ChatRequest(question="Question", language=language))

    assert not result.abstained
    assert result.answer == text
    assert len(provider.requests) == 2
    assert result.latency.correction_ms >= 0
    assert result.latency.generation_ms >= 0


@pytest.mark.anyio
async def test_failed_citation_correction_abstains_safely(tmp_path: Path) -> None:
    provider = SequenceProvider(["Unsupported [S99].", "Still unsupported [S99]."])
    service = ChatService(
        settings=settings(tmp_path),
        retriever=FakeRetriever([hit()]),
        provider=provider,
    )

    result = await service.chat(ChatRequest(question="Question", language="en"))

    assert result.abstained
    assert result.answer == "The available sources do not contain enough evidence to answer this safely."
    assert result.error and result.error.code == "invalid_generated_citations"
    assert "safely linked" in result.error.message
    assert result.latency.correction_ms >= 0


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
