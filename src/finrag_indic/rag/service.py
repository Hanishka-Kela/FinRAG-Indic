import re
import time

from finrag_indic.config import Settings
from finrag_indic.contracts import (
    ChatError,
    ChatRequest,
    ChatResponse,
    EvidenceReference,
    LatencyInfo,
    ProviderRequest,
    SourceCitation,
    SupportedLanguage,
)
from finrag_indic.errors import (
    ProviderConfigurationError,
    ProviderRequestError,
    RetrievalUnavailableError,
)
from finrag_indic.providers.base import TextProvider
from finrag_indic.providers.mock import DeterministicMockProvider
from finrag_indic.providers.openai import OpenAIProvider
from finrag_indic.rag.prompting import build_grounded_prompt
from finrag_indic.rag.retriever import EvidenceRetriever, LocalFaissEvidenceRetriever
from finrag_indic.rag.safety import contains_sensitive_financial_information


_LOCALIZED = {
    "insufficient": {
        "en": "The available sources do not contain enough evidence to answer this safely.",
        "hi": "उपलब्ध स्रोतों में सुरक्षित उत्तर देने के लिए पर्याप्त प्रमाण नहीं है।",
        "ta": "பாதுகாப்பாகப் பதிலளிக்க கிடைக்கக்கூடிய ஆதாரங்கள் போதுமானதாக இல்லை.",
    },
    "sensitive": {
        "en": "Please remove any PIN, OTP, password, card, or account details and ask again.",
        "hi": "कृपया PIN, OTP, पासवर्ड, कार्ड या खाता विवरण हटाकर फिर पूछें।",
        "ta": "PIN, OTP, கடவுச்சொல், அட்டை அல்லது கணக்கு விவரங்களை நீக்கி மீண்டும் கேளுங்கள்.",
    },
    "unavailable": {
        "en": "The service could not generate a grounded answer right now.",
        "hi": "सेवा अभी प्रमाण-आधारित उत्तर तैयार नहीं कर सकी।",
        "ta": "இப்போது ஆதார அடிப்படையிலான பதிலைச் சேவை உருவாக்க முடியவில்லை.",
    },
}


def _localized(kind: str, language: SupportedLanguage) -> str:
    return _LOCALIZED[kind][language]


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


def _citations(evidence: list[EvidenceReference]) -> list[SourceCitation]:
    return [
        SourceCitation(
            citation_id=item.citation_id,
            document_id=item.document_id,
            title=item.title,
            page_number=item.page_number,
            source_url=item.source_url,
        )
        for item in evidence
    ]


def _invalid_citation_ids(answer: str, evidence: list[EvidenceReference]) -> set[str]:
    cited = set(re.findall(r"\[(S\d+)\]", answer))
    valid = {item.citation_id for item in evidence}
    if not cited:
        return {"missing"}
    return cited - valid


class ChatService:
    def __init__(
        self,
        *,
        settings: Settings,
        retriever: EvidenceRetriever,
        provider: TextProvider | None,
        provider_error: ProviderConfigurationError | None = None,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.provider = provider
        self.provider_error = provider_error

    def _response(
        self,
        *,
        started: float,
        language: SupportedLanguage,
        answer: str,
        evidence: list[EvidenceReference],
        abstained: bool,
        retrieval_ms: float = 0,
        generation_ms: float = 0,
        provider: str | None = None,
        model: str | None = None,
        error: ChatError | None = None,
    ) -> ChatResponse:
        return ChatResponse(
            answer=answer,
            language=language,
            evidence=evidence,
            citations=_citations(evidence),
            abstained=abstained,
            provider=provider,
            model=model,
            latency=LatencyInfo(
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
                total_ms=(time.perf_counter() - started) * 1000,
            ),
            error=error,
        )

    async def chat(self, request: ChatRequest) -> ChatResponse:
        started = time.perf_counter()
        if len(request.question) > self.settings.chat_max_question_chars:
            return self._response(
                started=started,
                language=request.language,
                answer=_localized("unavailable", request.language),
                evidence=[],
                abstained=True,
                error=ChatError(
                    code="question_too_long",
                    message=(
                        f"Question exceeds {self.settings.chat_max_question_chars} characters"
                    ),
                ),
            )
        if contains_sensitive_financial_information(request.question):
            return self._response(
                started=started,
                language=request.language,
                answer=_localized("sensitive", request.language),
                evidence=[],
                abstained=True,
                error=ChatError(
                    code="sensitive_financial_information",
                    message="Potential credentials or financial identifiers were not processed",
                ),
            )
        if self.provider_error is not None or self.provider is None:
            message = str(self.provider_error or "The LLM provider is not configured")
            return self._response(
                started=started,
                language=request.language,
                answer=_localized("unavailable", request.language),
                evidence=[],
                abstained=True,
                error=ChatError(code="provider_not_configured", message=message),
            )
        retrieval_started = time.perf_counter()
        try:
            hits = await self.retriever.retrieve(
                request.question, top_k=self.settings.retrieval_top_k
            )
        except RetrievalUnavailableError:
            return self._response(
                started=started,
                language=request.language,
                answer=_localized("unavailable", request.language),
                evidence=[],
                abstained=True,
                error=ChatError(
                    code="retrieval_unavailable",
                    message="The local retrieval service is unavailable",
                    retryable=True,
                ),
            )
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
        evidence = _evidence_from_hits(hits)
        if not evidence:
            return self._response(
                started=started,
                language=request.language,
                answer=_localized("insufficient", request.language),
                evidence=[],
                abstained=True,
                retrieval_ms=retrieval_ms,
                error=ChatError(
                    code="empty_retrieval",
                    message="No source passages were retrieved",
                ),
            )
        prompt = build_grounded_prompt(
            question=request.question,
            language=request.language,
            evidence=evidence,
        )
        generation_started = time.perf_counter()
        try:
            generated = await self.provider.generate(
                ProviderRequest(prompt=prompt, language=request.language)
            )
        except ProviderConfigurationError as exc:
            return self._response(
                started=started,
                language=request.language,
                answer=_localized("unavailable", request.language),
                evidence=evidence,
                abstained=True,
                retrieval_ms=retrieval_ms,
                generation_ms=(time.perf_counter() - generation_started) * 1000,
                error=ChatError(code="provider_not_configured", message=str(exc)),
            )
        except ProviderRequestError as exc:
            return self._response(
                started=started,
                language=request.language,
                answer=_localized("unavailable", request.language),
                evidence=evidence,
                abstained=True,
                retrieval_ms=retrieval_ms,
                generation_ms=(time.perf_counter() - generation_started) * 1000,
                error=ChatError(
                    code="provider_request_failed",
                    message=str(exc),
                    retryable=exc.retryable,
                ),
            )
        generation_ms = (time.perf_counter() - generation_started) * 1000
        if not generated.abstained:
            invalid = _invalid_citation_ids(generated.text, evidence)
            if invalid:
                return self._response(
                    started=started,
                    language=request.language,
                    answer=_localized("insufficient", request.language),
                    evidence=evidence,
                    abstained=True,
                    retrieval_ms=retrieval_ms,
                    generation_ms=generation_ms,
                    provider=generated.provider,
                    model=generated.model,
                    error=ChatError(
                        code="invalid_generated_citations",
                        message="The generated answer did not contain only valid source markers",
                    ),
                )
        return self._response(
            started=started,
            language=request.language,
            answer=generated.text,
            evidence=evidence,
            abstained=generated.abstained,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            provider=generated.provider,
            model=generated.model,
        )


def create_chat_service(settings: Settings) -> ChatService:
    retriever = LocalFaissEvidenceRetriever(settings)
    if settings.llm_provider == "mock":
        return ChatService(
            settings=settings,
            retriever=retriever,
            provider=DeterministicMockProvider(),
        )
    try:
        provider = OpenAIProvider(settings)
    except ProviderConfigurationError as exc:
        return ChatService(
            settings=settings,
            retriever=retriever,
            provider=None,
            provider_error=exc,
        )
    return ChatService(settings=settings, retriever=retriever, provider=provider)
