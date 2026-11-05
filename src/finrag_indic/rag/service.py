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
from finrag_indic.evaluation.query_normalization import normalize_financial_query
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
    "clarification": {
        "en": "Does ₹1,000 refer to your monthly income or your monthly repayment capacity?",
        "hi": "क्या ₹1,000 आपकी मासिक आय है या आपकी मासिक पुनर्भुगतान क्षमता?",
        "ta": "₹1,000 என்பது உங்கள் மாத வருமானமா அல்லது மாதத் திருப்பிச் செலுத்தும் திறனா?",
    },
}


def _localized(kind: str, language: SupportedLanguage) -> str:
    return _LOCALIZED[kind][language]


_PERSONALIZED_DECISION_PATTERNS = (
    r"\bshould\s+i\b",
    r"\bcan\s+i\s+afford\b",
    r"\bwhich\s+(?:investment|stock|fund|scheme)\b",
    r"\bwhat\s+should\s+i\s+(?:buy|invest)\b",
    r"\bloan\s+(?:lena|len|le)\s+(?:chahiye|chaahiye)\b",
    r"\b(?:invest|buy)\s+kar(?:u|oon|na)\b",
    r"(?:लेना|लेनी)\s+चाहिए",
    r"क्या मैं .*?(?:ले सकता|निवेश कर सकता)",
    r"(?:வாங்க|எடுக்க)லாமா",
    r"(?:முதலீடு|கடன்).*?(?:வேண்டுமா|செய்யலாமா)",
)
_PERSONALIZED_DECISION_RE = re.compile(
    "|".join(_PERSONALIZED_DECISION_PATTERNS), re.IGNORECASE
)


def _is_personalized_decision(question: str) -> bool:
    return bool(_PERSONALIZED_DECISION_RE.search(question))


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


_CITATION_GROUP = re.compile(
    r"\[\s*S\s*\d+(?:\s*(?:[,;]|\s)\s*S?\s*\d+)*\s*\]",
    re.IGNORECASE,
)
_CITATION_ID = re.compile(r"S\s*(\d+)", re.IGNORECASE)


def _normalize_citation_format(answer: str) -> str:
    def replace_group(match: re.Match[str]) -> str:
        ids = list(dict.fromkeys(_CITATION_ID.findall(match.group(0))))
        return " ".join(f"[S{identifier}]" for identifier in ids)

    return _CITATION_GROUP.sub(replace_group, answer)


def _invalid_citation_ids(answer: str, evidence: list[EvidenceReference]) -> set[str]:
    cited = {
        f"S{identifier}"
        for group in _CITATION_GROUP.findall(answer)
        for identifier in _CITATION_ID.findall(group)
    }
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
        correction_ms: float = 0,
        provider: str | None = None,
        model: str | None = None,
        error: ChatError | None = None,
        needs_clarification: bool = False,
        clarification_question: str | None = None,
        abstention_reason: str | None = None,
    ) -> ChatResponse:
        return ChatResponse(
            answer=answer,
            language=language,
            evidence=evidence,
            citations=_citations(evidence),
            abstained=abstained,
            needs_clarification=needs_clarification,
            clarification_question=clarification_question,
            abstention_reason=abstention_reason,
            provider=provider,
            model=model,
            latency=LatencyInfo(
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
                correction_ms=correction_ms,
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
                normalize_financial_query(request.question),
                top_k=self.settings.retrieval_top_k,
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
        personalized_decision = _is_personalized_decision(request.question)
        prompt = build_grounded_prompt(
            question=request.question,
            language=request.language,
            evidence=evidence,
            personalized_decision=personalized_decision,
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
        normalized = _normalize_citation_format(generated.text)
        invalid = _invalid_citation_ids(normalized, evidence)
        has_citations = invalid != {"missing"}
        if invalid and (not generated.abstained or has_citations):
                correction_started = time.perf_counter()
                correction_prompt = (
                    "Correct the answer below using only the permitted citation "
                    "markers. Keep the answer in the selected language. Every "
                    "factual claim must use one or more markers from this exact "
                    f"set: {', '.join(item.citation_id for item in evidence)}. "
                    "If the evidence is insufficient, begin exactly with "
                    "INSUFFICIENT_EVIDENCE: and explain why without citations. "
                    "Do not invent or rename a source marker. Use only the "
                    "evidence supplied below.\n\n"
                    "Use each marker only where it supports a factual claim; "
                    "do not repeat identical marker groups.\n\n"
                    + (
                        "This is a personalized decision question. Do not give "
                        "a yes/no recommendation; explain the limitation and "
                        "ask at most one useful clarification question.\n\n"
                        if personalized_decision
                        else ""
                    )
                    + "RETRIEVED EVIDENCE:\n"
                    + "\n".join(
                        f"{item.citation_id}: {item.title} (page {item.page_number})\n"
                        f"{item.passage}"
                        for item in evidence
                    )
                    + "\n\n"
                    f"ANSWER TO CORRECT:\n{normalized}\n"
                )
                try:
                    corrected = await self.provider.generate(
                        ProviderRequest(
                            prompt=correction_prompt, language=request.language
                        )
                    )
                except (ProviderConfigurationError, ProviderRequestError):
                    corrected = None
                correction_ms = (time.perf_counter() - correction_started) * 1000
                if corrected is not None:
                    corrected_text = _normalize_citation_format(corrected.text)
                    corrected_invalid = _invalid_citation_ids(
                        corrected_text, evidence
                    )
                    corrected_has_citations = corrected_invalid != {"missing"}
                    if (
                        (corrected.abstained and not corrected_has_citations)
                        or not corrected_invalid
                    ):
                        return self._response(
                            started=started,
                            language=request.language,
                            answer=corrected_text,
                            evidence=evidence,
                            abstained=corrected.abstained or personalized_decision,
                            retrieval_ms=retrieval_ms,
                            generation_ms=generation_ms,
                            correction_ms=correction_ms,
                            provider=corrected.provider,
                            model=corrected.model,
                            needs_clarification=personalized_decision,
                            clarification_question=(
                                _localized("clarification", request.language)
                                if personalized_decision
                                else None
                            ),
                            abstention_reason=(
                                "personalized_decision"
                                if personalized_decision
                                else None
                            ),
                        )
                return self._response(
                    started=started,
                    language=request.language,
                    answer=_localized("insufficient", request.language),
                    evidence=evidence,
                    abstained=True,
                    retrieval_ms=retrieval_ms,
                    generation_ms=generation_ms,
                    correction_ms=correction_ms,
                    provider=generated.provider,
                    model=generated.model,
                    error=ChatError(
                        code="invalid_generated_citations",
                        message="The answer could not be safely linked to the retrieved sources",
                    ),
                    needs_clarification=personalized_decision,
                    clarification_question=(
                        _localized("clarification", request.language)
                        if personalized_decision
                        else None
                    ),
                    abstention_reason=(
                        "personalized_decision"
                        if personalized_decision
                        else "invalid_citations"
                    ),
                )
        return self._response(
            started=started,
            language=request.language,
            answer=_normalize_citation_format(generated.text),
            evidence=evidence,
            abstained=generated.abstained or personalized_decision,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            provider=generated.provider,
            model=generated.model,
            needs_clarification=personalized_decision,
            clarification_question=(
                _localized("clarification", request.language)
                if personalized_decision
                else None
            ),
            abstention_reason=(
                "personalized_decision"
                if personalized_decision
                else ("insufficient_evidence" if generated.abstained else None)
            ),
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
