from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


SupportedLanguage = Literal["en", "hi", "ta"]


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    service: str
    status: Literal["ok"]
    version: str
    environment: str


class ProviderRequest(BaseModel):
    """Grounded prompt passed to a replaceable text-generation provider."""

    model_config = ConfigDict(extra="forbid")

    prompt: str
    language: SupportedLanguage = "en"


class ProviderResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    provider: str
    model: str
    abstained: bool = False
    request_id: str | None = None


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2000)
    language: SupportedLanguage

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be blank")
        return stripped


class EvidenceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    citation_id: str
    rank: int = Field(ge=1)
    score: float
    chunk_id: str
    document_id: str
    title: str
    document_language: SupportedLanguage
    page_number: int | None = Field(default=None, ge=1)
    source_url: HttpUrl
    passage: str
    review_flags: list[str] = Field(default_factory=list)


class SourceCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    citation_id: str
    document_id: str
    title: str
    page_number: int | None = Field(default=None, ge=1)
    source_url: HttpUrl


class LatencyInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    retrieval_ms: float = Field(ge=0)
    generation_ms: float = Field(ge=0)
    correction_ms: float = Field(default=0, ge=0)
    total_ms: float = Field(ge=0)


class ChatError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    retryable: bool = False


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    language: SupportedLanguage
    evidence: list[EvidenceReference]
    citations: list[SourceCitation]
    abstained: bool
    needs_clarification: bool = False
    clarification_question: str | None = None
    abstention_reason: str | None = None
    provider: str | None = None
    model: str | None = None
    latency: LatencyInfo
    error: ChatError | None = None


class SystemStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ready", "degraded"]
    llm_provider: str
    llm_model: str | None
    llm_configured: bool
    retrieval_index: str
    retrieval_index_available: bool
    embedding_model: str
    supported_languages: list[SupportedLanguage]
