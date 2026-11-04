from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class Language(StrEnum):
    ENGLISH = "en"
    HINDI = "hi"
    TAMIL = "ta"


class ExtractionStatus(StrEnum):
    NOT_ACQUIRED = "not_acquired"
    ACQUIRED = "acquired"
    EXTRACTED = "extracted"
    FAILED = "failed"


class ApprovalStatus(StrEnum):
    CANDIDATE = "candidate"
    APPROVED = "approved"
    REJECTED = "rejected"


class TranslationKind(StrEnum):
    ORIGINAL = "original"
    OFFICIAL_TRANSLATION = "official_translation"
    MACHINE_TRANSLATION = "machine_translation"


class SourceDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]+$")
    title: str = Field(min_length=1)
    issuing_organization: Literal["RBI", "NPCI", "SEBI"]
    official_source_url: HttpUrl
    language: Language
    publication_date: date | None = None
    effective_date: date | None = None
    retrieval_timestamp: datetime | None = None
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    local_file_path: Path | None = None
    source_revision: str | None = None
    extraction_status: ExtractionStatus = ExtractionStatus.NOT_ACQUIRED
    http_status: int | None = Field(default=None, ge=100, le=599)
    content_type: str | None = None
    resolved_source_url: HttpUrl | None = None
    license_or_reuse_notes: str = Field(min_length=1)
    approval_status: ApprovalStatus = ApprovalStatus.CANDIDATE
    translation_kind: TranslationKind = TranslationKind.ORIGINAL
    coverage_group: str | None = None
    extraction_error: str | None = None

    @model_validator(mode="after")
    def acquired_source_has_integrity_metadata(self) -> "SourceDocument":
        if self.extraction_status != ExtractionStatus.NOT_ACQUIRED:
            if (
                not self.retrieval_timestamp
                or not self.sha256
                or not self.local_file_path
                or self.http_status is None
                or not self.content_type
                or self.resolved_source_url is None
            ):
                raise ValueError(
                    "acquired sources require retrieval_timestamp, SHA-256, local path, "
                    "HTTP status, content type, and resolved URL"
                )
        return self


class ExtractedBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    language: Language
    page_number: int | None = Field(default=None, ge=1)
    section_heading: str | None = None
    text: str = Field(min_length=1)


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    language: Language
    text: str = Field(min_length=1)
    page_number: int | None = Field(default=None, ge=1)
    section_heading: str | None = None
    chunk_index: int = Field(ge=0)
    token_count: int = Field(ge=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_url: HttpUrl
    title: str
    issuing_organization: str
    source_revision: str | None = None
    coverage_group: str | None = None
    review_flags: list[str] = Field(default_factory=list)


class RetrievalHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int = Field(ge=1)
    score: float
    chunk: Chunk

