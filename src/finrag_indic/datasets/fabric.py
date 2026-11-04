import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

import httpx
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, model_validator

from finrag_indic.errors import AcquisitionError


FABRIC_DATASET_ID = "agenticclass/fabric"
FABRIC_REVISION = "d8476ea6a9081f7dec4852200b379bfd1213ad47"
FABRIC_FILE = "data/train-00000-of-00001.parquet"
FABRIC_URL = (
    "https://huggingface.co/datasets/agenticclass/fabric/resolve/"
    f"{FABRIC_REVISION}/{FABRIC_FILE}"
)
REQUIRED_COLUMNS = {
    "id",
    "category",
    "subcategory",
    "difficulty",
    "question_en",
    "question_hi",
    "question_ta",
    "answer",
    "source",
}


class ReviewStatus(StrEnum):
    UNREVIEWED = "unreviewed"
    VERIFIED = "verified"
    REJECTED = "rejected"
    NEEDS_REVIEW = "needs_review"


class FabricRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(min_length=1)
    question_en: str = Field(min_length=1)
    question_hi: str = Field(min_length=1)
    question_ta: str = Field(min_length=1)
    original_reference_answer: str = Field(min_length=1)
    category: str = Field(min_length=1)
    subcategory: str = Field(min_length=1)
    difficulty: str = Field(min_length=1)
    source_description: str
    dataset_id: str = FABRIC_DATASET_ID
    dataset_revision: str = FABRIC_REVISION
    provenance_url: str = FABRIC_URL


class FabricReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str
    source_verification: ReviewStatus = ReviewStatus.UNREVIEWED
    reference_answer_correctness: ReviewStatus = ReviewStatus.UNREVIEWED
    hindi_translation_verification: ReviewStatus = ReviewStatus.UNREVIEWED
    tamil_translation_verification: ReviewStatus = ReviewStatus.UNREVIEWED
    regulatory_currency: ReviewStatus = ReviewStatus.UNREVIEWED
    question_suitability: ReviewStatus = ReviewStatus.UNREVIEWED
    reviewer_approved: bool = False
    reviewer_id: str | None = None
    reviewed_at: datetime | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def approval_requires_complete_verification(self) -> "FabricReview":
        statuses = (
            self.source_verification,
            self.reference_answer_correctness,
            self.hindi_translation_verification,
            self.tamil_translation_verification,
            self.regulatory_currency,
            self.question_suitability,
        )
        if self.reviewer_approved:
            if not self.reviewer_id or not self.reviewed_at:
                raise ValueError("approved reviews require reviewer_id and reviewed_at")
            if any(status != ReviewStatus.VERIFIED for status in statuses):
                raise ValueError("approved reviews require all review fields to be verified")
        return self


class FabricSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str = FABRIC_DATASET_ID
    revision: str = FABRIC_REVISION
    source_url: str = FABRIC_URL
    retrieved_at: datetime
    file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    record_count: int = Field(gt=0)
    inspected_question_ids: list[str] = Field(default_factory=list)
    missing_source_description_count: int = Field(default=0, ge=0)
    verification_status: ReviewStatus = ReviewStatus.UNREVIEWED


def acquire_fabric(target: Path, client: httpx.Client | None = None) -> FabricSnapshot:
    own_client = client is None
    active_client = client or httpx.Client(follow_redirects=True, timeout=120.0)
    try:
        response = active_client.get(FABRIC_URL)
        response.raise_for_status()
        content = response.content
    except httpx.HTTPError as exc:
        raise AcquisitionError(f"failed to acquire FABRIC: {exc}") from exc
    finally:
        if own_client:
            active_client.close()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    records = load_fabric(target)
    return FabricSnapshot(
        retrieved_at=datetime.now(UTC),
        file_sha256=hashlib.sha256(content).hexdigest(),
        record_count=len(records),
    )


def load_fabric(path: Path) -> list[FabricRecord]:
    if not path.exists():
        raise AcquisitionError(f"FABRIC parquet file does not exist: {path}")
    try:
        table = pq.read_table(path)
    except Exception as exc:
        raise AcquisitionError(f"failed to read FABRIC parquet: {exc}") from exc
    missing = REQUIRED_COLUMNS - set(table.column_names)
    if missing:
        raise AcquisitionError(f"FABRIC schema is missing columns: {sorted(missing)}")
    records = [
        FabricRecord(
            question_id=row["id"],
            question_en=row["question_en"],
            question_hi=row["question_hi"],
            question_ta=row["question_ta"],
            original_reference_answer=row["answer"],
            category=row["category"],
            subcategory=row["subcategory"],
            difficulty=row["difficulty"],
            source_description=row["source"],
        )
        for row in table.to_pylist()
    ]
    identifiers = [record.question_id for record in records]
    if len(identifiers) != len(set(identifiers)):
        raise AcquisitionError("FABRIC contains duplicate question IDs")
    return records


def write_review_template(records: list[FabricRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            review = FabricReview(question_id=record.question_id)
            handle.write(review.model_dump_json() + "\n")


def save_snapshot(snapshot: FabricSnapshot, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(snapshot.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

