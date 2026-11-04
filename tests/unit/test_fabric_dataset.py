from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from finrag_indic.datasets.fabric import (
    FabricReview,
    ReviewStatus,
    load_fabric,
)
from finrag_indic.errors import AcquisitionError


def write_parquet(path: Path, *, include_tamil: bool = True) -> None:
    data = {
        "id": ["q1"],
        "category": ["upi"],
        "subcategory": ["safety"],
        "difficulty": ["easy"],
        "question_en": ["What is a PIN?"],
        "question_hi": ["पिन क्या है?"],
        "answer": ["Unverified upstream answer"],
        "source": ["Upstream source description"],
    }
    if include_tamil:
        data["question_ta"] = ["PIN என்றால் என்ன?"]
    pq.write_table(pa.table(data), path)


def test_fabric_schema_preserves_provenance(tmp_path: Path) -> None:
    path = tmp_path / "fabric.parquet"
    write_parquet(path)

    records = load_fabric(path)

    assert records[0].question_id == "q1"
    assert records[0].question_ta == "PIN என்றால் என்ன?"
    assert records[0].original_reference_answer == "Unverified upstream answer"
    assert len(records[0].dataset_revision) == 40


def test_fabric_missing_language_column_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "fabric.parquet"
    write_parquet(path, include_tamil=False)

    with pytest.raises(AcquisitionError, match="missing columns"):
        load_fabric(path)


def test_empty_source_description_is_preserved_for_review(tmp_path: Path) -> None:
    path = tmp_path / "fabric.parquet"
    write_parquet(path)
    table = pq.read_table(path)
    table = table.set_column(table.schema.get_field_index("source"), "source", pa.array([""]))
    pq.write_table(table, path)

    record = load_fabric(path)[0]

    assert record.source_description == ""


def test_review_cannot_be_approved_while_unreviewed() -> None:
    with pytest.raises(ValidationError, match="all review fields"):
        FabricReview(
            question_id="q1",
            reviewer_approved=True,
            reviewer_id="reviewer-1",
            reviewed_at=datetime.now(UTC),
        )


def test_fully_verified_review_can_be_approved() -> None:
    review = FabricReview(
        question_id="q1",
        source_verification=ReviewStatus.VERIFIED,
        reference_answer_correctness=ReviewStatus.VERIFIED,
        hindi_translation_verification=ReviewStatus.VERIFIED,
        tamil_translation_verification=ReviewStatus.VERIFIED,
        regulatory_currency=ReviewStatus.VERIFIED,
        question_suitability=ReviewStatus.VERIFIED,
        reviewer_approved=True,
        reviewer_id="reviewer-1",
        reviewed_at=datetime.now(UTC),
    )

    assert review.reviewer_approved is True

