from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from finrag_indic.errors import ManifestError
from finrag_indic.knowledge.manifest import SourceManifest, load_manifest, save_manifest
from finrag_indic.knowledge.models import (
    ExtractionStatus,
    Language,
    SourceDocument,
)


def source(document_id: str = "rbi-test") -> SourceDocument:
    return SourceDocument(
        document_id=document_id,
        title="Test source",
        issuing_organization="RBI",
        official_source_url=f"https://www.rbi.org.in/{document_id}.pdf",
        language=Language.ENGLISH,
        license_or_reuse_notes="Test fixture only.",
    )


def test_manifest_round_trip(tmp_path: Path) -> None:
    manifest = SourceManifest(sources=[source()])
    path = tmp_path / "sources.json"

    save_manifest(manifest, path)
    restored = load_manifest(path)

    assert restored == manifest
    assert restored.by_id("rbi-test").title == "Test source"


def test_manifest_rejects_duplicate_documents() -> None:
    with pytest.raises(ValidationError, match="duplicate document_id"):
        SourceManifest(sources=[source(), source()])


def test_acquired_source_requires_integrity_metadata() -> None:
    with pytest.raises(ValidationError, match="require retrieval_timestamp"):
        SourceDocument.model_validate(
            {
                **source().model_dump(),
                "extraction_status": ExtractionStatus.ACQUIRED,
            }
        )


def test_unknown_document_id_is_reported() -> None:
    with pytest.raises(ManifestError, match="unknown document_id"):
        SourceManifest().by_id("missing")


def test_complete_acquired_source_is_valid(tmp_path: Path) -> None:
    acquired = source().model_copy(
        update={
            "extraction_status": ExtractionStatus.ACQUIRED,
            "retrieval_timestamp": datetime.now(UTC),
            "sha256": "a" * 64,
            "local_file_path": tmp_path / "source.pdf",
            "http_status": 200,
            "content_type": "application/pdf",
            "resolved_source_url": "https://www.rbi.org.in/rbi-test.pdf",
        }
    )

    assert acquired.sha256 == "a" * 64

