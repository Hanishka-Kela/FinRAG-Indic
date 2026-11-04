from pathlib import Path
import hashlib

import httpx
import pytest

from finrag_indic.errors import AcquisitionError
from finrag_indic.knowledge.acquisition import SourceDownloader
from finrag_indic.knowledge.models import (
    ApprovalStatus,
    ExtractionStatus,
    Language,
    SourceDocument,
)


def source(approved: bool) -> SourceDocument:
    return SourceDocument(
        document_id="npci-test",
        title="NPCI test",
        issuing_organization="NPCI",
        official_source_url="https://www.npci.org.in/test",
        language=Language.ENGLISH,
        license_or_reuse_notes="Test fixture only.",
        approval_status=ApprovalStatus.APPROVED if approved else ApprovalStatus.CANDIDATE,
    )


def test_downloader_rejects_unapproved_source(tmp_path: Path) -> None:
    with pytest.raises(AcquisitionError, match="not explicitly approved"):
        SourceDownloader().download(source(False), tmp_path)


def test_downloader_preserves_hash_and_path(tmp_path: Path) -> None:
    content = (
        b"<html><body><p>Official fixture content for controlled acquisition "
        b"validation. This paragraph is deliberately long enough to represent "
        b"a genuine HTML content response.</p></body></html>"
    )
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            content=content,
            headers={"content-type": "text/html"},
            request=request,
        )
    )
    with httpx.Client(transport=transport) as client:
        acquired = SourceDownloader(client).download(source(True), tmp_path)

    assert acquired.extraction_status == ExtractionStatus.ACQUIRED
    assert acquired.sha256 == hashlib.sha256(content).hexdigest()
    assert acquired.local_file_path is not None
    assert acquired.local_file_path.read_bytes() == content


def test_downloader_preserves_http_failure_metadata(tmp_path: Path) -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            403,
            content=b"blocked",
            headers={"content-type": "text/html; charset=utf-8"},
            request=request,
        )
    )
    with httpx.Client(transport=transport) as client:
        with pytest.raises(AcquisitionError) as raised:
            SourceDownloader(client).download(source(True), tmp_path)

    assert raised.value.http_status == 403
    assert raised.value.content_type == "text/html"
    assert raised.value.resolved_source_url == "https://www.npci.org.in/test"

