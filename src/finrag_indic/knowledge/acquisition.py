import hashlib
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx
import pymupdf
from bs4 import BeautifulSoup

from finrag_indic.errors import AcquisitionError
from finrag_indic.knowledge.models import (
    ApprovalStatus,
    ExtractionStatus,
    SourceDocument,
)


OFFICIAL_DOMAINS = {
    "rbi.org.in",
    "www.rbi.org.in",
    "rbidocs.rbi.org.in",
    "www.npci.org.in",
    "npci.org.in",
    "www.sebi.gov.in",
    "sebi.gov.in",
    "investor.sebi.gov.in",
}


class SourceDownloader:
    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client

    def download(self, source: SourceDocument, raw_root: Path) -> SourceDocument:
        if source.approval_status != ApprovalStatus.APPROVED:
            raise AcquisitionError(
                f"source {source.document_id} is not explicitly approved for download"
            )
        host = (urlparse(str(source.official_source_url)).hostname or "").lower()
        if host not in OFFICIAL_DOMAINS:
            raise AcquisitionError(f"source host is not in the official allowlist: {host}")

        client = self._client or httpx.Client(
            follow_redirects=True,
            timeout=httpx.Timeout(60.0),
            headers={"User-Agent": "FinRAG-Indic/0.1 research-source-acquisition"},
        )
        should_close = self._client is None
        try:
            response = client.get(str(source.official_source_url))
            response.raise_for_status()
            content = response.content
        except httpx.HTTPStatusError as exc:
            failed_response = exc.response
            content_type = (
                failed_response.headers.get("content-type", "").split(";", 1)[0].lower()
                or None
            )
            raise AcquisitionError(
                f"failed to download {source.document_id}: {exc}",
                http_status=failed_response.status_code,
                content_type=content_type,
                resolved_source_url=str(failed_response.url),
            ) from exc
        except httpx.HTTPError as exc:
            raise AcquisitionError(f"failed to download {source.document_id}: {exc}") from exc
        finally:
            if should_close:
                client.close()

        resolved_host = (response.url.host or "").lower()
        if resolved_host not in OFFICIAL_DOMAINS:
            raise AcquisitionError(
                f"approved source redirected to a non-official host: {resolved_host}"
            )
        content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
        expected_pdf = str(source.official_source_url).lower().endswith(".pdf")
        if expected_pdf:
            if content_type != "application/pdf" or not content.startswith(b"%PDF-"):
                raise AcquisitionError(
                    f"{source.document_id} did not return a genuine PDF "
                    f"(content-type={content_type!r})"
                )
            try:
                with pymupdf.open(stream=content, filetype="pdf") as pdf:
                    if pdf.page_count < 1:
                        raise AcquisitionError(f"{source.document_id} PDF has no pages")
            except (RuntimeError, ValueError) as exc:
                raise AcquisitionError(
                    f"{source.document_id} PDF content could not be parsed: {exc}"
                ) from exc
            suffix = ".pdf"
        else:
            if content_type not in {"text/html", "application/xhtml+xml"}:
                raise AcquisitionError(
                    f"{source.document_id} did not return HTML "
                    f"(content-type={content_type!r})"
                )
            soup = BeautifulSoup(content, "html.parser")
            visible_text = soup.get_text(" ", strip=True)
            if len(visible_text) < 100:
                raise AcquisitionError(
                    f"{source.document_id} HTML has insufficient visible content"
                )
            suffix = ".html"
        target = raw_root / source.issuing_organization.lower() / f"{source.document_id}{suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

        return SourceDocument.model_validate(
            {
                **source.model_dump(),
                "retrieval_timestamp": datetime.now(UTC),
                "sha256": hashlib.sha256(content).hexdigest(),
                "local_file_path": target,
                "extraction_status": ExtractionStatus.ACQUIRED,
                "http_status": response.status_code,
                "content_type": content_type,
                "resolved_source_url": str(response.url),
                "extraction_error": None,
            }
        )

