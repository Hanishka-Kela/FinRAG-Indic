import hashlib
import re
import unicodedata
from collections import OrderedDict
from pathlib import Path

import pymupdf
from bs4 import BeautifulSoup

from finrag_indic.errors import ExtractionError
from finrag_indic.knowledge.models import (
    Chunk,
    ExtractedBlock,
    ExtractionStatus,
    SourceDocument,
)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize(
        "NFC", text.replace("\u00a0", " ").replace("\r\n", "\n").replace("\r", "\n")
    )
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 100 or stripped.endswith(('.', '?', '!', ';')):
        return False
    words = stripped.split()
    return len(words) <= 12 and (stripped.isupper() or stripped.istitle())


def _blocks_from_lines(
    *, document: SourceDocument, lines: list[str], page_number: int | None
) -> list[ExtractedBlock]:
    blocks: list[ExtractedBlock] = []
    section: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            text = normalize_text("\n".join(buffer))
            if text:
                blocks.append(
                    ExtractedBlock(
                        document_id=document.document_id,
                        language=document.language,
                        page_number=page_number,
                        section_heading=section,
                        text=text,
                    )
                )
            buffer.clear()

    for line in lines:
        cleaned = normalize_text(line)
        if not cleaned:
            flush()
        elif _looks_like_heading(cleaned):
            flush()
            section = cleaned
        else:
            buffer.append(cleaned)
    flush()
    return blocks


def extract_pdf(document: SourceDocument) -> list[ExtractedBlock]:
    path = _require_source_file(document)
    try:
        with pymupdf.open(path) as pdf:
            blocks: list[ExtractedBlock] = []
            for page_index, page in enumerate(pdf):
                text = page.get_text("text", sort=True)
                blocks.extend(
                    _blocks_from_lines(
                        document=document,
                        lines=text.splitlines(),
                        page_number=page_index + 1,
                    )
                )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ExtractionError(f"PDF extraction failed for {document.document_id}: {exc}") from exc
    if not blocks:
        raise ExtractionError(f"PDF contains no extractable text: {document.document_id}")
    return blocks


def extract_html(document: SourceDocument) -> list[ExtractedBlock]:
    path = _require_source_file(document)
    try:
        html = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ExtractionError(f"HTML read failed for {document.document_id}: {exc}") from exc
    soup = BeautifulSoup(html, "html.parser")
    for element in soup(["script", "style", "noscript", "nav", "footer"]):
        element.decompose()

    blocks: list[ExtractedBlock] = []
    section: str | None = None
    for element in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "dt", "dd"]):
        text = normalize_text(element.get_text(" ", strip=True))
        if not text:
            continue
        if element.name and element.name.startswith("h"):
            section = text
            continue
        blocks.append(
            ExtractedBlock(
                document_id=document.document_id,
                language=document.language,
                section_heading=section,
                text=text,
            )
        )
    if not blocks:
        raise ExtractionError(f"HTML contains no extractable content: {document.document_id}")
    return blocks


def extract_document(document: SourceDocument) -> list[ExtractedBlock]:
    path = _require_source_file(document)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(document)
    if suffix in {".html", ".htm"}:
        return extract_html(document)
    raise ExtractionError(f"unsupported source format: {suffix}")


def chunk_blocks(
    document: SourceDocument,
    blocks: list[ExtractedBlock],
    *,
    chunk_size: int,
    overlap: int,
    min_chunk_tokens: int = 1,
) -> list[Chunk]:
    if (
        chunk_size <= 0
        or overlap < 0
        or overlap >= chunk_size
        or min_chunk_tokens < 1
        or min_chunk_tokens > chunk_size
    ):
        raise ValueError("chunk_size must be positive and overlap smaller than chunk_size")

    grouped: OrderedDict[
        tuple[int | None, str | None], dict[str, object]
    ] = OrderedDict()
    for block in blocks:
        key = (
            block.page_number,
            None if block.page_number is not None else block.section_heading,
        )
        group = grouped.setdefault(
            key,
            {"texts": [], "text_hashes": set(), "headings": []},
        )
        text_hash = hashlib.sha256(block.text.encode("utf-8")).hexdigest()
        text_hashes = group["text_hashes"]
        if isinstance(text_hashes, set) and text_hash in text_hashes:
            continue
        if isinstance(text_hashes, set):
            text_hashes.add(text_hash)
        texts = group["texts"]
        if isinstance(texts, list):
            texts.append(block.text)
        headings = group["headings"]
        if (
            block.section_heading
            and isinstance(headings, list)
            and block.section_heading not in headings
        ):
            headings.append(block.section_heading)

    chunks: list[Chunk] = []
    seen_hashes: set[str] = set()
    step = chunk_size - overlap
    for (page_number, html_section), group in grouped.items():
        texts = group["texts"]
        headings = group["headings"]
        combined_text = "\n\n".join(texts) if isinstance(texts, list) else ""
        words = combined_text.split()
        section_heading = html_section
        if section_heading is None and isinstance(headings, list) and headings:
            section_heading = " | ".join(str(item) for item in headings[:5])
        for start in range(0, len(words), step):
            window = words[start : start + chunk_size]
            if len(window) < min_chunk_tokens:
                continue
            text = " ".join(window)
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest in seen_hashes:
                if start + chunk_size >= len(words):
                    break
                continue
            seen_hashes.add(digest)
            chunk_index = len(chunks)
            identity = (
                f"{document.document_id}|{page_number}|{section_heading}|"
                f"{chunk_index}|{digest}"
            )
            chunk_id = f"{document.document_id}-{hashlib.sha256(identity.encode()).hexdigest()[:16]}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    language=document.language,
                    text=text,
                    page_number=page_number,
                    section_heading=section_heading,
                    chunk_index=chunk_index,
                    token_count=len(window),
                    content_sha256=digest,
                    source_url=document.official_source_url,
                    title=document.title,
                    issuing_organization=document.issuing_organization,
                    source_revision=document.source_revision,
                    coverage_group=document.coverage_group,
                    review_flags=_currency_review_flags(document, text),
                )
            )
            if start + chunk_size >= len(words):
                break
    if not chunks:
        raise ExtractionError(f"no chunks produced for {document.document_id}")
    return chunks


def _currency_review_flags(document: SourceDocument, text: str) -> list[str]:
    """Flag date-sensitive content for review without deciding its legal currency."""
    normalized = text.casefold()
    flags: list[str] = []
    if document.issuing_organization == "RBI" and "ombudsman" in normalized:
        flags.append("potentially_superseded_ombudsman_process")
    if document.issuing_organization == "SEBI" and "scores" in normalized:
        flags.append("potentially_superseded_grievance_process")
    if "last amended" in normalized:
        flags.append("potentially_superseded_regulatory_reference")
    if "settlement cycle" in normalized:
        flags.append("date_sensitive_settlement_cycle")
    if "income tax" in normalized or "section 80" in normalized:
        flags.append("date_sensitive_tax_information")
    if "government schemes" in normalized or "small savings scheme" in normalized:
        flags.append("date_sensitive_scheme_information")
    return flags


def _require_source_file(document: SourceDocument) -> Path:
    if document.extraction_status == ExtractionStatus.NOT_ACQUIRED:
        raise ExtractionError(f"source has not been acquired: {document.document_id}")
    if document.local_file_path is None or not document.local_file_path.exists():
        raise ExtractionError(f"source file is missing: {document.local_file_path}")
    return document.local_file_path

