import hashlib
import re
import unicodedata
from collections import OrderedDict
from io import BytesIO
from pathlib import Path

import pymupdf
from bs4 import BeautifulSoup
from fontTools.cffLib import CFFFontSet

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


def glyph_name_to_unicode(glyph_name: str) -> str | None:
    """Decode Adobe-style ``uniXXXX``/``uXXXXX`` glyph names without guessing."""
    base = glyph_name.split(".", 1)[0]
    if base.startswith("uni"):
        payload = base[3:]
        if payload and len(payload) % 4 == 0 and re.fullmatch(r"[0-9A-Fa-f]+", payload):
            return "".join(chr(int(payload[index : index + 4], 16)) for index in range(0, len(payload), 4))
    if base.startswith("u") and re.fullmatch(r"[0-9A-Fa-f]{4,6}", base[1:]):
        return chr(int(base[1:], 16))
    return None


def repair_devanagari_visual_order(units: list[str]) -> str:
    """Move visually encoded pre-base i-matras behind their following glyph cluster."""
    repaired: list[str] = []
    index = 0
    while index < len(units):
        unit = units[index]
        if unit.startswith("\u093f") and index + 1 < len(units):
            repaired.append(units[index + 1] + unit)
            index += 2
        else:
            repaired.append(unit)
            index += 1
    return "".join(repaired)


def _cff_glyph_order(font_bytes: bytes) -> list[str]:
    cff = CFFFontSet()
    cff.decompile(BytesIO(font_bytes), None)
    return list(cff[cff.fontNames[0]].charset)


def _page_glyph_orders(document: pymupdf.Document, page: pymupdf.Page) -> dict[str, list[str]]:
    orders: dict[str, list[str]] = {}
    for font in page.get_fonts(full=True):
        xref, extension, _, embedded_name = font[:4]
        if extension != "cff":
            continue
        trace_name = embedded_name.split("+", 1)[-1]
        try:
            extracted = document.extract_font(xref, named=True)
            orders[trace_name] = _cff_glyph_order(extracted["content"])
        except (KeyError, OSError, ValueError):
            continue
    return orders


def _trace_character_map(page: pymupdf.Page) -> dict[tuple[str, float, float], list[tuple[int, int]]]:
    character_map: dict[tuple[str, float, float], list[tuple[int, int]]] = {}
    for span in page.get_texttrace():
        if span.get("type") != 0:
            continue
        font_name = str(span.get("font", ""))
        for unicode_value, glyph_id, origin, _ in span.get("chars", ()):
            key = (font_name, round(float(origin[0]), 3), round(float(origin[1]), 3))
            character_map.setdefault(key, []).append((unicode_value, glyph_id))
    return character_map


def _decode_raw_character(
    *,
    raw_character: dict[str, object],
    font_name: str,
    glyph_orders: dict[str, list[str]],
    trace_characters: dict[tuple[str, float, float], list[tuple[int, int]]],
) -> str:
    origin = raw_character["origin"]
    key = (font_name, round(float(origin[0]), 3), round(float(origin[1]), 3))
    traced = trace_characters.get(key, [])
    if not traced:
        return str(raw_character.get("c", ""))
    raw_text = str(raw_character.get("c", ""))
    matching = next(
        (
            item
            for item in traced
            if len(raw_text) == 1 and item[0] == ord(raw_text)
        ),
        None,
    )
    if matching is None and raw_text and ord(raw_text[0]) < 32:
        matching = next((item for item in traced if item[0] == 0xFFFD), None)
    unicode_value, glyph_id = matching or traced[0]
    if unicode_value != 0xFFFD:
        return chr(unicode_value)
    glyph_order = glyph_orders.get(font_name, [])
    if 0 <= glyph_id < len(glyph_order):
        decoded = glyph_name_to_unicode(glyph_order[glyph_id])
        if decoded is not None:
            return decoded
    return "\ufffd"


def extract_pdf_devanagari_glyphs(document: SourceDocument) -> list[ExtractedBlock]:
    """Extract visual-order Devanagari PDFs using their embedded CFF glyph names.

    Unknown glyphs remain U+FFFD so extraction quality remains auditable.
    """
    path = _require_source_file(document)
    try:
        with pymupdf.open(path) as pdf:
            blocks: list[ExtractedBlock] = []
            for page_index, page in enumerate(pdf):
                glyph_orders = _page_glyph_orders(pdf, page)
                trace_characters = _trace_character_map(page)
                lines: list[str] = []
                raw_page = page.get_text("rawdict", sort=True)
                for raw_block in raw_page["blocks"]:
                    if raw_block.get("type") != 0:
                        continue
                    for raw_line in raw_block.get("lines", []):
                        units: list[str] = []
                        for raw_span in raw_line.get("spans", []):
                            font_name = str(raw_span.get("font", ""))
                            for raw_character in raw_span.get("chars", []):
                                units.append(
                                    _decode_raw_character(
                                        raw_character=raw_character,
                                        font_name=font_name,
                                        glyph_orders=glyph_orders,
                                        trace_characters=trace_characters,
                                    )
                                )
                        lines.append(repair_devanagari_visual_order(units))
                blocks.extend(
                    _blocks_from_lines(
                        document=document,
                        lines=lines,
                        page_number=page_index + 1,
                    )
                )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ExtractionError(
            f"Devanagari glyph extraction failed for {document.document_id}: {exc}"
        ) from exc
    if not blocks:
        raise ExtractionError(f"PDF contains no extractable text: {document.document_id}")
    return blocks


def parse_rbi_table_rows(
    rows: list[list[str | None]], *, page_number: int, document: SourceDocument
) -> list[ExtractedBlock]:
    """Convert a detected RBI TAT table to one block per payment-system incident row."""
    if len(rows) < 3:
        return []
    description_column = next(
        (index for index, value in enumerate(rows[0]) if value and "Description" in value),
        2,
    )
    timeline_column = next(
        (index for index, value in enumerate(rows[1]) if value and "Timeline" in value),
        None,
    )
    compensation_column = next(
        (index for index, value in enumerate(rows[1]) if value and "Compensation" in value),
        None,
    )
    if timeline_column is None or compensation_column is None:
        return []

    system: str | None = None
    blocks: list[ExtractedBlock] = []
    for row in rows[3:]:
        serial = normalize_text(row[0] or "")
        if serial.isdigit():
            candidates = [normalize_text(value or "") for value in row[description_column:]]
            system = next((value for value in candidates if value), None)
            continue
        if serial not in {"a", "b", "c", "d"} or not system:
            continue
        incident = normalize_text(row[description_column] or "")
        timeline = normalize_text(row[timeline_column] or "")
        compensation = normalize_text(row[compensation_column] or "")
        if not incident:
            continue
        fields = [f"Payment system: {system}", f"Incident: {incident}"]
        if timeline:
            fields.append(f"Auto-reversal timeline: {timeline}")
        if compensation:
            fields.append(f"Compensation: {compensation}")
        blocks.append(
            ExtractedBlock(
                document_id=document.document_id,
                language=document.language,
                page_number=page_number,
                section_heading=f"{system} — row {serial}",
                text="\n".join(fields),
            )
        )
    return blocks


def extract_rbi_failed_transaction_tables(document: SourceDocument) -> list[ExtractedBlock]:
    """Extract the circular's detected tables while preserving logical row boundaries."""
    path = _require_source_file(document)
    try:
        with pymupdf.open(path) as pdf:
            blocks: list[ExtractedBlock] = []
            for page_index, page in enumerate(pdf):
                for table in page.find_tables().tables:
                    blocks.extend(
                        parse_rbi_table_rows(
                            table.extract(), page_number=page_index + 1, document=document
                        )
                    )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ExtractionError(f"RBI table extraction failed for {document.document_id}: {exc}") from exc
    if not blocks:
        raise ExtractionError(f"No RBI TAT table rows detected: {document.document_id}")
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
    preserve_section_boundaries: bool = False,
    extraction_method: str = "pymupdf_standard",
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
            block.section_heading
            if preserve_section_boundaries
            else (None if block.page_number is not None else block.section_heading),
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
                    extraction_method=extraction_method,
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
    if document.document_id == "rbi-failed-transactions-tat-2019":
        flags.append("currency_verification_inconclusive_check_amendments")
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

