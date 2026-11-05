import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pymupdf
import pytest

from finrag_indic.errors import ExtractionError
from finrag_indic.knowledge.ingestion import (
    chunk_blocks,
    extract_html,
    extract_pdf,
    glyph_name_to_unicode,
    normalize_text,
    parse_rbi_table_rows,
    repair_devanagari_visual_order,
)
from finrag_indic.knowledge.models import (
    ExtractedBlock,
    ExtractionStatus,
    Language,
    SourceDocument,
)


def acquired_source(path: Path, language: Language = Language.ENGLISH) -> SourceDocument:
    content = path.read_bytes() if path.exists() else b"missing"
    return SourceDocument(
        document_id="rbi-fixture",
        title="RBI fixture",
        issuing_organization="RBI",
        official_source_url="https://www.rbi.org.in/fixture.pdf",
        language=language,
        retrieval_timestamp=datetime.now(UTC),
        sha256=hashlib.sha256(content).hexdigest(),
        local_file_path=path,
        extraction_status=ExtractionStatus.ACQUIRED,
        http_status=200,
        content_type="application/pdf" if path.suffix == ".pdf" else "text/html",
        resolved_source_url="https://www.rbi.org.in/fixture.pdf",
        license_or_reuse_notes="Test fixture only.",
    )


def test_unicode_normalization_preserves_indic_scripts() -> None:
    text = "हिन्दी   प्रश्न\n\n\nதமிழ் கேள்வி"

    normalized = normalize_text(text)

    assert normalized == "हिन्दी प्रश्न\n\nதமிழ் கேள்வி"


def test_devanagari_ligatures_are_decoded_and_visual_order_is_repaired() -> None:
    assert glyph_name_to_unicode("uni0924094D0924") == "त्त"
    units = ["ि", "व", "त्त", "ी", "य", " ", "ि", "श", "क्ष", "ा"]

    assert repair_devanagari_visual_order(units) == "वित्तीय शिक्षा"


def test_pdf_extraction_preserves_page_numbers(tmp_path: Path) -> None:
    path = tmp_path / "fixture.pdf"
    pdf = pymupdf.open()
    first = pdf.new_page()
    first.insert_text((72, 72), "BANKING SAFETY\nNever share a secret PIN.")
    second = pdf.new_page()
    second.insert_text((72, 72), "COMPLAINTS\nContact the regulated entity first.")
    pdf.save(path)
    pdf.close()

    blocks = extract_pdf(acquired_source(path))

    assert {block.page_number for block in blocks} == {1, 2}
    assert "secret PIN" in " ".join(block.text for block in blocks)


def test_html_extraction_preserves_unicode_and_sections(tmp_path: Path) -> None:
    path = tmp_path / "fixture.html"
    path.write_text(
        "<html><h2>सुरक्षा</h2><p>अपना पिन साझा न करें।</p>"
        "<h2>பாதுகாப்பு</h2><p>PIN பகிர வேண்டாம்.</p></html>",
        encoding="utf-8",
    )

    blocks = extract_html(acquired_source(path, Language.HINDI))

    assert blocks[0].section_heading == "सुरक्षा"
    assert "பகிர" in blocks[1].text


def test_chunking_preserves_metadata_and_removes_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "fixture.html"
    path.write_text("fixture", encoding="utf-8")
    document = acquired_source(path)
    duplicate = ExtractedBlock(
        document_id=document.document_id,
        language=document.language,
        page_number=1,
        section_heading="Safety",
        text="one two three four five",
    )

    chunks = chunk_blocks(document, [duplicate, duplicate], chunk_size=3, overlap=1)

    assert len(chunks) == 2
    assert chunks[0].document_id == document.document_id
    assert chunks[0].page_number == 1
    assert chunks[0].section_heading == "Safety"


def test_missing_source_file_is_reported(tmp_path: Path) -> None:
    document = acquired_source(tmp_path / "missing.pdf")

    with pytest.raises(ExtractionError, match="source file is missing"):
        extract_pdf(document)


def test_chunking_flags_date_sensitive_material_for_review(tmp_path: Path) -> None:
    path = tmp_path / "fixture.html"
    path.write_text("fixture", encoding="utf-8")
    document = acquired_source(path)
    block = ExtractedBlock(
        document_id=document.document_id,
        language=document.language,
        page_number=1,
        text="Contact the Banking Ombudsman after following the stated process.",
    )

    chunks = chunk_blocks(document, [block], chunk_size=20, overlap=2)

    assert chunks[0].review_flags == ["potentially_superseded_ombudsman_process"]


def test_rbi_table_rows_keep_upi_and_aadhaar_systems_separate(tmp_path: Path) -> None:
    path = tmp_path / "fixture.pdf"
    path.write_bytes(b"fixture")
    document = acquired_source(path)
    header = [None] * 8
    header[2] = "Description of the incident"
    subheader = [None] * 8
    subheader[4] = "Timeline for auto-reversal"
    subheader[6] = "Compensation payable"
    columns = [None] * 8
    rows = [
        header,
        subheader,
        columns,
        ["4", None, None, "Unified Payments Interface (UPI)", None, None, None, None],
        ["a", None, "Beneficiary not credited.", None, "T + 1 day.", None, "₹100/day.", None],
        ["5", None, None, "Aadhaar Enabled Payment System", None, None, None, None],
        ["a", None, "Merchant confirmation missing.", None, "T + 5 days.", None, "₹100/day.", None],
    ]

    blocks = parse_rbi_table_rows(rows, page_number=4, document=document)

    assert len(blocks) == 2
    assert "UPI" in blocks[0].text and "Aadhaar" not in blocks[0].text
    assert "Aadhaar" in blocks[1].text and "UPI" not in blocks[1].text
    assert all(block.page_number == 4 for block in blocks)

