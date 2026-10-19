"""Tests for the annual-report PDF parser."""

import json
from pathlib import Path

import pymupdf
import pytest

from app.ingestion.manifest import (
    ManifestDocument,
    get_manifest_document,
)
from app.ingestion.parse import (
    RAW_REPORTS_DIR,
    detect_printed_page,
    extract_page_tables,
    parse_pdf,
    save_pages,
)
from app.ingestion.schema import ExtractedPage


TEST_METADATA = ManifestDocument(
    doc_id="test_report",
    company="Test Company",
    fiscal_year="2024-25",
    fiscal_year_end="2025-03-31",
    title="Test Annual Report",
    filename="test_report.pdf",
    source_url="https://example.com/test_report.pdf",
    sha256="a" * 64,
    downloaded_at="2026-10-19T00:00:00+00:00",
    pdf_pages=2,
)


def create_test_pdf(pdf_path: Path) -> None:
    """Create a deterministic two-page PDF with one empty page."""

    with pymupdf.open() as document:
        first_page = document.new_page()
        first_page.insert_text((72, 72), "First page text")
        first_page.insert_text(
            (40, first_page.rect.height - 20),
            "10",
        )

        document.new_page()
        document.save(str(pdf_path))


def create_table_pdf(pdf_path: Path) -> None:
    """Create a small ruled table with headers and two data rows."""

    with pymupdf.open() as document:
        page = document.new_page()

        for x in (50, 180, 280, 380):
            page.draw_line((x, 100), (x, 190))

        for y in (100, 130, 160, 190):
            page.draw_line((50, y), (380, y))

        values = [
            (60, 120, "Metric"),
            (190, 120, "FY2025"),
            (290, 120, "FY2024"),
            (60, 150, "Revenue"),
            (190, 150, "100"),
            (290, 150, "90"),
            (60, 180, "Profit"),
            (190, 180, "20"),
            (290, 180, "18"),
        ]

        for x, y, value in values:
            page.insert_text((x, y), value)

        document.save(str(pdf_path))


def test_parse_pdf_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        parse_pdf(
            pdf_path=tmp_path / "missing.pdf",
            document_metadata=TEST_METADATA,
        )


def test_parse_pdf_preserves_pages_metadata_and_empty_page(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / TEST_METADATA.filename
    create_test_pdf(pdf_path)

    pages = parse_pdf(pdf_path, TEST_METADATA)

    assert len(pages) == 2
    assert [page.pdf_page for page in pages] == [1, 2]
    assert pages[0].printed_page == "10"
    assert "First page text" in pages[0].text
    assert pages[1].text == ""
    assert pages[1].tables == []

    for page in pages:
        assert page.doc_id == "test_report"
        assert page.company == "Test Company"
        assert page.fiscal_year == "2024-25"
        assert page.source_filename == "test_report.pdf"


def test_extract_page_tables_preserves_headers_and_rows(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "table.pdf"
    create_table_pdf(pdf_path)

    with pymupdf.open(pdf_path) as document:
        tables = extract_page_tables(document[0])

    assert len(tables) == 1
    assert tables[0].headers == ["Metric", "FY2025", "FY2024"]
    assert tables[0].rows == [
        ["Revenue", "100", "90"],
        ["Profit", "20", "18"],
    ]


def test_parse_pdf_is_stable(tmp_path: Path) -> None:
    pdf_path = tmp_path / TEST_METADATA.filename
    create_test_pdf(pdf_path)

    first_result = parse_pdf(pdf_path, TEST_METADATA)
    second_result = parse_pdf(pdf_path, TEST_METADATA)

    assert first_result == second_result


def test_save_pages_writes_valid_jsonl(tmp_path: Path) -> None:
    page = ExtractedPage(
        doc_id="test_report",
        company="Test Company",
        fiscal_year="2024-25",
        pdf_page=1,
        text="Example",
        source_filename="test_report.pdf",
    )
    output_path = tmp_path / "processed" / "pages.jsonl"

    save_pages([page], output_path)

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["doc_id"] == "test_report"


@pytest.mark.integration
def test_infosys_fy2025_parser() -> None:
    metadata = get_manifest_document("infosys_fy2025")
    pdf_path = RAW_REPORTS_DIR / metadata.filename

    pages = parse_pdf(pdf_path, metadata)

    assert len(pages) == 369
    assert pages[0].pdf_page == 1
    assert pages[-1].pdf_page == 369
    assert pages[14].printed_page == "28-29"
    assert pages[23].printed_page == "46-47"
    assert "3,23,578" in pages[23].text
    assert pages[16].text.index(
        "Financial year 2025"
    ) < pages[16].text.index(
        "During the year, Infosys"
    )

    first_financial_table = pages[14].tables[0]
    assert first_financial_table.headers[0] == (
        "In ₹ crore, except per equity share data"
    )
    assert first_financial_table.rows[0][0] == "Revenues*"

    page_15_content = json.dumps(
        [table.model_dump() for table in pages[14].tables],
        ensure_ascii=False,
    )
    assert "1,62,990" in page_15_content
