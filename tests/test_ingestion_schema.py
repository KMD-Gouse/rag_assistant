"""Tests for ingestion data models."""

import pytest
from pydantic import ValidationError

from app.ingestion.schema import ExtractedPage, ExtractedTable


def test_extracted_page_accepts_complete_metadata() -> None:
    table = ExtractedTable(
        table_index=1,
        bbox=(50.0, 100.0, 550.0, 400.0),
        headers=["Metric", "FY2025"],
        rows=[
            ["Revenue", "1,62,990"],
            ["Employees", "3,23,578"],
        ],
    )

    page = ExtractedPage(
        doc_id="infosys_fy2025",
        company="Infosys",
        fiscal_year="2024-25",
        pdf_page=24,
        printed_page="46",
        text="Employees globally 3,23,578",
        tables=[table],
        source_filename="infosys_fy2025.pdf",
    )

    assert page.company == "Infosys"
    assert page.fiscal_year == "2024-25"
    assert page.pdf_page == 24
    assert page.tables[0].headers == ["Metric", "FY2025"]
    assert page.tables[0].rows[0] == ["Revenue", "1,62,990"]


def test_extracted_page_allows_empty_content() -> None:
    page = ExtractedPage(
        doc_id="infosys_fy2025",
        company="Infosys",
        fiscal_year="2024-25",
        pdf_page=1,
        text="",
        source_filename="infosys_fy2025.pdf",
    )

    assert page.text == ""
    assert page.tables == []
    assert page.printed_page is None


def test_extracted_table_allows_header_only_detection() -> None:
    table = ExtractedTable(
        table_index=1,
        bbox=(50.0, 100.0, 550.0, 130.0),
        headers=["Metric", "FY2025"],
    )

    assert table.rows == []


def test_extracted_page_rejects_zero_page_number() -> None:
    with pytest.raises(ValidationError):
        ExtractedPage(
            doc_id="infosys_fy2025",
            company="Infosys",
            fiscal_year="2024-25",
            pdf_page=0,
            text="Example",
            source_filename="infosys_fy2025.pdf",
        )


def test_extracted_page_rejects_invalid_fiscal_year() -> None:
    with pytest.raises(ValidationError):
        ExtractedPage(
            doc_id="infosys_fy2025",
            company="Infosys",
            fiscal_year="2025",
            pdf_page=1,
            text="Example",
            source_filename="infosys_fy2025.pdf",
        )


def test_extracted_page_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ExtractedPage(
            doc_id="infosys_fy2025",
            company="Infosys",
            fiscal_year="2024-25",
            pdf_page=1,
            text="Example",
            source_filename="infosys_fy2025.pdf",
            unexpected_field="not allowed",
        )
