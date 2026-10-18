"""Tests for retrieval chunk creation."""

from pathlib import Path

import pytest

from app.ingestion.chunk import (
    create_document_chunks,
    create_page_chunks,
    detect_section,
    estimate_tokens,
    load_pages,
    remove_page_boilerplate,
    split_prose,
    split_table,
)
from app.ingestion.schema import (
    ExtractedPage,
    ExtractedTable,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_REPORTS_DIR = PROJECT_ROOT / "data" / "processed"


def make_page(
    text: str,
    *,
    pdf_page: int = 1,
    printed_page: str | None = None,
    tables: list[ExtractedTable] | None = None,
) -> ExtractedPage:
    """Create a page with complete test metadata."""

    return ExtractedPage(
        doc_id="test_report",
        company="Test Company",
        fiscal_year="2024-25",
        pdf_page=pdf_page,
        printed_page=printed_page,
        text=text,
        tables=tables or [],
        source_filename="test_report.pdf",
    )


def test_split_prose_respects_maximum() -> None:
    sentence = (
        "Infosys delivers digital services to global "
        "customers while preserving useful context."
    )
    text = "\n\n".join([sentence] * 150)

    chunks = split_prose(text)

    assert len(chunks) > 1
    assert all(
        estimate_tokens(chunk) <= 500
        for chunk in chunks
    )
    assert all(chunk.strip() for chunk in chunks)


def test_split_table_repeats_headers() -> None:
    table = ExtractedTable(
        table_index=1,
        bbox=(50.0, 100.0, 550.0, 500.0),
        headers=[
            "In ₹ crore",
            "FY2025",
            "FY2024",
        ],
        rows=[
            [f"Metric {number}", str(number), str(number - 1)]
            for number in range(1, 41)
        ],
    )

    chunks = split_table(
        table,
        max_tokens=100,
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert "Headers: In ₹ crore" in chunk
        assert estimate_tokens(chunk) <= 100

    combined_text = "\n".join(chunks)

    for number in range(1, 41):
        assert f"Metric {number}" in combined_text


def test_boilerplate_is_removed() -> None:
    text = "\n".join(
        [
            "Infosys Integrated Annual Report 2024-25",
            "77",
            "Sl. No.",
        ]
    )

    cleaned_text = remove_page_boilerplate(
        text=text,
        printed_page="77",
    )

    assert cleaned_text == "Sl. No."

    page = make_page(
        text,
        printed_page="77",
    )

    assert create_document_chunks([page]) == []


@pytest.mark.parametrize(
    "text",
    [
        "Particulars\nRevenue data follows.",
        "Infosys Limited\nRevenue data follows.",
        "(In ₹ crore)\nRevenue data follows.",
        "Name of the\nRevenue data follows.",
        "Fiscal 2025\nRevenue data follows.",
        "Not applicable\nRevenue data follows.",
        "Profit /\nRevenue data follows.",
    ],
)
def test_generic_lines_are_not_sections(text: str) -> None:
    assert detect_section(text) is None


def test_real_heading_is_detected() -> None:
    assert detect_section("Human Capital\nEmployee data") == (
        "Human Capital"
    )


def test_repeated_report_footer_is_removed() -> None:
    cleaned_text = remove_page_boilerplate(
        text=(
            "Useful sustainability information.\n"
            "Business Responsibility and Sustainability Report"
        ),
        printed_page="206",
    )

    assert cleaned_text == "Useful sustainability information."


def test_decorative_spaced_letters_are_not_meaningful() -> None:
    page = make_page("A I Y O U R E N T E R P R I S E")

    assert create_document_chunks([page]) == []


def test_chunk_ids_and_metadata_are_deterministic() -> None:
    text = "\n".join(
        [
            "Human Capital",
            (
                "Employees globally 3,23,578. "
                "Infosys supports employee learning "
                "and workplace inclusivity."
            ),
        ]
    )

    page = make_page(
        text,
        pdf_page=24,
        printed_page="46-47",
    )

    first_result = create_document_chunks([page])
    second_result = create_document_chunks([page])

    assert first_result == second_result
    assert len(first_result) == 1

    chunk = first_result[0]

    assert chunk.chunk_id == (
        "test_report:p0024:prose:001"
    )
    assert chunk.doc_id == "test_report"
    assert chunk.company == "Test Company"
    assert chunk.fiscal_year == "2024-25"
    assert chunk.pdf_page == 24
    assert chunk.printed_page == "46-47"
    assert chunk.section == "Human Capital"
    assert chunk.content_type == "prose"
    assert chunk.table_index is None
    assert "Employees globally" in chunk.text
    assert "3,23,578" in chunk.text


def test_table_chunk_preserves_source_metadata() -> None:
    table = ExtractedTable(
        table_index=2,
        bbox=(50.0, 100.0, 550.0, 300.0),
        headers=["Metric", "FY2025"],
        rows=[["Revenue", "1,62,990"]],
    )
    page = make_page(
        "Business highlights and financial performance.",
        pdf_page=15,
        printed_page="28-29",
        tables=[table],
    )

    chunks = create_page_chunks(
        page=page,
        section="Business highlights",
    )

    table_chunks = [
        chunk
        for chunk in chunks
        if chunk.content_type == "table"
    ]

    assert len(table_chunks) == 1

    chunk = table_chunks[0]

    assert chunk.chunk_id == (
        "test_report:p0015:table:001"
    )
    assert chunk.table_index == 2
    assert chunk.pdf_page == 15
    assert chunk.printed_page == "28-29"
    assert "Headers: Metric | FY2025" in chunk.text
    assert "Revenue | 1,62,990" in chunk.text


@pytest.mark.integration
def test_infosys_fy2025_chunk_quality() -> None:
    pages_path = (
        PROCESSED_REPORTS_DIR
        / "infosys_fy2025.pages.jsonl"
    )

    if not pages_path.is_file():
        pytest.fail(
            "Run python -m app.ingestion.parse before "
            "the chunk integration test."
        )

    pages = load_pages(pages_path)

    first_result = create_document_chunks(pages)
    second_result = create_document_chunks(pages)

    assert first_result == second_result

    chunk_ids = [
        chunk.chunk_id
        for chunk in first_result
    ]

    assert len(chunk_ids) == len(set(chunk_ids))

    assert all(
        chunk.doc_id == "infosys_fy2025"
        for chunk in first_result
    )
    assert all(
        chunk.company == "Infosys"
        for chunk in first_result
    )
    assert all(
        chunk.fiscal_year == "2024-25"
        for chunk in first_result
    )
    assert all(
        1 <= chunk.pdf_page <= 369
        for chunk in first_result
    )
    assert all(
        1 <= chunk.estimated_tokens <= 500
        for chunk in first_result
    )

    page_15_tables = [
        chunk
        for chunk in first_result
        if (
            chunk.pdf_page == 15
            and chunk.content_type == "table"
        )
    ]

    assert page_15_tables
    assert any(
        "In ₹ crore" in chunk.text
        and "Revenues" in chunk.text
        and "1,62,990" in chunk.text
        for chunk in page_15_tables
    )
    assert any(
        "In US$ million" in chunk.text
        and "19,277" in chunk.text
        for chunk in page_15_tables
    )

    page_24_chunks = [
        chunk
        for chunk in first_result
        if chunk.pdf_page == 24
    ]

    assert any(
        "Employees globally" in chunk.text
        and "3,23,578" in chunk.text
        for chunk in page_24_chunks
    )
    assert all(
        chunk.section == "Human Capital"
        for chunk in page_24_chunks
    )
