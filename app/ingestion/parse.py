"""Parse annual-report PDFs into page-level JSONL records."""

import argparse
import re
from pathlib import Path

import pymupdf

from app.ingestion.manifest import (
    ManifestDocument,
    get_manifest_document,
    get_manifest_document_ids,
)
from app.ingestion.schema import ExtractedPage, ExtractedTable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_REPORTS_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_REPORTS_DIR = PROJECT_ROOT / "data" / "processed"


def extract_page_text(page: pymupdf.Page) -> str:
    """Extract text in the PDF's native content-stream order."""

    return page.get_text("text", sort=False)


def _normalized_cells(
    cells: list[str | None],
) -> list[str]:
    """Normalize cells only for internal row comparisons."""

    return [
        cell.strip() if cell is not None else ""
        for cell in cells
    ]


def _split_table_header(
    headers: list[str | None],
    rows: list[list[str | None]],
) -> tuple[list[str | None], list[list[str | None]]]:
    """Separate a detected header from the table's data rows."""

    if not rows:
        return headers, rows

    first_row = rows[0]

    if _normalized_cells(first_row) == _normalized_cells(headers):
        return headers, rows[1:]

    populated_headers = sum(
        bool(value)
        for value in _normalized_cells(headers)
    )
    populated_first_row = sum(
        bool(value)
        for value in _normalized_cells(first_row)
    )

    if populated_headers <= 1 and populated_first_row > 1:
        return list(first_row), rows[1:]

    return headers, rows


def extract_page_tables(
    page: pymupdf.Page,
) -> list[ExtractedTable]:
    """Extract tables separately from the page's prose text."""

    found_tables = page.find_tables()
    extracted_tables: list[ExtractedTable] = []

    for table_index, table in enumerate(
        found_tables.tables,
        start=1,
    ):
        rows = table.extract()
        headers = (
            list(table.header.names)
            if table.header is not None
            else []
        )
        headers, rows = _split_table_header(headers, rows)

        if not headers and not rows:
            continue

        extracted_table = ExtractedTable(
            table_index=table_index,
            bbox=tuple(table.bbox),
            headers=headers,
            rows=rows,
        )

        extracted_tables.append(extracted_table)

    return extracted_tables


def detect_printed_page(
    page: pymupdf.Page,
    pdf_page: int,
) -> str | None:
    """Return footer page numbers or a non-default PDF page label."""

    footer_start = page.rect.height * 0.94
    left_edge = page.rect.width * 0.12
    right_edge = page.rect.width * 0.88
    footer_numbers: list[tuple[float, str]] = []

    for word in page.get_text("words", sort=False):
        x0, y0, x1, _, text = word[:5]

        if y0 < footer_start:
            continue

        if x0 > left_edge and x1 < right_edge:
            continue

        if re.fullmatch(r"\d{1,4}", text):
            footer_numbers.append((x0, text))

    ordered_numbers: list[str] = []

    for _, number in sorted(footer_numbers):
        if number not in ordered_numbers:
            ordered_numbers.append(number)

    if ordered_numbers:
        return "-".join(ordered_numbers)

    page_label = (page.get_label() or "").strip()

    if not page_label:
        return None

    if page_label == str(pdf_page):
        return None

    return page_label


def parse_pdf(
    pdf_path: Path,
    document_metadata: ManifestDocument,
) -> list[ExtractedPage]:
    """Parse every page of one annual-report PDF."""

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    parsed_pages: list[ExtractedPage] = []

    with pymupdf.open(pdf_path) as document:
        if not document.is_pdf:
            raise ValueError(f"Not a valid PDF: {pdf_path}")

        if document.needs_pass:
            raise ValueError(
                f"PDF requires a password: {pdf_path}"
            )

        for pdf_page, page in enumerate(document, start=1):
            parsed_page = ExtractedPage(
                doc_id=document_metadata.doc_id,
                company=document_metadata.company,
                fiscal_year=document_metadata.fiscal_year,
                pdf_page=pdf_page,
                printed_page=detect_printed_page(
                    page,
                    pdf_page,
                ),
                text=extract_page_text(page),
                tables=extract_page_tables(page),
                source_filename=document_metadata.filename,
            )

            parsed_pages.append(parsed_page)

    return parsed_pages


def save_pages(
    pages: list[ExtractedPage],
    output_path: Path,
) -> None:
    """Save parsed pages as one JSON object per line."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w", encoding="utf-8") as output_file:
        for page in pages:
            output_file.write(page.model_dump_json())
            output_file.write("\n")


def parse_document(doc_id: str) -> list[ExtractedPage]:
    """Parse and save one report selected by document ID."""

    metadata = get_manifest_document(doc_id)

    pdf_path = RAW_REPORTS_DIR / metadata.filename
    output_path = (
        PROCESSED_REPORTS_DIR
        / f"{doc_id}.pages.jsonl"
    )

    pages = parse_pdf(
        pdf_path=pdf_path,
        document_metadata=metadata,
    )

    if len(pages) != metadata.pdf_pages:
        raise ValueError(
            f"Expected {metadata.pdf_pages} pages, "
            f"parsed {len(pages)}."
        )

    save_pages(pages, output_path)

    table_count = sum(
        len(page.tables)
        for page in pages
    )

    print(f"Parsed document: {doc_id}")
    print(f"Company: {metadata.company}")
    print(f"Fiscal year: {metadata.fiscal_year}")
    print(f"Parsed pages: {len(pages)}")
    print(f"Extracted tables: {table_count}")
    print(f"Output: {output_path}")

    return pages


def main() -> None:
    """Parse one manifest report selected by document ID."""

    argument_parser = argparse.ArgumentParser(
        description=(
            "Parse an annual-report PDF into page JSONL."
        )
    )
    argument_parser.add_argument(
        "doc_id",
        choices=get_manifest_document_ids(),
        help="Document ID from data/manifest.json.",
    )
    arguments = argument_parser.parse_args()
    parse_document(arguments.doc_id)


if __name__ == "__main__":
    main()
