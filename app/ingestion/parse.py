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

# TCS FY2024-25 embeds a subset font whose ToUnicode map assigns decimal
# digits to the contiguous Greek code points U+03EC..U+03F5. Other symbols
# below are likewise deterministic font-map substitutions verified against
# rendered pages. Keep this character-level: never infer or rewrite values.
PDF_CHARACTER_TRANSLATION = str.maketrans(
    {
        **{chr(0x03EC + digit): str(digit) for digit in range(10)},
        "༤": "₹",
        "й": "%",
        "н": "+",
        "͘": ".",
        "͗": ":",
        "ͻ": "•",
    }
)


def repair_extracted_text(text: str) -> str:
    """Repair known PDF font mappings and split capitalized word fragments."""

    repaired = text.translate(PDF_CHARACTER_TRANSLATION)
    # Some text runs split a capitalized word after its first letter, e.g.
    # ``M\nanaging`` and ``W\ne``. Joining only an uppercase one-letter
    # fragment avoids merging normal prose line wraps.
    return re.sub(r"(?<!\w)([A-Z])\s*\n\s*([a-z])", r"\1\2", repaired)


def repair_table_cell(text: str) -> str:
    """Repair a table cell and collapse spacing inside numeric-only cells."""

    repaired = repair_extracted_text(text)
    # A few TCS BRSR table headers use decorative / subset glyphs without a
    # usable Unicode mapping. These replacements are constrained by their
    # surrounding labels and were checked against the rendered PDF pages.
    repaired = repaired.replace("FY 2�2��", "FY 2025*")
    repaired = repaired.replace("FY 2�2�", "FY 2025")
    if "water discharged75� Parameter Water discharge" in repaired:
        repaired = repaired[repaired.index("Parameter Water discharge") :]
        repaired = repaired.replace("�in kilolitres (KL)�", "[in kilolitres (KL)]")
    if repaired.startswith("arameter ater withdrawal by source"):
        repaired = "Parameter Water withdrawal by source" + repaired.removeprefix(
            "arameter ater withdrawal by source"
        )
    stripped = repaired.strip()
    if re.fullmatch(r"[₹`$€£\d\s,.:;%()+\-/]+", stripped):
        return re.sub(r"(?<=\d)\s+(?=[\d.,:%])|(?<=[.,:])\s+(?=\d)", "", repaired)
    return repaired


def extract_page_text(page: pymupdf.Page) -> str:
    """Extract native-order text without PDF-generated intra-word spaces."""

    flags = pymupdf.TEXTFLAGS_TEXT | pymupdf.TEXT_INHIBIT_SPACES
    return repair_extracted_text(page.get_text("text", sort=False, flags=flags))


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
    flags = pymupdf.TEXTFLAGS_TEXT | pymupdf.TEXT_INHIBIT_SPACES
    page_words = page.get_text("words", flags=flags, sort=False) if found_tables.tables else []

    def text_in_cell(cell_rect: tuple[float, float, float, float] | None) -> str:
        if cell_rect is None:
            return ""
        rectangle = pymupdf.Rect(cell_rect)
        return " ".join(
            word[4]
            for word in page_words
            if rectangle.contains(
                pymupdf.Point((word[0] + word[2]) / 2, (word[1] + word[3]) / 2)
            )
        )

    for table_index, table in enumerate(
        found_tables.tables,
        start=1,
    ):
        extracted_rows = table.extract()
        geometry_rows = list(table.rows)
        raw_headers = list(table.header.names) if table.header is not None else []
        # Remove a duplicated header before geometry reconstruction. Adjacent
        # header cells can overlap slightly, which otherwise pulls a neighbor's
        # text into the reconstructed value and defeats equality detection.
        if (
            extracted_rows
            and raw_headers
            and _normalized_cells(extracted_rows[0]) == _normalized_cells(raw_headers)
        ):
            extracted_rows = extracted_rows[1:]
            geometry_rows = geometry_rows[1:]
        rows = []
        for extracted_row, geometry_row in zip(extracted_rows, geometry_rows, strict=True):
            cells: list[str | None] = []
            for extracted_cell, cell_rect in zip(extracted_row, geometry_row.cells, strict=True):
                if extracted_cell is None:
                    cells.append(None)
                    continue
                # ``table.extract()`` uses a separate text path that can add
                # spaces inside words and clip a final digit at cell borders.
                # Reconstruct from the page's clean word stream, retaining the
                # extractor value only when a cell contains no mapped words.
                direct = text_in_cell(cell_rect)
                for leading_symbol in ("%", "₹", "$", "€", "£"):
                    if (
                        extracted_cell.lstrip().startswith(leading_symbol)
                        and not direct.lstrip().startswith(leading_symbol)
                    ):
                        direct = f"{leading_symbol} {direct}"
                repaired = repair_table_cell(direct or extracted_cell)
                cells.append(repaired)
            rows.append(cells)
        headers = (
            [
                repair_table_cell(header) if header is not None else None
                for header in raw_headers
            ]
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
