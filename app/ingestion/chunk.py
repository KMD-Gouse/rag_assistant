import math
import re
from pathlib import Path
from typing import Literal

from app.ingestion.schema import (
    DocumentChunk,
    ExtractedPage,
    ExtractedTable,
)

REPORT_TITLE_PATTERN = re.compile(
    r"^(?:(?:infosys\s+)?integrated annual report \d{4}-\d{2}"
    r"|business responsibility and sustainability report)$",
    re.IGNORECASE,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_REPORTS_DIR = PROJECT_ROOT / "data" / "processed"

TARGET_DOC_ID = "infosys_fy2025"

MIN_MEANINGFUL_CHARACTERS = 20

BOILERPLATE_PATTERNS = [
    re.compile(
        r"^infosys integrated annual report",
        re.IGNORECASE,
    ),
    re.compile(r"^\d+(?:-\d+)?$"),
    re.compile(r"^www\.", re.IGNORECASE),
]

INCOMPLETE_HEADING_ENDINGS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}

GENERIC_SECTION_PATTERNS = [
    re.compile(r"^(?:table of )?contents$", re.IGNORECASE),
    re.compile(
        r"^(?:infosys )?integrated annual report(?:\s+\d{4}(?:-\d{2})?)?$",
        re.IGNORECASE,
    ),
    re.compile(r"^page\s+\d+(?:\s+of\s+\d+)?$", re.IGNORECASE),
    re.compile(r"^name(?:\s+of\b.*)?$", re.IGNORECASE),
    re.compile(r"^particulars$", re.IGNORECASE),
    re.compile(r"^year ended\b", re.IGNORECASE),
    re.compile(r"^fiscal\s+\d", re.IGNORECASE),
    re.compile(r"^parameter$", re.IGNORECASE),
    re.compile(r"^infosys limited$", re.IGNORECASE),
    re.compile(r"^\(in .+\)$", re.IGNORECASE),
    re.compile(r"^sl\.?\s*no\.?$", re.IGNORECASE),
    re.compile(r"^not applicable$", re.IGNORECASE),
]


MIN_CHUNK_TOKENS = 300
MAX_CHUNK_TOKENS = 500

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)

SENTENCE_BOUNDARY = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z0-9₹])"
)


def estimate_tokens(text: str) -> int:
    """Estimate the number of BGE tokenizer tokens."""

    basic_tokens = TOKEN_PATTERN.findall(text)

    return math.ceil(len(basic_tokens) * 1.2)


def normalize_paragraphs(text: str) -> list[str]:
    """Normalize PDF line wrapping while retaining paragraphs."""

    raw_paragraphs = re.split(r"\n\s*\n", text)
    normalized_paragraphs: list[str] = []

    for raw_paragraph in raw_paragraphs:
        # Repair words hyphenated across PDF line boundaries.
        paragraph = re.sub(
            r"(?<=\w)-\s*\n\s*(?=\w)",
            "-",
            raw_paragraph,
        )

        # A remaining newline is normally a visual line wrap.
        paragraph = re.sub(
            r"\s*\n\s*",
            " ",
            paragraph,
        )

        paragraph = re.sub(
            r"[ \t]+",
            " ",
            paragraph,
        ).strip()

        if paragraph:
            normalized_paragraphs.append(paragraph)

    return normalized_paragraphs


def split_by_words(
    text: str,
    max_tokens: int,
) -> list[str]:
    """Split text that has no usable sentence boundaries."""

    words = text.split()
    pieces: list[str] = []
    current_words: list[str] = []

    for word in words:
        candidate = " ".join(current_words + [word])

        if (
            current_words
            and estimate_tokens(candidate) > max_tokens
        ):
            pieces.append(" ".join(current_words))
            current_words = [word]
        else:
            current_words.append(word)

    if current_words:
        pieces.append(" ".join(current_words))

    return pieces


def split_oversized_paragraph(
    paragraph: str,
    max_tokens: int,
) -> list[str]:
    """Split a large paragraph at sentence boundaries."""

    sentences = SENTENCE_BOUNDARY.split(paragraph)

    pieces: list[str] = []
    current_sentences: list[str] = []

    for sentence in sentences:
        if estimate_tokens(sentence) > max_tokens:
            if current_sentences:
                pieces.append(" ".join(current_sentences))
                current_sentences = []

            pieces.extend(
                split_by_words(sentence, max_tokens)
            )
            continue

        candidate = " ".join(
            current_sentences + [sentence]
        )

        if (
            current_sentences
            and estimate_tokens(candidate) > max_tokens
        ):
            pieces.append(" ".join(current_sentences))
            current_sentences = [sentence]
        else:
            current_sentences.append(sentence)

    if current_sentences:
        pieces.append(" ".join(current_sentences))

    return pieces


def split_prose(
    text: str,
    min_tokens: int = MIN_CHUNK_TOKENS,
    max_tokens: int = MAX_CHUNK_TOKENS,
) -> list[str]:
    """Split prose into paragraph-aware retrieval chunks."""

    if min_tokens < 1:
        raise ValueError("min_tokens must be at least 1.")

    if max_tokens < min_tokens:
        raise ValueError(
            "max_tokens must be greater than or equal to "
            "min_tokens."
        )

    paragraphs = normalize_paragraphs(text)
    units: list[str] = []

    for paragraph in paragraphs:
        if estimate_tokens(paragraph) <= max_tokens:
            units.append(paragraph)
        else:
            units.extend(
                split_oversized_paragraph(
                    paragraph,
                    max_tokens,
                )
            )

    chunks: list[str] = []
    current_units: list[str] = []

    for unit in units:
        candidate = "\n\n".join(
            current_units + [unit]
        )

        if (
            current_units
            and estimate_tokens(candidate) > max_tokens
        ):
            chunks.append("\n\n".join(current_units))
            current_units = [unit]
        else:
            current_units.append(unit)

    if current_units:
        chunks.append("\n\n".join(current_units))

    # Merge a small final chunk when it still fits.
    if len(chunks) >= 2:
        combined_tail = "\n\n".join(chunks[-2:])

        if (
            estimate_tokens(chunks[-1]) < min_tokens
            and estimate_tokens(combined_tail)
            <= max_tokens
        ):
            chunks[-2:] = [combined_tail]

    return chunks

def normalize_table_cell(cell: str | None) -> str:
    """Convert one table cell into compact readable text."""

    if cell is None:
        return ""

    return re.sub(r"\s+", " ", cell).strip()


def format_table_header(table: ExtractedTable) -> str:
    """Format table identity, headers, and units."""

    lines = [f"Table {table.table_index}"]

    headers = [
        normalize_table_cell(header)
        for header in table.headers
    ]

    if any(headers):
        lines.append(
            "Headers: " + " | ".join(headers)
        )

    return "\n".join(lines)


def format_table_row(
    row_number: int,
    row: list[str | None],
) -> str:
    """Format one table row without losing empty cells."""

    cells = [
        normalize_table_cell(cell)
        for cell in row
    ]

    return (
        f"Row {row_number}: "
        + " | ".join(cells)
    )


def split_oversized_table_row(
    header: str,
    row_number: int,
    row: list[str | None],
    max_tokens: int,
) -> list[str]:
    """Split one unusually large row as a last resort."""

    row_text = " | ".join(
        normalize_table_cell(cell)
        for cell in row
    )

    available_tokens = (
        max_tokens
        - estimate_tokens(header)
        - 15
    )

    if available_tokens < 1:
        raise ValueError(
            "Table header is too large to fit in a chunk."
        )

    row_parts = split_by_words(
        row_text,
        available_tokens,
    )

    chunks: list[str] = []

    for part_number, part in enumerate(
        row_parts,
        start=1,
    ):
        chunk = (
            f"{header}\n"
            f"Row {row_number} "
            f"(part {part_number}/{len(row_parts)}): "
            f"{part}"
        )

        if estimate_tokens(chunk) > max_tokens:
            raise ValueError(
                "Unable to split oversized table row "
                f"{row_number} within {max_tokens} tokens."
            )

        chunks.append(chunk)

    return chunks


def split_table(
    table: ExtractedTable,
    max_tokens: int = MAX_CHUNK_TOKENS,
) -> list[str]:
    """Split a table by rows while repeating its headers."""

    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1.")

    # A detected header without rows is not a useful table chunk.
    # Its visible content is still present in the page's prose.
    if not table.rows:
        return []

    header = format_table_header(table)

    if estimate_tokens(header) >= max_tokens:
        raise ValueError(
            "Table header is too large to fit in a chunk."
        )

    chunks: list[str] = []
    current_rows: list[str] = []

    for row_number, row in enumerate(
        table.rows,
        start=1,
    ):
        formatted_row = format_table_row(
            row_number,
            row,
        )

        candidate = "\n".join(
            [header, *current_rows, formatted_row]
        )

        if estimate_tokens(candidate) <= max_tokens:
            current_rows.append(formatted_row)
            continue

        if current_rows:
            chunks.append(
                "\n".join([header, *current_rows])
            )
            current_rows = []

        single_row_chunk = "\n".join(
            [header, formatted_row]
        )

        if (
            estimate_tokens(single_row_chunk)
            <= max_tokens
        ):
            current_rows.append(formatted_row)
        else:
            chunks.extend(
                split_oversized_table_row(
                    header=header,
                    row_number=row_number,
                    row=row,
                    max_tokens=max_tokens,
                )
            )

    if current_rows:
        chunks.append(
            "\n".join([header, *current_rows])
        )

    return chunks

def is_meaningful_text(text: str) -> bool:
    """Return whether text is useful enough to index."""

    stripped_text = text.strip()

    if len(stripped_text) < MIN_MEANINGFUL_CHARACTERS:
        return False

    words = re.findall(r"\w+", stripped_text)

    if len(words) < 3:
        return False

    single_character_words = sum(
        len(word) == 1
        for word in words
    )

    return single_character_words / len(words) < 0.8

def remove_page_boilerplate(
    text: str,
    printed_page: str | None,
) -> str:
    """Remove repeated report titles and footer page numbers."""

    printed_numbers: set[str] = set()

    if printed_page is not None:
        printed_numbers.update(
            printed_page.split("-")
        )

    retained_lines: list[str] = []

    for raw_line in text.splitlines():
        normalized_line = re.sub(
            r"\s+",
            " ",
            raw_line,
        ).strip()

        if REPORT_TITLE_PATTERN.fullmatch(
            normalized_line
        ):
            continue

        if normalized_line in printed_numbers:
            continue

        retained_lines.append(raw_line)

    return "\n".join(retained_lines).strip()

def detect_section(text: str) -> str | None:
    """Find a conservative section heading near the page start."""

    inspected_lines = 0

    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()

        if not line:
            continue

        inspected_lines += 1

        if inspected_lines > 12:
            break

        if len(line) > 80:
            continue

        words = line.split()

        if len(words) < 2 or len(words) > 10:
            continue

        if line.endswith((".", "?", "!", ";", ",")):
            continue

        if re.search(r"[/\-–—]$", line):
            continue

        if words[-1].lower() in INCOMPLETE_HEADING_ENDINGS:
            continue

        if any(
            pattern.search(line)
            for pattern in GENERIC_SECTION_PATTERNS
        ):
            continue

        first_letter = next(
            (
                character
                for character in line
                if character.isalpha()
            ),
            None,
        )

        if first_letter is None:
            continue

        if not first_letter.isupper():
            continue

        return line

    return None


def make_chunk_id(
    doc_id: str,
    pdf_page: int,
    content_type: Literal["prose", "table"],
    sequence_number: int,
) -> str:
    """Create a deterministic chunk identifier."""

    return (
        f"{doc_id}:"
        f"p{pdf_page:04d}:"
        f"{content_type}:"
        f"{sequence_number:03d}"
    )


def create_page_chunks(
    page: ExtractedPage,
    section: str | None,
    prose_text: str | None = None,
) -> list[DocumentChunk]:
    """Create prose and table chunks from one page."""

    chunks: list[DocumentChunk] = []

    if prose_text is None:
        prose_text = remove_page_boilerplate(
            text=page.text,
            printed_page=page.printed_page,
        )

    prose_sequence = 0

    for prose_chunk_text in split_prose(prose_text):
        if not is_meaningful_text(prose_chunk_text):
            continue

        prose_sequence += 1

        chunk = DocumentChunk(
            chunk_id=make_chunk_id(
                doc_id=page.doc_id,
                pdf_page=page.pdf_page,
                content_type="prose",
                sequence_number=prose_sequence,
            ),
            doc_id=page.doc_id,
            company=page.company,
            fiscal_year=page.fiscal_year,
            pdf_page=page.pdf_page,
            printed_page=page.printed_page,
            section=section,
            content_type="prose",
            text=prose_chunk_text,
            estimated_tokens=estimate_tokens(
                prose_chunk_text
            ),
            source_filename=page.source_filename,
        )

        chunks.append(chunk)

    table_sequence = 0

    for table in page.tables:
        for table_text in split_table(table):
            if not is_meaningful_text(table_text):
                continue

            table_sequence += 1

            chunk = DocumentChunk(
                chunk_id=make_chunk_id(
                    doc_id=page.doc_id,
                    pdf_page=page.pdf_page,
                    content_type="table",
                    sequence_number=table_sequence,
                ),
                doc_id=page.doc_id,
                company=page.company,
                fiscal_year=page.fiscal_year,
                pdf_page=page.pdf_page,
                printed_page=page.printed_page,
                section=section,
                content_type="table",
                table_index=table.table_index,
                text=table_text,
                estimated_tokens=estimate_tokens(
                    table_text
                ),
                source_filename=page.source_filename,
            )

            chunks.append(chunk)

    return chunks


def create_document_chunks(
    pages: list[ExtractedPage],
) -> list[DocumentChunk]:
    """Create deterministic chunks for an entire document."""

    chunks: list[DocumentChunk] = []
    current_section: str | None = None

    for page in pages:
        cleaned_text = remove_page_boilerplate(
            text=page.text,
            printed_page=page.printed_page,
        )

        detected_section = detect_section(cleaned_text)

        if detected_section is not None:
            current_section = detected_section

        chunks.extend(
            create_page_chunks(
                page=page,
                section=current_section,
                prose_text=cleaned_text,
            )
        )

    return chunks


def load_pages(
    input_path: Path,
) -> list[ExtractedPage]:
    """Load and validate page records from JSONL."""

    pages: list[ExtractedPage] = []

    with input_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(
            input_file,
            start=1,
        ):
            if not line.strip():
                raise ValueError(
                    "Blank page record at "
                    f"line {line_number}."
                )

            pages.append(
                ExtractedPage.model_validate_json(line)
            )

    return pages


def save_chunks(
    chunks: list[DocumentChunk],
    output_path: Path,
) -> None:
    """Save chunks as one JSON object per line."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w", encoding="utf-8") as output_file:
        for chunk in chunks:
            output_file.write(
                chunk.model_dump_json()
            )
            output_file.write("\n")


def main() -> None:
    """Chunk the parsed Infosys FY2025 report."""

    input_path = (
        PROCESSED_REPORTS_DIR
        / f"{TARGET_DOC_ID}.pages.jsonl"
    )
    output_path = (
        PROCESSED_REPORTS_DIR
        / f"{TARGET_DOC_ID}.chunks.jsonl"
    )

    pages = load_pages(input_path)
    chunks = create_document_chunks(pages)
    save_chunks(chunks, output_path)

    prose_count = sum(
        chunk.content_type == "prose"
        for chunk in chunks
    )
    table_count = sum(
        chunk.content_type == "table"
        for chunk in chunks
    )

    token_counts = [
        chunk.estimated_tokens
        for chunk in chunks
    ]

    print(f"Input pages: {len(pages)}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Prose chunks: {prose_count}")
    print(f"Table chunks: {table_count}")

    if token_counts:
        print(f"Smallest chunk: {min(token_counts)} tokens")
        print(f"Largest chunk: {max(token_counts)} tokens")

    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
