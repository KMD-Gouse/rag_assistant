"""Data models produced by the PDF ingestion pipeline."""

from pydantic import BaseModel, ConfigDict, Field


class ExtractedTable(BaseModel):
    """One table extracted from a PDF page."""

    model_config = ConfigDict(extra="forbid")

    table_index: int = Field(ge=1)
    bbox: tuple[float, float, float, float]
    headers: list[str | None] = Field(default_factory=list)
    rows: list[list[str | None]] = Field(default_factory=list)


class ExtractedPage(BaseModel):
    """Content and source metadata extracted from one PDF page."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(pattern=r"^[a-z0-9_]+$")
    company: str = Field(min_length=1)
    fiscal_year: str = Field(pattern=r"^\d{4}-\d{2}$")
    pdf_page: int = Field(ge=1)
    printed_page: str | None = None
    text: str
    tables: list[ExtractedTable] = Field(default_factory=list)
    source_filename: str = Field(
        min_length=5,
        pattern=r"^.+\.pdf$",
    )
