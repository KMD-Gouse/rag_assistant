"""Data models produced by the PDF ingestion pipeline."""

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


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

ChunkContentType = Literal["prose", "table"]


class DocumentChunk(BaseModel):
    """One retrieval unit produced from an extracted PDF page."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(
        pattern=(
            r"^[a-z0-9_]+:"
            r"p\d{4,}:"
            r"(prose|table):"
            r"\d{3,}$"
        )
    )

    doc_id: str = Field(pattern=r"^[a-z0-9_]+$")
    company: str = Field(min_length=1)
    fiscal_year: str = Field(pattern=r"^\d{4}-\d{2}$")
    pdf_page: int = Field(ge=1)
    printed_page: str | None = None
    section: str | None = None
    content_type: ChunkContentType
    table_index: int | None = Field(default=None, ge=1)
    text: str = Field(min_length=1)
    estimated_tokens: int = Field(ge=1)
    source_filename: str = Field(
        min_length=5,
        pattern=r"^.+\.pdf$",
    )

    @model_validator(mode="after")
    def validate_table_index(self) -> "DocumentChunk":
        if self.content_type == "table":
            if self.table_index is None:
                raise ValueError(
                    "Table chunks require table_index."
                )

        if self.content_type == "prose":
            if self.table_index is not None:
                raise ValueError(
                    "Prose chunks cannot have table_index."
                )

        return self
