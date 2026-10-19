"""Read and validate annual-report manifest metadata."""

import json
from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.json"


class ManifestDocument(BaseModel):
    """Metadata for one annual report."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(pattern=r"^[a-z0-9_]+$")
    company: str = Field(min_length=1)
    fiscal_year: str = Field(pattern=r"^\d{4}-\d{2}$")
    fiscal_year_end: str
    title: str = Field(min_length=1)
    filename: str = Field(pattern=r"^.+\.pdf$")
    source_url: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    downloaded_at: str
    pdf_pages: int = Field(ge=1)


class Manifest(BaseModel):
    """Validated annual-report manifest."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(ge=1)
    documents: list[ManifestDocument]

    @model_validator(mode="after")
    def validate_unique_document_ids(self) -> "Manifest":
        document_ids = [
            document.doc_id
            for document in self.documents
        ]

        if len(document_ids) != len(set(document_ids)):
            raise ValueError(
                "Manifest contains duplicate doc_id values."
            )

        return self


def load_manifest(
    manifest_path: Path = MANIFEST_PATH,
) -> Manifest:
    """Load and validate the complete manifest."""

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    manifest_data = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    return Manifest.model_validate(manifest_data)


def get_manifest_document(
    doc_id: str,
    manifest_path: Path = MANIFEST_PATH,
) -> ManifestDocument:
    """Return exactly one manifest document by ID."""

    manifest = load_manifest(manifest_path)

    matches = [
        document
        for document in manifest.documents
        if document.doc_id == doc_id
    ]

    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one manifest document "
            f"for doc_id={doc_id!r}; found {len(matches)}."
        )

    return matches[0]


def get_manifest_document_ids(
    manifest_path: Path = MANIFEST_PATH,
) -> list[str]:
    """Return document IDs in manifest order."""

    manifest = load_manifest(manifest_path)

    return [
        document.doc_id
        for document in manifest.documents
    ]