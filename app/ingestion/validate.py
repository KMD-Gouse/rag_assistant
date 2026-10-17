"""Validate downloaded reports and update their manifest metadata."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pymupdf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.json"
RAW_REPORTS_DIR = PROJECT_ROOT / "data" / "raw"


def calculate_sha256(file_path: Path) -> str:
    """Calculate a file's SHA-256 checksum without loading it all at once."""

    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def validate_pdf(file_path: Path) -> dict[str, Any]:
    """Validate a PDF and return metadata gathered from it."""

    if not file_path.is_file():
        raise FileNotFoundError(f"Report not found: {file_path}")

    with pymupdf.open(file_path) as document:
        if not document.is_pdf:
            raise ValueError(f"File is not a valid PDF: {file_path}")

        if document.needs_pass:
            raise ValueError(f"PDF requires a password: {file_path}")

        page_count = document.page_count

    if page_count == 0:
        raise ValueError(f"PDF contains no pages: {file_path}")

    return {
        "sha256": calculate_sha256(file_path),
        "pdf_pages": page_count,
    }


def update_manifest(doc_id: str) -> dict[str, Any]:
    """Validate one manifest document and save its metadata."""

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    matches = [
        document
        for document in manifest["documents"]
        if document["doc_id"] == doc_id
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one document with doc_id={doc_id!r}, "
            f"found {len(matches)}."
        )

    document = matches[0]
    file_path = RAW_REPORTS_DIR / document["filename"]
    metadata = validate_pdf(file_path)

    document.update(metadata)

    if not document.get("downloaded_at"):
        document["downloaded_at"] = datetime.now(timezone.utc).isoformat()

    temporary_path = MANIFEST_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(MANIFEST_PATH)

    return document


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a downloaded annual report."
    )
    parser.add_argument("doc_id", help="Document ID from data/manifest.json")
    arguments = parser.parse_args()

    document = update_manifest(arguments.doc_id)

    print(f"Validated: {document['doc_id']}")
    print(f"Pages: {document['pdf_pages']}")
    print(f"SHA-256: {document['sha256']}")


if __name__ == "__main__":
    main()