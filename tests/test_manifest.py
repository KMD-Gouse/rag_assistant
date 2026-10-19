"""Tests for annual-report manifest access."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.ingestion.manifest import (
    MANIFEST_PATH,
    get_manifest_document,
    get_manifest_document_ids,
    load_manifest,
)


EXPECTED_DOCUMENT_IDS = [
    "infosys_fy2025",
    "infosys_fy2024",
    "tcs_fy2025",
    "tcs_fy2024",
]


def test_real_manifest_contains_four_reports() -> None:
    manifest = load_manifest()

    assert manifest.schema_version == 1
    assert len(manifest.documents) == 4
    assert get_manifest_document_ids() == (
        EXPECTED_DOCUMENT_IDS
    )


def test_get_manifest_document() -> None:
    document = get_manifest_document(
        "tcs_fy2025"
    )

    assert document.company == "TCS"
    assert document.fiscal_year == "2024-25"
    assert document.filename == "tcs_fy2025.pdf"
    assert len(document.sha256) == 64
    assert document.pdf_pages == 336


def test_unknown_document_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="found 0",
    ):
        get_manifest_document("unknown_report")


def test_duplicate_document_ids_are_rejected(
    tmp_path: Path,
) -> None:
    manifest_data = json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )

    manifest_data["documents"].append(
        manifest_data["documents"][0]
    )

    duplicate_manifest_path = (
        tmp_path / "manifest.json"
    )
    duplicate_manifest_path.write_text(
        json.dumps(manifest_data),
        encoding="utf-8",
    )

    with pytest.raises(
        ValidationError,
        match="duplicate doc_id",
    ):
        load_manifest(duplicate_manifest_path)