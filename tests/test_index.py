"""Tests for deterministic and idempotent Qdrant indexing."""

import json
from pathlib import Path

import numpy as np
import pytest
from qdrant_client import QdrantClient

from app.ingestion.embed import EMBEDDING_DIMENSION
from app.ingestion.index import (
    build_qdrant_points,
    document_is_current,
    ensure_collection,
    expected_point_ids,
    load_document_checksum,
    load_existing_document_state,
    make_qdrant_point_id,
    synchronize_document_points,
    upload_points,
)
from app.ingestion.schema import DocumentChunk


COLLECTION_NAME = "test_chunks"


def make_chunk(
    sequence_number: int,
    text: str,
) -> DocumentChunk:
    """Create a valid test chunk."""

    return DocumentChunk(
        chunk_id=(
            "test_report:"
            "p0001:"
            f"prose:{sequence_number:03d}"
        ),
        doc_id="test_report",
        company="Test Company",
        fiscal_year="2024-25",
        pdf_page=1,
        printed_page="1",
        section="Test section",
        content_type="prose",
        text=text,
        estimated_tokens=10,
        source_filename="test_report.pdf",
    )


def make_embeddings(count: int) -> np.ndarray:
    """Create normalized vectors for tests."""

    embeddings = np.zeros(
        (count, EMBEDDING_DIMENSION),
        dtype=np.float32,
    )

    for index in range(count):
        embeddings[index, index] = 1.0

    return embeddings


def make_client() -> QdrantClient:
    """Create an isolated in-memory Qdrant collection."""

    client = QdrantClient(":memory:")
    ensure_collection(client, COLLECTION_NAME)
    return client


def test_point_id_is_deterministic() -> None:
    chunk_id = "test_report:p0001:prose:001"

    assert make_qdrant_point_id(chunk_id) == (
        make_qdrant_point_id(chunk_id)
    )


def test_load_document_checksum(tmp_path: Path) -> None:
    checksum = "a" * 64
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "doc_id": "test_report",
                        "company": "Test Company",
                        "fiscal_year": "2024-25",
                        "fiscal_year_end": "2025-03-31",
                        "title": "Test Annual Report",
                        "filename": "test_report.pdf",
                        "source_url": "https://example.com/test.pdf",
                        "sha256": checksum,
                        "downloaded_at": (
                            "2026-10-19T00:00:00+00:00"
                        ),
                        "pdf_pages": 1,
                    }
                ],
                "schema_version": 1,
            }
        ),
        encoding="utf-8",
    )

    assert load_document_checksum(
        "test_report",
        manifest_path,
    ) == checksum


def test_unchanged_document_is_current() -> None:
    client = make_client()
    chunks = [
        make_chunk(1, "First chunk"),
        make_chunk(2, "Second chunk"),
    ]
    checksum = "a" * 64
    points = build_qdrant_points(
        chunks,
        make_embeddings(2),
        checksum,
    )
    upload_points(
        client,
        COLLECTION_NAME,
        points,
        batch_size=1,
    )

    point_ids, checksums = load_existing_document_state(
        client,
        COLLECTION_NAME,
        "test_report",
    )

    assert point_ids == expected_point_ids(chunks)
    assert checksums == {checksum}
    assert document_is_current(
        chunks,
        checksum,
        point_ids,
        checksums,
    )


def test_changed_document_replaces_and_removes_stale_points() -> None:
    client = make_client()
    old_chunks = [
        make_chunk(1, "Old first chunk"),
        make_chunk(2, "Old stale chunk"),
    ]
    upload_points(
        client,
        COLLECTION_NAME,
        build_qdrant_points(
            old_chunks,
            make_embeddings(2),
            "a" * 64,
        ),
    )

    existing_ids, _ = load_existing_document_state(
        client,
        COLLECTION_NAME,
        "test_report",
    )
    new_chunks = [
        make_chunk(1, "Updated first chunk"),
        make_chunk(3, "New third chunk"),
    ]

    uploaded, deleted = synchronize_document_points(
        client=client,
        collection_name=COLLECTION_NAME,
        chunks=new_chunks,
        embeddings=make_embeddings(2),
        document_checksum="b" * 64,
        existing_point_ids=existing_ids,
    )

    final_ids, final_checksums = load_existing_document_state(
        client,
        COLLECTION_NAME,
        "test_report",
    )

    assert uploaded == 2
    assert deleted == 1
    assert final_ids == expected_point_ids(new_chunks)
    assert final_checksums == {"b" * 64}
    assert client.count(COLLECTION_NAME).count == 2

    updated = client.retrieve(
        COLLECTION_NAME,
        [make_qdrant_point_id(new_chunks[0].chunk_id)],
        with_payload=True,
    )
    assert updated[0].payload is not None
    assert updated[0].payload["text"] == "Updated first chunk"


def test_upload_failure_does_not_delete_existing_points() -> None:
    class FailingClient:
        delete_called = False

        def upsert(self, **kwargs: object) -> None:
            raise RuntimeError("simulated upload failure")

        def delete(self, **kwargs: object) -> None:
            self.delete_called = True

    client = FailingClient()
    chunk = make_chunk(1, "Replacement chunk")

    with pytest.raises(
        RuntimeError,
        match="simulated upload failure",
    ):
        synchronize_document_points(
            client=client,  # type: ignore[arg-type]
            collection_name=COLLECTION_NAME,
            chunks=[chunk],
            embeddings=make_embeddings(1),
            document_checksum="b" * 64,
            existing_point_ids={"stale-id"},
        )

    assert client.delete_called is False
