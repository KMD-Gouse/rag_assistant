"""Index document chunks and embeddings in Qdrant."""
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from qdrant_client import QdrantClient, models

from app.config import get_settings
from app.ingestion.embed import (
    DEFAULT_BATCH_SIZE,
    EMBEDDING_DIMENSION,
    embed_chunks,
    load_chunks,
    load_embedding_model,
    validate_embeddings,
)
from app.ingestion.schema import DocumentChunk
from app.ingestion.manifest import (
    get_manifest_document,
)
from app.qdrant import get_qdrant_client


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.json"
TARGET_DOC_ID = "infosys_fy2025"
QDRANT_UPLOAD_BATCH_SIZE = 128


def load_document_checksum(
    doc_id: str,
    manifest_path: Path = MANIFEST_PATH,
) -> str:
    """Load one report checksum from the manifest."""

    document = get_manifest_document(
        doc_id=doc_id,
        manifest_path=manifest_path,
    )
    return document.sha256.lower()


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
) -> bool:
    """Create the vector collection when it does not exist."""

    if client.collection_exists(collection_name):
        return False

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=EMBEDDING_DIMENSION,
            distance=models.Distance.COSINE,
        ),
    )
    return True


def make_qdrant_point_id(chunk_id: str) -> str:
    """Create a deterministic UUID for a chunk's Qdrant point."""

    return str(uuid5(NAMESPACE_URL, chunk_id))


def make_point_payload(
    chunk: DocumentChunk,
    document_checksum: str,
) -> dict[str, Any]:
    """Build searchable metadata stored with a vector."""

    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "company": chunk.company,
        "fiscal_year": chunk.fiscal_year,
        "section": chunk.section,
        "pdf_page": chunk.pdf_page,
        "printed_page": chunk.printed_page,
        "content_type": chunk.content_type,
        "text": chunk.text,
        "document_checksum": document_checksum,
    }


def build_qdrant_points(
    chunks: Sequence[DocumentChunk],
    embeddings: np.ndarray,
    document_checksum: str,
) -> list[models.PointStruct]:
    """Pair chunks with embeddings in their original order."""

    validate_embeddings(
        embeddings=embeddings,
        expected_count=len(chunks),
    )

    return [
        models.PointStruct(
            id=make_qdrant_point_id(chunk.chunk_id),
            vector=embedding.tolist(),
            payload=make_point_payload(
                chunk=chunk,
                document_checksum=document_checksum,
            ),
        )
        for chunk, embedding in zip(
            chunks,
            embeddings,
            strict=True,
        )
    ]


def make_document_filter(
    doc_id: str,
) -> models.Filter:
    """Create a Qdrant filter for one report."""

    return models.Filter(
        must=[
            models.FieldCondition(
                key="doc_id",
                match=models.MatchValue(
                    value=doc_id,
                ),
            )
        ]
    )


def load_existing_document_state(
    client: QdrantClient,
    collection_name: str,
    doc_id: str,
) -> tuple[set[str], set[str]]:
    """Read point IDs and checksums already stored for a report."""

    point_ids: set[str] = set()
    checksums: set[str] = set()
    next_offset = None

    while True:
        records, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=make_document_filter(doc_id),
            limit=256,
            offset=next_offset,
            with_payload=["document_checksum"],
            with_vectors=False,
        )

        for record in records:
            point_ids.add(str(record.id))

            payload = record.payload or {}
            checksum = payload.get(
                "document_checksum"
            )

            if isinstance(checksum, str):
                checksums.add(checksum)

        if next_offset is None:
            break

    return point_ids, checksums


def expected_point_ids(
    chunks: Sequence[DocumentChunk],
) -> set[str]:
    """Return deterministic Qdrant IDs for the supplied chunks."""

    return {
        make_qdrant_point_id(chunk.chunk_id)
        for chunk in chunks
    }


def document_is_current(
    chunks: Sequence[DocumentChunk],
    document_checksum: str,
    existing_point_ids: set[str],
    existing_checksums: set[str],
) -> bool:
    """Return whether Qdrant exactly matches local chunks."""

    return (
        existing_point_ids == expected_point_ids(chunks)
        and existing_checksums == {
            document_checksum
        }
    )


def upload_points(
    client: QdrantClient,
    collection_name: str,
    points: Sequence[models.PointStruct],
    batch_size: int = QDRANT_UPLOAD_BATCH_SIZE,
) -> int:
    """Upload Qdrant points in bounded batches."""

    if batch_size < 1:
        raise ValueError(
            "Upload batch_size must be at least 1."
        )

    uploaded_count = 0

    for batch_start in range(0, len(points), batch_size):
        batch = list(
            points[batch_start : batch_start + batch_size]
        )
        client.upsert(
            collection_name=collection_name,
            points=batch,
            wait=True,
        )
        uploaded_count += len(batch)
        print(f"Uploaded {uploaded_count}/{len(points)} points")

    return uploaded_count


def synchronize_document_points(
    client: QdrantClient,
    collection_name: str,
    chunks: Sequence[DocumentChunk],
    embeddings: np.ndarray,
    document_checksum: str,
    existing_point_ids: set[str],
) -> tuple[int, int]:
    """Upsert current points, then remove stale points for the report."""

    points = build_qdrant_points(
        chunks=chunks,
        embeddings=embeddings,
        document_checksum=document_checksum,
    )
    uploaded_count = upload_points(
        client=client,
        collection_name=collection_name,
        points=points,
    )

    stale_point_ids = existing_point_ids - expected_point_ids(chunks)
    if stale_point_ids:
        client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(
                points=sorted(stale_point_ids),
            ),
            wait=True,
        )

    return uploaded_count, len(stale_point_ids)


def index_documents(
    doc_ids: Sequence[str],
) -> dict[str, int]:
    """Index each report, leaving unchanged Qdrant points untouched."""

    if not doc_ids:
        raise ValueError("At least one doc_id is required.")

    settings = get_settings()
    client = get_qdrant_client()
    collection_name = settings.qdrant_collection
    created = ensure_collection(client, collection_name)
    action = "Created" if created else "Using"
    print(f"{action} Qdrant collection: {collection_name}")

    pending_documents: list[
        tuple[str, list[DocumentChunk], str, set[str]]
    ] = []
    counts: dict[str, int] = {}

    for doc_id in doc_ids:
        chunks_path = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / f"{doc_id}.chunks.jsonl"
        )
        print(f"Loading chunks: {chunks_path}")
        chunks = load_chunks(chunks_path)

        if not chunks:
            raise ValueError(f"No chunks found in {chunks_path}.")

        document_checksum = load_document_checksum(doc_id)
        existing_point_ids, existing_checksums = (
            load_existing_document_state(
                client=client,
                collection_name=collection_name,
                doc_id=doc_id,
            )
        )

        if document_is_current(
            chunks=chunks,
            document_checksum=document_checksum,
            existing_point_ids=existing_point_ids,
            existing_checksums=existing_checksums,
        ):
            print(f"{doc_id} is already current; skipping.")
            counts[doc_id] = len(chunks)
            continue

        pending_documents.append(
            (
                doc_id,
                chunks,
                document_checksum,
                existing_point_ids,
            )
        )

    if not pending_documents:
        return counts

    print(f"Loading embedding model: {settings.embedding_model}")
    model = load_embedding_model()

    for doc_id, chunks, document_checksum, existing_point_ids in (
        pending_documents
    ):
        print(f"Embedding {len(chunks)} chunks for {doc_id}...")
        embeddings = embed_chunks(
            chunks=chunks,
            model=model,
            batch_size=DEFAULT_BATCH_SIZE,
        )
        uploaded_count, deleted_count = synchronize_document_points(
            client=client,
            collection_name=collection_name,
            chunks=chunks,
            embeddings=embeddings,
            document_checksum=document_checksum,
            existing_point_ids=existing_point_ids,
        )
        counts[doc_id] = len(chunks)
        print(
            f"{doc_id}: uploaded {uploaded_count}, "
            f"removed {deleted_count} stale points."
        )

    return counts


def main() -> None:
    """Embed chunks and index them in Qdrant."""

    settings = get_settings()
    client = get_qdrant_client()
    counts = index_documents([TARGET_DOC_ID])

    collection = client.get_collection(
        settings.qdrant_collection
    )
    print(f"Indexed points: {counts[TARGET_DOC_ID]}")
    print(f"Collection point count: {collection.points_count}")
    print("Indexing completed.")


if __name__ == "__main__":
    main()
