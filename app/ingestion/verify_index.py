"""Verify indexed chunks and semantic search results."""

from pathlib import Path
from typing import Any

from qdrant_client import models

from app.config import get_settings
from app.ingestion.embed import (
    embed_texts,
    load_chunks,
    load_embedding_model,
)
from app.ingestion.index import (
    make_qdrant_point_id,
)
from app.qdrant import get_qdrant_client


QUERY_INSTRUCTION = (
    "Represent this sentence for searching relevant passages: "
)

TEST_QUESTION = (
    "How many employees does Infosys have globally?"
)
TARGET_DOC_ID = "infosys_fy2025"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / f"{TARGET_DOC_ID}.chunks.jsonl"
)


def require_payload(
    payload: dict[str, Any] | None,
) -> dict[str, Any]:
    """Ensure a retrieved point contains payload metadata."""

    if payload is None:
        raise ValueError("Qdrant point has no payload.")

    required_fields = {
        "chunk_id",
        "doc_id",
        "company",
        "fiscal_year",
        "section",
        "pdf_page",
        "printed_page",
        "content_type",
        "text",
        "document_checksum",
    }

    missing_fields = required_fields - payload.keys()

    if missing_fields:
        raise ValueError(
            "Qdrant payload is missing fields: "
            f"{sorted(missing_fields)}"
        )

    return payload


def main() -> None:
    """Verify count, point retrieval, and semantic search."""

    settings = get_settings()
    client = get_qdrant_client()
    chunks = load_chunks(CHUNKS_PATH)

    report_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="doc_id",
                match=models.MatchValue(
                    value=TARGET_DOC_ID,
                ),
            )
        ]
    )

    indexed_count = client.count(
        collection_name=settings.qdrant_collection,
        count_filter=report_filter,
        exact=True,
    ).count

    if indexed_count != len(chunks):
        raise ValueError(
            "Qdrant count does not match local chunks: "
            f"{indexed_count} != {len(chunks)}."
        )

    print(
        "Chunk count verified: "
        f"{indexed_count}"
    )

    sample_chunk = chunks[0]
    sample_point_id = make_qdrant_point_id(
        sample_chunk.chunk_id
    )

    retrieved = client.retrieve(
        collection_name=settings.qdrant_collection,
        ids=[sample_point_id],
        with_payload=True,
        with_vectors=False,
    )

    if len(retrieved) != 1:
        raise ValueError(
            "Could not retrieve the known sample point."
        )

    sample_payload = require_payload(
        retrieved[0].payload
    )

    if sample_payload["chunk_id"] != sample_chunk.chunk_id:
        raise ValueError(
            "Retrieved point has the wrong chunk_id."
        )

    print(
        "Deterministic point retrieval verified: "
        f"{sample_payload['chunk_id']}"
    )

    model = load_embedding_model()

    query_vector = embed_texts(
        texts=[
            QUERY_INSTRUCTION + TEST_QUESTION
        ],
        model=model,
        batch_size=1,
    )[0]

    response = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector.tolist(),
        query_filter=report_filter,
        limit=5,
        with_payload=True,
        with_vectors=False,
    )

    if not response.points:
        raise ValueError(
            "Semantic search returned no results."
        )

    print(f"\nQuestion: {TEST_QUESTION}")
    print("\nTop results:")

    answer_found = False

    for rank, result in enumerate(
        response.points,
        start=1,
    ):
        payload = require_payload(result.payload)
        text = str(payload["text"])

        if "3,23,578" in text:
            answer_found = True

        print(
            f"\n{rank}. Score: {result.score:.4f}"
        )
        print(f"   Chunk: {payload['chunk_id']}")
        print(f"   PDF page: {payload['pdf_page']}")
        print(f"   Section: {payload['section']}")
        print(f"   Text: {text[:250]}")

    if not answer_found:
        raise ValueError(
            "Expected employee value was not found "
            "in the top five results."
        )

    print("\nSemantic search validation passed.")


if __name__ == "__main__":
    main()
