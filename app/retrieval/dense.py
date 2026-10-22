"""Dense semantic search over indexed annual-report chunks."""

import argparse
from functools import lru_cache
from typing import Any, Sequence

from qdrant_client import models
from sentence_transformers import SentenceTransformer

from app.config import get_settings
from app.ingestion.embed import (
    embed_texts,
    load_embedding_model,
)
from app.qdrant import get_qdrant_client


QUERY_INSTRUCTION = (
    "Represent this sentence for searching relevant passages: "
)


@lru_cache(maxsize=1)
def get_search_model() -> SentenceTransformer:
    """Load and reuse one embedding model per search process."""

    return load_embedding_model()


def build_search_filter(
    doc_id: str | None = None,
    company: str | Sequence[str] | None = None,
    fiscal_year: str | Sequence[str] | None = None,
) -> models.Filter | None:
    """Build an AND filter from the provided metadata."""

    conditions: list[models.Condition] = []

    for key, value in (
        ("doc_id", doc_id),
        ("company", company),
        ("fiscal_year", fiscal_year),
    ):
        if value is not None:
            match = (
                models.MatchAny(any=list(value))
                if not isinstance(value, str)
                else models.MatchValue(value=value)
            )
            conditions.append(
                models.FieldCondition(
                    key=key,
                    match=match,
                )
            )

    if not conditions:
        return None

    return models.Filter(must=conditions)


def search_chunks(
    question: str,
    limit: int = 5,
    doc_id: str | None = None,
    company: str | Sequence[str] | None = None,
    fiscal_year: str | Sequence[str] | None = None,
) -> list[models.ScoredPoint]:
    """Return the most semantically relevant indexed chunks."""

    question = question.strip()

    if not question:
        raise ValueError("Question must not be empty.")

    if limit < 1:
        raise ValueError("Search limit must be at least 1.")

    settings = get_settings()
    client = get_qdrant_client()
    model = get_search_model()

    query_vector = embed_texts(
        texts=[QUERY_INSTRUCTION + question],
        model=model,
        batch_size=1,
    )[0]

    query_filter = build_search_filter(
        doc_id=doc_id,
        company=company,
        fiscal_year=fiscal_year,
    )

    response = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector.tolist(),
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )

    return list(response.points)


def print_results(
    question: str,
    results: list[models.ScoredPoint],
) -> None:
    """Print retrieved chunks in a readable format."""

    print(f"\nQuestion: {question}")
    print(f"Results returned: {len(results)}")

    for rank, result in enumerate(results, start=1):
        payload: dict[str, Any] = result.payload or {}

        print(f"\n{rank}. Score: {result.score:.4f}")
        print(f"   Document: {payload.get('doc_id')}")
        print(f"   Company: {payload.get('company')}")
        print(f"   Fiscal year: {payload.get('fiscal_year')}")
        print(f"   PDF page: {payload.get('pdf_page')}")
        print(f"   Printed page: {payload.get('printed_page')}")
        print(f"   Section: {payload.get('section')}")
        print(f"   Content type: {payload.get('content_type')}")
        print(f"   Text: {str(payload.get('text', ''))[:500]}")


def main() -> None:
    """Run semantic search from the command line."""

    argument_parser = argparse.ArgumentParser(
        description="Search indexed annual-report chunks."
    )
    argument_parser.add_argument(
        "question",
        help="Question to search for.",
    )
    argument_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to return.",
    )
    argument_parser.add_argument(
        "--doc-id",
        choices=[
            "infosys_fy2024",
            "infosys_fy2025",
            "tcs_fy2024",
            "tcs_fy2025",
        ],
        help="Search only one indexed report.",
    )
    argument_parser.add_argument(
        "--company",
        choices=["Infosys", "TCS"],
        help="Search only one company.",
    )
    argument_parser.add_argument(
        "--fiscal-year",
        choices=["2023-24", "2024-25"],
        help="Search only one fiscal year.",
    )
    arguments = argument_parser.parse_args()

    results = search_chunks(
        question=arguments.question,
        limit=arguments.limit,
        doc_id=arguments.doc_id,
        company=arguments.company,
        fiscal_year=arguments.fiscal_year,
    )
    print_results(
        question=arguments.question,
        results=results,
    )


if __name__ == "__main__":
    main()
