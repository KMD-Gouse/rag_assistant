"""Run the complete ingestion pipeline for one or all reports."""

import argparse

from app.ingestion.chunk import chunk_document
from app.ingestion.index import index_documents
from app.ingestion.manifest import get_manifest_document_ids
from app.ingestion.parse import parse_document


def run_pipeline(doc_ids: list[str]) -> dict[str, int]:
    """Parse, chunk, embed, and index selected reports."""

    if not doc_ids:
        raise ValueError("At least one doc_id is required.")

    for doc_id in doc_ids:
        print(f"\n=== Parsing {doc_id} ===")
        parse_document(doc_id)

        print(f"\n=== Chunking {doc_id} ===")
        chunk_document(doc_id)

    print("\n=== Embedding and indexing ===")
    counts = index_documents(doc_ids)

    print("\n=== Pipeline summary ===")
    for doc_id in doc_ids:
        print(f"{doc_id}: {counts[doc_id]} indexed points")

    return counts


def main() -> None:
    """Select one report or every manifest report."""

    document_ids = get_manifest_document_ids()
    argument_parser = argparse.ArgumentParser(
        description=(
            "Parse, chunk, embed, and index annual reports."
        )
    )
    argument_parser.add_argument(
        "doc_id",
        nargs="?",
        choices=document_ids,
        help="One document ID from data/manifest.json.",
    )
    argument_parser.add_argument(
        "--all",
        action="store_true",
        help="Process every document in manifest order.",
    )
    arguments = argument_parser.parse_args()

    if arguments.all and arguments.doc_id is not None:
        argument_parser.error(
            "Choose either one doc_id or --all, not both."
        )
    if not arguments.all and arguments.doc_id is None:
        argument_parser.error("Provide a doc_id or use --all.")

    selected_ids = (
        document_ids
        if arguments.all
        else [arguments.doc_id]
    )
    run_pipeline(selected_ids)


if __name__ == "__main__":
    main()
