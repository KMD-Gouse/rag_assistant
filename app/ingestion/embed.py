"""Generate normalized embeddings for document chunks."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from app.ingestion.schema import DocumentChunk


MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION = 384
DEFAULT_BATCH_SIZE = 32

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_REPORTS_DIR = PROJECT_ROOT / "data" / "processed"

TARGET_DOC_ID = "infosys_fy2025"


def load_chunks(
    input_path: Path,
) -> list[DocumentChunk]:
    """Load and validate document chunks from JSONL."""

    chunks: list[DocumentChunk] = []

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as input_file:
        for line_number, line in enumerate(
            input_file,
            start=1,
        ):
            if not line.strip():
                raise ValueError(
                    "Blank chunk record at "
                    f"line {line_number}."
                )

            chunks.append(
                DocumentChunk.model_validate_json(line)
            )

    return chunks


def load_embedding_model() -> SentenceTransformer:
    """Load the embedding model once for an ingestion run."""

    return SentenceTransformer(MODEL_NAME)


def embed_texts(
    texts: Sequence[str],
    model: SentenceTransformer,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """Convert text strings into normalized embedding vectors."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    if not texts:
        return np.empty(
            shape=(0, EMBEDDING_DIMENSION),
            dtype=np.float32,
        )

    if any(not text.strip() for text in texts):
        raise ValueError("Cannot embed empty text.")

    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    expected_shape = (
        len(texts),
        EMBEDDING_DIMENSION,
    )

    if embeddings.shape != expected_shape:
        raise ValueError(
            "Unexpected embedding shape: "
            f"expected {expected_shape}, "
            f"received {embeddings.shape}."
        )

    return embeddings


def embed_chunks(
    chunks: Sequence[DocumentChunk],
    model: SentenceTransformer,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """Embed chunks while preserving their existing order."""

    texts = [
        chunk.text
        for chunk in chunks
    ]

    return embed_texts(
        texts=texts,
        model=model,
        batch_size=batch_size,
    )


def validate_embeddings(
    embeddings: np.ndarray,
    expected_count: int,
) -> None:
    """Validate embedding count, dimensions, and normalization."""

    expected_shape = (
        expected_count,
        EMBEDDING_DIMENSION,
    )

    if embeddings.shape != expected_shape:
        raise ValueError(
            "Unexpected embedding matrix shape: "
            f"expected {expected_shape}, "
            f"received {embeddings.shape}."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "Embeddings contain NaN or infinite values."
        )

    if expected_count == 0:
        return

    vector_norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    if not np.allclose(
        vector_norms,
        1.0,
        atol=1e-4,
    ):
        raise ValueError(
            "Embedding vectors are not normalized."
        )


def main() -> None:
    """Generate and validate embeddings for one report."""

    input_path = (
        PROCESSED_REPORTS_DIR
        / f"{TARGET_DOC_ID}.chunks.jsonl"
    )

    chunks = load_chunks(input_path)

    print(f"Loaded chunks: {len(chunks)}")
    print(f"Loading model: {MODEL_NAME}")

    model = load_embedding_model()

    print("Generating embeddings...")

    embeddings = embed_chunks(
        chunks=chunks,
        model=model,
        batch_size=DEFAULT_BATCH_SIZE,
    )

    validate_embeddings(
        embeddings=embeddings,
        expected_count=len(chunks),
    )

    vector_norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    print(f"Embedding shape: {embeddings.shape}")
    print(f"Data type: {embeddings.dtype}")
    print(f"Minimum norm: {vector_norms.min():.6f}")
    print(f"Maximum norm: {vector_norms.max():.6f}")
    print("Embedding validation passed.")


if __name__ == "__main__":
    main()
