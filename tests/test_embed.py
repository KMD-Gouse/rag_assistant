"""Tests for document embedding generation."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.ingestion.embed import (
    EMBEDDING_DIMENSION,
    embed_chunks,
    embed_texts,
    load_chunks,
    validate_embeddings,
)
from app.ingestion.schema import DocumentChunk


class FakeEmbeddingModel:
    """Predictable replacement for SentenceTransformer."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        self.calls.append(
            {
                "sentences": sentences,
                "batch_size": batch_size,
                "show_progress_bar": show_progress_bar,
                "convert_to_numpy": convert_to_numpy,
                "normalize_embeddings": normalize_embeddings,
            }
        )

        embeddings = np.zeros(
            (
                len(sentences),
                EMBEDDING_DIMENSION,
            ),
            dtype=np.float32,
        )

        for index in range(len(sentences)):
            embeddings[
                index,
                index % EMBEDDING_DIMENSION,
            ] = 1.0

        return embeddings


class WrongShapeModel(FakeEmbeddingModel):
    """Return vectors with an invalid dimension."""

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        return np.zeros(
            (len(sentences), 10),
            dtype=np.float32,
        )


def make_chunk(
    text: str,
    sequence_number: int,
) -> DocumentChunk:
    """Create a valid chunk for embedding tests."""

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


def test_embed_texts_returns_normalized_matrix() -> None:
    model = FakeEmbeddingModel()

    embeddings = embed_texts(
        texts=[
            "Infosys revenue increased.",
            "Employees globally 3,23,578.",
        ],
        model=model,  # type: ignore[arg-type]
        batch_size=2,
    )

    assert embeddings.shape == (
        2,
        EMBEDDING_DIMENSION,
    )
    assert embeddings.dtype == np.float32

    norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    assert np.allclose(norms, 1.0)

    assert len(model.calls) == 1

    call = model.calls[0]

    assert call["batch_size"] == 2
    assert call["normalize_embeddings"] is True
    assert call["convert_to_numpy"] is True


def test_embed_texts_handles_empty_input() -> None:
    model = FakeEmbeddingModel()

    embeddings = embed_texts(
        texts=[],
        model=model,  # type: ignore[arg-type]
    )

    assert embeddings.shape == (
        0,
        EMBEDDING_DIMENSION,
    )
    assert embeddings.dtype == np.float32
    assert model.calls == []


@pytest.mark.parametrize(
    "batch_size",
    [0, -1],
)
def test_embed_texts_rejects_invalid_batch_size(
    batch_size: int,
) -> None:
    model = FakeEmbeddingModel()

    with pytest.raises(
        ValueError,
        match="batch_size must be at least 1",
    ):
        embed_texts(
            texts=["Valid text"],
            model=model,  # type: ignore[arg-type]
            batch_size=batch_size,
        )


def test_embed_texts_rejects_empty_text() -> None:
    model = FakeEmbeddingModel()

    with pytest.raises(
        ValueError,
        match="Cannot embed empty text",
    ):
        embed_texts(
            texts=["Valid text", "   "],
            model=model,  # type: ignore[arg-type]
        )


def test_embed_texts_rejects_wrong_dimension() -> None:
    model = WrongShapeModel()

    with pytest.raises(
        ValueError,
        match="Unexpected embedding shape",
    ):
        embed_texts(
            texts=["Test text"],
            model=model,  # type: ignore[arg-type]
        )


def test_embed_chunks_preserves_text_order() -> None:
    model = FakeEmbeddingModel()

    chunks = [
        make_chunk(
            "First chunk",
            sequence_number=1,
        ),
        make_chunk(
            "Second chunk",
            sequence_number=2,
        ),
    ]

    embeddings = embed_chunks(
        chunks=chunks,
        model=model,  # type: ignore[arg-type]
    )

    assert embeddings.shape == (
        2,
        EMBEDDING_DIMENSION,
    )

    assert model.calls[0]["sentences"] == [
        "First chunk",
        "Second chunk",
    ]


def test_validate_embeddings_accepts_valid_vectors() -> None:
    embeddings = np.zeros(
        (2, EMBEDDING_DIMENSION),
        dtype=np.float32,
    )

    embeddings[:, 0] = 1.0

    validate_embeddings(
        embeddings=embeddings,
        expected_count=2,
    )


def test_validate_embeddings_rejects_nan() -> None:
    embeddings = np.zeros(
        (1, EMBEDDING_DIMENSION),
        dtype=np.float32,
    )

    embeddings[0, 0] = np.nan

    with pytest.raises(
        ValueError,
        match="NaN or infinite",
    ):
        validate_embeddings(
            embeddings=embeddings,
            expected_count=1,
        )


def test_validate_embeddings_rejects_unnormalized_vectors() -> None:
    embeddings = np.ones(
        (1, EMBEDDING_DIMENSION),
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="not normalized",
    ):
        validate_embeddings(
            embeddings=embeddings,
            expected_count=1,
        )


def test_load_chunks_validates_jsonl(
    tmp_path: Path,
) -> None:
    chunk = make_chunk(
        "Test chunk",
        sequence_number=1,
    )

    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(
        chunk.model_dump_json() + "\n",
        encoding="utf-8",
    )

    loaded_chunks = load_chunks(chunks_path)

    assert loaded_chunks == [chunk]
