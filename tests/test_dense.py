"""Tests for dense semantic retrieval and metadata filtering."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from qdrant_client import models

from app.retrieval import dense


def filter_values(
    query_filter: models.Filter,
) -> dict[str, object]:
    """Return exact-match values from a Qdrant filter."""

    values: dict[str, object] = {}

    for condition in query_filter.must or []:
        if not isinstance(condition, models.FieldCondition):
            continue
        if not isinstance(condition.match, models.MatchValue):
            continue

        values[condition.key] = condition.match.value

    return values


def test_empty_metadata_creates_no_filter() -> None:
    assert dense.build_search_filter() is None


def test_metadata_filters_are_combined() -> None:
    query_filter = dense.build_search_filter(
        doc_id="infosys_fy2025",
        company="Infosys",
        fiscal_year="2024-25",
    )

    assert query_filter is not None
    assert filter_values(query_filter) == {
        "doc_id": "infosys_fy2025",
        "company": "Infosys",
        "fiscal_year": "2024-25",
    }


@pytest.mark.parametrize(
    ("question", "limit", "message"),
    [
        ("   ", 5, "Question must not be empty"),
        ("Valid question", 0, "Search limit must be at least 1"),
    ],
)
def test_search_rejects_invalid_input(
    question: str,
    limit: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        dense.search_chunks(question, limit=limit)


def test_search_embeds_question_and_sends_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dense.get_search_model.cache_clear()
    calls: dict[str, Any] = {}
    expected_point = models.ScoredPoint(
        id="point-1",
        version=1,
        score=0.9,
        payload={"doc_id": "infosys_fy2025"},
    )

    class FakeClient:
        def query_points(self, **kwargs: Any) -> object:
            calls["query"] = kwargs
            return SimpleNamespace(points=[expected_point])

    def fake_embed_texts(**kwargs: Any) -> np.ndarray:
        calls["embedding"] = kwargs
        return np.array([[0.25, 0.75]], dtype=np.float32)

    fake_model = object()
    monkeypatch.setattr(
        dense,
        "get_settings",
        lambda: SimpleNamespace(qdrant_collection="test_chunks"),
    )
    monkeypatch.setattr(
        dense,
        "get_qdrant_client",
        lambda: FakeClient(),
    )
    monkeypatch.setattr(
        dense,
        "load_embedding_model",
        lambda: fake_model,
    )
    monkeypatch.setattr(dense, "embed_texts", fake_embed_texts)

    results = dense.search_chunks(
        "  How many employees?  ",
        limit=3,
        doc_id="infosys_fy2025",
    )

    assert results == [expected_point]
    assert calls["embedding"] == {
        "texts": [dense.QUERY_INSTRUCTION + "How many employees?"],
        "model": fake_model,
        "batch_size": 1,
    }

    query_call = calls["query"]
    assert query_call["collection_name"] == "test_chunks"
    assert query_call["query"] == [0.25, 0.75]
    assert query_call["limit"] == 3
    assert query_call["with_payload"] is True
    assert query_call["with_vectors"] is False
    assert filter_values(query_call["query_filter"]) == {
        "doc_id": "infosys_fy2025"
    }
    dense.get_search_model.cache_clear()


@pytest.mark.integration
@pytest.mark.parametrize(
    ("doc_id", "question", "expected_text"),
    [
        (
            "infosys_fy2024",
            "How many employees did Infosys report globally?",
            "3,17,240",
        ),
        (
            "infosys_fy2025",
            "How many employees did Infosys report globally?",
            "3,23,578",
        ),
        (
            "tcs_fy2024",
            "How many employees did TCS report?",
            "601,546",
        ),
        (
            "tcs_fy2025",
            "How many permanent employees did TCS report?",
            "6,07,979",
        ),
    ],
)
def test_live_filtered_search_finds_known_fact(
    doc_id: str,
    question: str,
    expected_text: str,
) -> None:
    results = dense.search_chunks(
        question,
        limit=10,
        doc_id=doc_id,
    )

    assert results
    assert all(
        result.payload is not None
        and result.payload["doc_id"] == doc_id
        for result in results
    )
    assert any(
        result.payload is not None
        and expected_text in str(result.payload["text"])
        for result in results
    )
