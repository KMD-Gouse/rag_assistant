"""Tests for multi-report ingestion orchestration."""

import pytest

from app.ingestion import pipeline


def test_pipeline_runs_stages_in_order_and_indexes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    document_ids = ["first_report", "second_report"]

    monkeypatch.setattr(
        pipeline,
        "parse_document",
        lambda doc_id: events.append(f"parse:{doc_id}"),
    )
    monkeypatch.setattr(
        pipeline,
        "chunk_document",
        lambda doc_id: events.append(f"chunk:{doc_id}"),
    )

    def fake_index_documents(
        doc_ids: list[str],
    ) -> dict[str, int]:
        events.append("index:" + ",".join(doc_ids))
        return {doc_id: 1 for doc_id in doc_ids}

    monkeypatch.setattr(
        pipeline,
        "index_documents",
        fake_index_documents,
    )

    counts = pipeline.run_pipeline(document_ids)

    assert events == [
        "parse:first_report",
        "chunk:first_report",
        "parse:second_report",
        "chunk:second_report",
        "index:first_report,second_report",
    ]
    assert counts == {
        "first_report": 1,
        "second_report": 1,
    }


def test_pipeline_rejects_empty_document_list() -> None:
    with pytest.raises(
        ValueError,
        match="At least one doc_id",
    ):
        pipeline.run_pipeline([])
