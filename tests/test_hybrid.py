"""Ranking, filtering, pagination, and evaluation regression tests."""

import math
from types import SimpleNamespace

import pytest
from qdrant_client import models

from app.retrieval import bm25
from app.retrieval.bm25 import BM25Index, tokenize
from app.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from evaluation.run_eval import evaluate, retrieval_query, score_results
from evaluation.schema import EvaluationQuestion


def point(id, text="revenue", doc="a", page=1):
    return models.ScoredPoint(id=id, version=0, score=0.9,
                              payload={"text": text, "doc_id": doc, "pdf_page": page,
                                       "company": doc, "fiscal_year": "2024-25"})


def test_bm25_formula_filters_and_zero_matches():
    index = BM25Index([point(1, "revenue revenue"), point(2, "employees", "b")])
    results = index.search("revenue")
    expected = math.log(2) * 2 * 2.5 / (2 + 1.5 * (0.25 + 0.75 * 2 / 1.5))
    assert [(p.id, p.score) for p in results] == [(1, pytest.approx(expected))]
    assert index.search("revenue", company="b") == []
    assert index.search("unknown") == []
    assert index.search("revenue", fiscal_year="2023-24") == []
    assert tokenize("REVENUE 1,62,990") == ["revenue", "162990"]


def test_empty_corpus_and_stable_ties():
    assert BM25Index([]).search("revenue") == []
    assert [p.id for p in BM25Index([point(2), point(1)]).search("revenue")] == [1, 2]
    with pytest.raises(ValueError):
        BM25Index([point(1), point(1)])


def test_scroll_reads_every_page(monkeypatch):
    calls = []
    def scroll(**kwargs):
        calls.append(kwargs)
        return ([point(1)], 1) if kwargs["offset"] is None else ([point(2)], None)
    monkeypatch.setattr(bm25, "get_qdrant_client", lambda: SimpleNamespace(scroll=scroll))
    index = BM25Index.from_qdrant()
    assert len(index.records) == 2
    assert [c["offset"] for c in calls] == [None, 1]
    assert all(c["with_vectors"] is False for c in calls)


def test_rrf_deduplicates_and_preserves_inputs():
    a, b, c = point(1), point(2), point(3)
    results = reciprocal_rank_fusion([[a, b, b], [c, b]], limit=3)
    assert [p.id for p in results] == [2, 1, 3]
    assert results[0].score == pytest.approx(2 / 62)
    assert b.score == 0.9
    assert results[0].payload == b.payload
    assert reciprocal_rank_fusion([[], []]) == []


@pytest.mark.parametrize("k", [-1, float("nan"), float("inf")])
def test_invalid_rrf_parameters(k):
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], rrf_k=k)


def test_hybrid_passes_identical_filters(monkeypatch):
    from app.retrieval import dense
    calls = []
    def search(question, **kwargs):
        calls.append((question, kwargs))
        return [point(1)]
    monkeypatch.setattr(dense, "search_chunks", search)
    retriever = HybridRetriever(SimpleNamespace(search=search), candidate_limit=10)
    assert len(retriever.search("revenue", limit=2, doc_id="a")) == 1
    assert calls == [("revenue", {"limit": 10, "doc_id": "a"})] * 2
    with pytest.raises(ValueError):
        retriever.search("revenue", limit=11)


def question(answerable=True):
    return EvaluationQuestion(
        id="q001", question="What about revenue?", type="follow_up" if answerable else "unanswerable",
        answerable=answerable, expected_answer="secret answer" if answerable else None,
        evidence=[{"doc_id": "a", "pdf_page": 1, "evidence_text": "secret evidence"},
                  {"doc_id": "b", "pdf_page": 2, "evidence_text": "other evidence"}] if answerable else [],
        history=[{"role": "user", "content": "Tell me about TCS."}], split="dev",
    )


def test_metrics_count_chunks_and_exclude_unanswerable():
    q = question()
    metrics = score_results(q, [point(1), point(2), point(3, doc="b", page=2)], 2)
    assert metrics == {"evidence_recall": 0.5, "document_recall": 0.5,
                       "mrr": 1.0, "all_evidence_hit": 0.0}
    assert all(v is None for v in score_results(question(False), [point(1)], 5).values())


def test_evaluation_uses_same_query_without_gold_and_reports_misses():
    calls = []
    def search(query, **kwargs):
        calls.append((query, kwargs))
        return [point(1)]
    report = evaluate([question(), question(False)], search, SimpleNamespace(search=search), k=2)
    assert len(report["questions"]) == 6
    assert all(query == retrieval_query(question()) for query, _ in calls)
    assert all("secret" not in query for query, _ in calls)
    assert all(kwargs == {"limit": 50} for _, kwargs in calls)
    assert report["questions"][0]["missing_evidence"] == [["b", 2]]
    overall = [r for r in report["summary"] if r["group"] == "all"]
    assert all(r["answerable"] == 1 and r["evidence_recall"] == 0.5 for r in overall)
