"""Reranking contracts and controlled Week 5 ablation regressions."""

from types import SimpleNamespace

import pytest

from app.retrieval.reranker import Reranker, RerankedRetriever, ensemble_rank
from evaluation.week5 import diagnose, evaluate_week5, render_analysis
from tests.test_hybrid import point, question


def test_pair_scoring_deduplicates_preserves_payload_and_input_scores():
    calls = []
    def predict(pairs, **kwargs):
        calls.append((pairs, kwargs))
        return [0.1, 0.9]
    a, b = point(1, "first"), point(2, "second")
    ranker = Reranker(model=SimpleNamespace(predict=predict))
    result = ranker.rerank("query", [a, b, a], limit=2)
    assert [p.id for p in result] == [2, 1]
    assert calls[0][0][0][0] == "query"
    assert calls[0][0][0][1].endswith("\nfirst")
    assert "Company: a" in calls[0][0][0][1]
    assert calls[0][0][1][1].endswith("\nsecond")
    assert result[0].payload == b.payload
    assert result[1].score == 0.1
    assert a.score == 0.9


@pytest.mark.parametrize("scores", [[float("nan")], [float("inf")], [], [0.1, 0.2]])
def test_bad_model_output_is_rejected(scores):
    ranker = Reranker(model=SimpleNamespace(predict=lambda *args, **kwargs: scores))
    with pytest.raises(ValueError, match="one finite score"):
        ranker.rerank("query", [point(1)])


def test_empty_pool_does_not_load_model_and_ties_are_stable():
    ranker = Reranker(model=SimpleNamespace(predict=lambda *a, **k: [0.5, 0.5]))
    assert ranker.rerank("query", []) == []
    assert [p.id for p in ranker.rerank("query", [point(2), point(1)])] == [1, 2]
    with pytest.raises(ValueError):
        ranker.rerank(" ", [])
    with pytest.raises(ValueError):
        ranker.rerank("query", [point(1, "")])


def test_ensemble_promotes_exact_content_phrase() -> None:
    exact = point(1, "The total contract value was US$39.4 billion.")
    broad = point(2, "The proposed transaction had a high total value.")
    results = ensemble_rank(
        "What total contract value was reported?",
        [broad, exact],
        [broad, exact],
        limit=2,
    )
    assert [result.id for result in results] == [1, 2]


def test_ensemble_rejects_invalid_input() -> None:
    with pytest.raises(ValueError):
        ensemble_rank(" ", [], [], 1)
    with pytest.raises(ValueError):
        ensemble_rank("query", [], [], 0)


def test_reranked_retriever_forwards_filters_and_candidate_budget():
    calls = []
    def search(query, **kwargs):
        calls.append((query, kwargs))
        return [point(1)]
    ranker = Reranker(model=SimpleNamespace(predict=lambda *a, **k: [0.3]))
    retriever = RerankedRetriever(SimpleNamespace(search=search, candidate_limit=10), ranker, 8)
    assert retriever.search("query", limit=2, company="a")[0].id == 1
    assert calls == [("query", {"limit": 8, "company": "a"})]
    with pytest.raises(ValueError):
        retriever.search("query", limit=9)


def test_failure_stages_distinguish_coverage_from_ordering():
    q = question()
    a, b = point(1), point(2, doc="b", page=2)
    corpus = {("a", 1), ("b", 2)}
    def stages(corpus, union, pool, final):
        return [e["stage"] for e in diagnose(q, corpus, union, pool, final)]
    assert stages(set(), [], [], []) == ["corpus_page_missing"] * 2
    assert stages(corpus, [a], [a], [a]) == ["hit", "candidate_miss"]
    assert stages(corpus, [a, b], [a], [a]) == ["hit", "fusion_cutoff"]
    assert stages(corpus, [a, b], [a, b], [a]) == ["hit", "reranker_top_k_miss"]


def test_ablation_uses_shared_pool_and_reports_reranker_gains():
    a, b = point(1), point(2, doc="b", page=2)
    irrelevant = point(3, doc="c", page=3)
    calls = []
    def search(query, **kwargs):
        calls.append((query, kwargs))
        return [irrelevant, a, b]
    class FakeReranker:
        def rerank(self, query, pool, limit):
            assert [p.id for p in pool] == [3, 1, 2]
            return [a, b, irrelevant][:limit]
    bm25 = SimpleNamespace(search=search, records=[a, b, irrelevant])
    report = evaluate_week5([question()], search, bm25, FakeReranker(), k=2, candidates=3, pool_size=3)
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert "secret" not in calls[0][0]
    rows = {r["method"]: r for r in report["questions"]}
    assert rows["v2_hybrid"]["metrics"]["evidence_recall"] == 0.5
    assert rows["v3_reranked"]["metrics"]["evidence_recall"] == 1.0
    assert report["diagnostics"][0]["gained_pages"] == [["b", 2]]
    assert report["diagnostics"][0]["pool_evidence_recall"] == 1.0
    assert "v3_reranked" in render_analysis(report)


def test_unanswerable_metrics_remain_excluded():
    bm25 = SimpleNamespace(search=lambda *a, **k: [], records=[])
    ranker = Reranker()
    report = evaluate_week5([question(False)], lambda *a, **k: [], bm25, ranker)
    assert all(r["metrics"]["mrr"] is None for r in report["questions"])
    assert report["diagnostics"][0]["failure_stages"] == []
    assert "Evidence loss stages" in render_analysis(report)
