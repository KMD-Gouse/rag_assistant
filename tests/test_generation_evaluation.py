import csv
from pathlib import Path

from app.generation.models import AnswerResult, ValidatedCitation
from evaluation.generation import (
    automatic_metrics,
    load_questions,
    numeric_recall,
    threshold_sweep,
    token_f1,
    write_review_csv,
)
from evaluation.schema import EvaluationQuestion


def question(answerable=True):
    return EvaluationQuestion.model_validate({
        "id": "q001",
        "question": "What revenue was reported?" if answerable else "Future revenue?",
        "type": "fact" if answerable else "unanswerable",
        "answerable": answerable,
        "expected_answer": "Revenue was ₹255,324 crore in FY2025." if answerable else None,
        "evidence": ([{
            "doc_id": "tcs_fy2025", "pdf_page": 88,
            "evidence_text": "Revenue 255,324", "group": "revenue",
        }] if answerable else []),
        "history": [],
        "split": "development",
    })


def result(abstained=False, score=0.8):
    citations = [] if abstained else [ValidatedCitation(
        evidence_id="E1", chunk_id="c1", doc_id="tcs_fy2025",
        company="TCS", fiscal_year="2024-25", pdf_page=88,
        quote="Revenue 255,324",
    )]
    return AnswerResult(
        question="What revenue was reported?",
        answer=("I don't know based on the available annual reports."
                if abstained else "Revenue was ₹255,324 crore in FY2025."),
        abstained=abstained,
        abstention_reason="model_says_unanswerable" if abstained else None,
        evidence_sufficiency=score,
        threshold=0.0,
        citations=citations,
        retrieved_chunks=5,
        model="model",
    ).model_dump(mode="json")


def test_reference_diagnostics_and_gold_citation_metrics():
    metrics = automatic_metrics(question(), result())
    assert token_f1("Revenue was 255,324", "Revenue was 255324") == 1.0
    assert numeric_recall("₹255,324 crore and 6.0%", "255324 crore; growth 6.0%") == 1.0
    assert metrics["answerability_decision_correct"]
    assert metrics["citation_location_precision"] == 1.0
    assert metrics["citation_group_recall"] == 1.0


def test_threshold_sweep_reuses_saved_scores():
    report = {"questions": [
        {"question_id": "q001", "gold_answerable": True, "result": result(score=0.8)},
        {"question_id": "q002", "gold_answerable": False, "result": result(score=0.6)},
    ]}
    at_seven = next(row for row in threshold_sweep(report) if row["threshold"] == 0.7)
    assert at_seven["answerability_accuracy"] == 1.0
    assert at_seven["answerable_coverage"] == 1.0
    assert at_seven["unanswerable_abstention_accuracy"] == 1.0


def test_review_csv_preserves_manual_columns(tmp_path: Path):
    path = tmp_path / "review.csv"
    report = {"questions": [{
        "question_id": "q001", "type": "fact", "question": "Question?",
        "gold_answerable": True, "expected_answer": "Expected", "result": result(),
    }]}
    write_review_csv(report, path)
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    rows[0]["answer_correct"] = "1"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    write_review_csv(report, path)
    saved = list(csv.DictReader(path.open(encoding="utf-8-sig")))[0]
    assert saved["answer_correct"] == "1"


def test_answerable_abstention_is_scored_without_manual_review(tmp_path: Path):
    path = tmp_path / "review.csv"
    abstained = result(abstained=True, score=0.0)
    report = {"questions": [{
        "question_id": "q001", "type": "fact", "question": "Question?",
        "gold_answerable": True, "expected_answer": "Expected", "result": abstained,
    }]}
    write_review_csv(report, path)
    saved = list(csv.DictReader(path.open(encoding="utf-8-sig")))[0]
    assert saved["answer_correct"] == "0"
    assert saved["citations_entail_answer"] == "0"
    assert saved["citations_complete"] == "0"
    sweep = threshold_sweep(report, {"q001": saved})
    assert sweep[0]["reviewed_end_to_end_success"] == 0.0
