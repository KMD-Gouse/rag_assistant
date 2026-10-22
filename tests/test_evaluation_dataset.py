"""Tests for the complete Week 3 evaluation dataset."""

from collections import Counter

from evaluation.validate_dataset import (
    EXPECTED_DEV_COUNT,
    EXPECTED_HELD_OUT_COUNT,
    EXPECTED_QUESTION_COUNT,
    EXPECTED_UNANSWERABLE_COUNT,
    validate_dataset,
)
from evaluation.validate_benchmark_v2 import validate_benchmark


def test_complete_evaluation_dataset() -> None:
    questions = validate_dataset()

    assert len(questions) == EXPECTED_QUESTION_COUNT
    assert Counter(question.split for question in questions) == {
        "dev": EXPECTED_DEV_COUNT,
        "held_out": EXPECTED_HELD_OUT_COUNT,
    }
    assert sum(
        not question.answerable
        for question in questions
    ) == EXPECTED_UNANSWERABLE_COUNT


def test_answerable_questions_cover_every_report() -> None:
    questions = validate_dataset()
    evidence_documents = {
        evidence.doc_id
        for question in questions
        if question.answerable
        for evidence in question.evidence
    }

    assert evidence_documents == {
        "infosys_fy2024",
        "infosys_fy2025",
        "tcs_fy2024",
        "tcs_fy2025",
    }


def test_benchmark_v2_is_balanced_and_has_verified_alternatives() -> None:
    questions = validate_benchmark()
    assert Counter(q.split for q in questions) == {"development": 40, "challenge": 20}
    q013 = next(question for question in questions if question.id == "q013")
    groups = Counter(evidence.group for evidence in q013.evidence)
    assert groups["employees_fy2024"] == 2
    assert groups["employees_fy2025"] == 2
