import pytest

from evaluation.metrics import (
    document_recall_at_k,
    evidence_recall_at_k,
)


def test_evidence_recall_at_k_finds_all_evidence() -> None:
    expected = [
        ("infosys_fy2025", 10),
        ("tcs_fy2025", 88),
    ]
    retrieved = [
        ("infosys_fy2025", 10),
        ("tcs_fy2025", 88),
        ("infosys_fy2024", 15),
    ]

    assert evidence_recall_at_k(expected, retrieved, k=2) == 1.0


def test_evidence_recall_at_k_finds_partial_evidence() -> None:
    expected = [
        ("infosys_fy2025", 10),
        ("tcs_fy2025", 88),
    ]
    retrieved = [
        ("infosys_fy2025", 10),
        ("tcs_fy2024", 75),
    ]

    assert evidence_recall_at_k(expected, retrieved, k=2) == 0.5


def test_evidence_recall_respects_k_limit() -> None:
    expected = [("tcs_fy2025", 88)]
    retrieved = [
        ("infosys_fy2025", 10),
        ("tcs_fy2025", 88),
    ]

    assert evidence_recall_at_k(expected, retrieved, k=1) == 0.0


def test_evidence_recall_is_not_applicable_without_expected_evidence() -> None:
    assert evidence_recall_at_k([], [], k=5) is None


def test_evidence_recall_rejects_invalid_k() -> None:
    with pytest.raises(ValueError, match="k must be at least 1"):
        evidence_recall_at_k([], [], k=0)


def test_document_recall_finds_all_required_documents() -> None:
    expected = [
        ("infosys_fy2025", 10),
        ("tcs_fy2025", 88),
    ]
    retrieved = [
        ("infosys_fy2025", 23),
        ("tcs_fy2025", 90),
    ]

    assert document_recall_at_k(expected, retrieved, k=2) == 1.0


def test_document_recall_finds_only_one_required_document() -> None:
    expected = [
        ("infosys_fy2025", 10),
        ("tcs_fy2025", 88),
    ]
    retrieved = [
        ("infosys_fy2025", 23),
        ("infosys_fy2025", 24),
    ]

    assert document_recall_at_k(expected, retrieved, k=2) == 0.5


def test_document_recall_respects_k_limit() -> None:
    expected = [
        ("infosys_fy2025", 10),
        ("tcs_fy2025", 88),
    ]
    retrieved = [
        ("infosys_fy2025", 23),
        ("tcs_fy2025", 90),
    ]

    assert document_recall_at_k(expected, retrieved, k=1) == 0.5


def test_document_recall_is_not_applicable_without_expected_evidence() -> None:
    assert document_recall_at_k([], [], k=5) is None
