"""Metrics for evaluating retrieval quality."""

from itertools import islice
from typing import Iterable


EvidenceKey = tuple[str, int]
EvidenceGroup = set[EvidenceKey]


def evidence_recall_at_k(
    expected_evidence: Iterable[EvidenceKey],
    retrieved_evidence: Iterable[EvidenceKey],
    k: int,
) -> float | None:
    """Calculate the fraction of expected evidence found in the top k results."""

    if k < 1:
        raise ValueError("k must be at least 1.")

    expected = set(expected_evidence)

    # Unanswerable questions have no expected evidence,
    # so retrieval recall does not apply to them.
    if not expected:
        return None

    retrieved_top_k = set(islice(retrieved_evidence, k))
    matches = expected.intersection(retrieved_top_k)

    return len(matches) / len(expected)


def document_recall_at_k(
    expected_evidence: Iterable[EvidenceKey],
    retrieved_evidence: Iterable[EvidenceKey],
    k: int,
) -> float | None:
    """Calculate the fraction of required documents found in the top k."""

    if k < 1:
        raise ValueError("k must be at least 1.")

    expected_documents = {
        doc_id
        for doc_id, _ in expected_evidence
    }

    if not expected_documents:
        return None

    retrieved_documents = {
        doc_id
        for doc_id, _ in islice(retrieved_evidence, k)
    }

    matches = expected_documents.intersection(retrieved_documents)

    return len(matches) / len(expected_documents)


def grouped_evidence_recall_at_k(
    expected_groups: Iterable[EvidenceGroup],
    retrieved_evidence: Iterable[EvidenceKey],
    k: int,
) -> float | None:
    """Recall required facts while accepting any verified location per fact."""

    if k < 1:
        raise ValueError("k must be at least 1.")
    groups = [set(group) for group in expected_groups]
    if not groups:
        return None
    if any(not group for group in groups):
        raise ValueError("Evidence groups must not be empty.")
    retrieved = set(islice(retrieved_evidence, k))
    return sum(bool(group & retrieved) for group in groups) / len(groups)


def grouped_document_recall_at_k(
    expected_groups: Iterable[EvidenceGroup],
    retrieved_evidence: Iterable[EvidenceKey],
    k: int,
) -> float | None:
    """Document recall with alternative documents accepted within a fact group."""

    groups = [set(group) for group in expected_groups]
    if not groups:
        return None
    document_groups = [{doc_id for doc_id, _ in group} for group in groups]
    retrieved_documents = {doc_id for doc_id, _ in islice(retrieved_evidence, k)}
    return sum(bool(group & retrieved_documents) for group in document_groups) / len(groups)
