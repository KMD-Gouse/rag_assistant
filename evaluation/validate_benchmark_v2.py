"""Validate benchmark v2 balance, evidence groups, and source references."""

from collections import Counter
import json
from pathlib import Path
import re

from evaluation.schema import EvaluationQuestion
from evaluation.validate_dataset import load_manifest_documents


PATH = Path(__file__).resolve().parent / "benchmark_v2.jsonl"


def validate_benchmark() -> list[EvaluationQuestion]:
    documents = load_manifest_documents()
    questions = [
        EvaluationQuestion.model_validate(json.loads(line))
        for line in PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(questions) != 60 or len({q.id for q in questions}) != 60:
        raise ValueError("Benchmark v2 requires 60 unique questions.")
    if Counter(q.split for q in questions) != {"development": 40, "challenge": 20}:
        raise ValueError("Benchmark v2 requires a 40/20 split.")
    expected_types = {"fact": 9, "comparison": 4, "table": 2, "follow_up": 2, "unanswerable": 3}
    actual_types = Counter(q.type for q in questions if q.split == "challenge")
    if actual_types != expected_types:
        raise ValueError(f"Challenge type balance differs: {actual_types}.")
    for question in questions:
        for evidence in question.evidence:
            document = documents.get(evidence.doc_id)
            if document is None or evidence.pdf_page > document["pdf_pages"]:
                raise ValueError(f"Invalid evidence location in {question.id}.")
    challenge = [q for q in questions if q.split == "challenge"]
    page_counts: Counter = Counter()
    document_counts: Counter = Counter()
    for question in challenge:
        seen_groups: set[str] = set()
        for index, evidence in enumerate(question.evidence):
            group = evidence.group or f"__{index}"
            if group in seen_groups:
                continue
            seen_groups.add(group)
            page_counts[(evidence.doc_id, evidence.pdf_page)] += 1
            document_counts[evidence.doc_id] += 1
    if page_counts and max(page_counts.values()) > 2:
        raise ValueError("Challenge split overuses one canonical evidence page.")
    if document_counts and max(document_counts.values()) - min(document_counts.values()) > 2:
        raise ValueError("Challenge evidence is imbalanced across reports.")

    # Alternative labels are especially easy to add incorrectly. Verify every
    # grouped location retains the numeric facts named in its evidence snippet.
    for question in questions:
        grouped_counts = Counter(e.group for e in question.evidence if e.group)
        for evidence in question.evidence:
            if evidence.group is None or grouped_counts[evidence.group] < 2:
                continue
            chunks_path = PATH.parents[1] / "data" / "processed" / f"{evidence.doc_id}.chunks.jsonl"
            page_text = " ".join(
                json.loads(line)["text"]
                for line in chunks_path.read_text(encoding="utf-8").splitlines()
                if json.loads(line)["pdf_page"] == evidence.pdf_page
            )
            expected_numbers = {
                re.sub(r"\D", "", token)
                for token in re.findall(r"\d[\d,.]*", evidence.evidence_text)
            }
            page_numbers = {
                re.sub(r"\D", "", token)
                for token in re.findall(r"\d[\d,.]*", page_text)
            }
            if not expected_numbers.issubset(page_numbers):
                raise ValueError(f"Alternative evidence text not found on page for {question.id}.")
    return questions


if __name__ == "__main__":
    result = validate_benchmark()
    print(f"Benchmark v2 is valid: {len(result)} questions")
