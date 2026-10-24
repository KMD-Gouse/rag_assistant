"""Validate the frozen Week 6 generation challenge before its one-time run."""

from collections import Counter
import json
from pathlib import Path

from evaluation.schema import EvaluationQuestion
from evaluation.validate_dataset import load_manifest_documents


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "evaluation" / "generation_challenge_v1.jsonl"
DEVELOPMENT_PATH = ROOT / "evaluation" / "benchmark_v2.jsonl"


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def validate_challenge() -> list[EvaluationQuestion]:
    questions = [
        EvaluationQuestion.model_validate_json(line)
        for line in PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(questions) != 20 or len({question.id for question in questions}) != 20:
        raise ValueError("Generation challenge requires 20 unique questions.")
    if [question.id for question in questions] != [f"q{i:03d}" for i in range(101, 121)]:
        raise ValueError("Generation challenge IDs must be q101 through q120 in order.")
    if Counter(question.split for question in questions) != {"challenge": 20}:
        raise ValueError("Every generation challenge question must use split='challenge'.")
    if sum(not question.answerable for question in questions) != 6:
        raise ValueError("Generation challenge requires exactly six unanswerable questions.")

    development_questions = {
        _normalized(EvaluationQuestion.model_validate_json(line).question)
        for line in DEVELOPMENT_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    overlap = [
        question.id for question in questions
        if _normalized(question.question) in development_questions
    ]
    if overlap:
        raise ValueError(f"Challenge duplicates development questions: {overlap}.")

    documents = load_manifest_documents()
    page_text: dict[tuple[str, int], str] = {}
    for document_id in documents:
        chunks_path = ROOT / "data" / "processed" / f"{document_id}.chunks.jsonl"
        for line in chunks_path.read_text(encoding="utf-8").splitlines():
            chunk = json.loads(line)
            key = (chunk["doc_id"], chunk["pdf_page"])
            page_text[key] = page_text.get(key, "") + "\n" + chunk["text"]

    for question in questions:
        for evidence in question.evidence:
            document = documents.get(evidence.doc_id)
            if document is None or evidence.pdf_page > document["pdf_pages"]:
                raise ValueError(f"Invalid evidence location in {question.id}.")
            if evidence.evidence_text not in page_text.get(
                (evidence.doc_id, evidence.pdf_page), ""
            ):
                raise ValueError(f"Evidence text is not exact source text in {question.id}.")
    return questions


if __name__ == "__main__":
    result = validate_challenge()
    print(f"Generation challenge is valid: {len(result)} questions")
    print(f"Types: {dict(Counter(question.type for question in result))}")
    print(f"Answerable: {sum(question.answerable for question in result)}")
    print(f"Unanswerable: {sum(not question.answerable for question in result)}")
