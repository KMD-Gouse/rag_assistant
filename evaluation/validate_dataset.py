"""Validate the JSONL evaluation dataset."""

import json
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from evaluation.schema import EvaluationQuestion


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = PROJECT_ROOT / "evaluation" / "questions.jsonl"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.json"


def load_manifest_documents() -> dict[str, dict]:
    """Load manifest documents indexed by document ID."""

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    documents = manifest["documents"]
    document_ids = [document["doc_id"] for document in documents]

    if len(document_ids) != len(set(document_ids)):
        raise ValueError("The manifest contains duplicate document IDs.")

    return {
        document["doc_id"]: document
        for document in documents
    }


def validate_dataset() -> list[EvaluationQuestion]:
    """Load and validate every non-empty JSONL line."""

    documents = load_manifest_documents()
    questions: list[EvaluationQuestion] = []
    seen_question_ids: set[str] = set()

    for line_number, raw_line in enumerate(
        QUESTIONS_PATH.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue

        try:
            raw_question = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Line {line_number} is not valid JSON: {error}"
            ) from error

        try:
            question = EvaluationQuestion.model_validate(raw_question)
        except ValidationError as error:
            raise ValueError(
                f"Line {line_number} failed schema validation:\n{error}"
            ) from error

        if question.id in seen_question_ids:
            raise ValueError(
                f"Line {line_number} has duplicate ID {question.id!r}."
            )

        seen_question_ids.add(question.id)

        evidence_doc_ids = {
            evidence.doc_id
            for evidence in question.evidence
        }

        unknown_doc_ids = evidence_doc_ids - documents.keys()

        if unknown_doc_ids:
            raise ValueError(
                f"Line {line_number} references unknown documents: "
                f"{sorted(unknown_doc_ids)}"
            )

        if question.type == "comparison" and len(evidence_doc_ids) < 2:
            raise ValueError(
                f"Line {line_number}: comparison questions require "
                "evidence from at least two documents."
            )

        for evidence in question.evidence:
            page_count = documents[evidence.doc_id]["pdf_pages"]

            if page_count is None:
                raise ValueError(
                    f"Document {evidence.doc_id!r} has not been validated."
                )

            if evidence.pdf_page > page_count:
                raise ValueError(
                    f"Line {line_number}: PDF page "
                    f"{evidence.pdf_page} exceeds the {page_count} pages "
                    f"in {evidence.doc_id!r}."
                )

        questions.append(question)

    return questions


def main() -> None:
    questions = validate_dataset()

    type_counts = Counter(question.type for question in questions)
    split_counts = Counter(question.split for question in questions)

    print(f"Dataset is valid: {len(questions)} questions")
    print(f"Question types: {dict(type_counts)}")
    print(f"Dataset splits: {dict(split_counts)}")


if __name__ == "__main__":
    main()