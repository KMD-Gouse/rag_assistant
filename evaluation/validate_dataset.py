"""Validate the JSONL evaluation dataset."""

import json
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from evaluation.schema import EvaluationQuestion


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = PROJECT_ROOT / "evaluation" / "questions.jsonl"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.json"
EXPECTED_QUESTION_COUNT = 60
EXPECTED_DEV_COUNT = 40
EXPECTED_HELD_OUT_COUNT = 20
EXPECTED_UNANSWERABLE_COUNT = 10


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

    expected_ids = [
        f"q{question_number:03d}"
        for question_number in range(
            1,
            EXPECTED_QUESTION_COUNT + 1,
        )
    ]
    actual_ids = [question.id for question in questions]

    if actual_ids != expected_ids:
        raise ValueError(
            "Question IDs must be sequential from q001 through "
            f"q{EXPECTED_QUESTION_COUNT:03d}."
        )

    normalized_questions = [
        " ".join(question.question.casefold().split())
        for question in questions
    ]
    duplicate_questions = sorted(
        question_text
        for question_text, count in Counter(
            normalized_questions
        ).items()
        if count > 1
    )

    if duplicate_questions:
        raise ValueError(
            "Dataset contains duplicate question text: "
            f"{duplicate_questions}."
        )

    split_counts = Counter(question.split for question in questions)
    expected_split_counts = {
        "dev": EXPECTED_DEV_COUNT,
        "held_out": EXPECTED_HELD_OUT_COUNT,
    }

    if dict(split_counts) != expected_split_counts:
        raise ValueError(
            "Dataset split counts must be "
            f"{expected_split_counts}, received {dict(split_counts)}."
        )

    incorrectly_split_ids = [
        question.id
        for index, question in enumerate(questions, start=1)
        if question.split
        != ("dev" if index <= EXPECTED_DEV_COUNT else "held_out")
    ]

    if incorrectly_split_ids:
        raise ValueError(
            "q001-q040 must be dev and q041-q060 must be held_out; "
            f"incorrect IDs: {incorrectly_split_ids}."
        )

    unanswerable_count = sum(
        not question.answerable
        for question in questions
    )

    if unanswerable_count != EXPECTED_UNANSWERABLE_COUNT:
        raise ValueError(
            "Dataset must contain exactly "
            f"{EXPECTED_UNANSWERABLE_COUNT} unanswerable questions; "
            f"received {unanswerable_count}."
        )

    return questions


def main() -> None:
    questions = validate_dataset()

    type_counts = Counter(question.type for question in questions)
    split_counts = Counter(question.split for question in questions)

    print(f"Dataset is valid: {len(questions)} questions")
    print(f"Question types: {dict(type_counts)}")
    print(f"Dataset splits: {dict(split_counts)}")
    print(
        "Unanswerable questions: "
        f"{sum(not question.answerable for question in questions)}"
    )


if __name__ == "__main__":
    main()
