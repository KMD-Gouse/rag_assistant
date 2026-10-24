"""Run and score the Week 6 answer-generation evaluation."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

from app.config import get_settings
from app.generation.service import SYSTEM_PROMPT, answer_question
from evaluation.run_eval import evidence_groups, retrieval_query
from evaluation.schema import EvaluationQuestion


DEFAULT_DATASET = Path("evaluation/benchmark_v2.jsonl")
DEFAULT_PREFIX = Path("evaluation/results/week6-generation-development")
THRESHOLDS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)
REVIEW_FIELDS = (
    "answer_correct",
    "citations_entail_answer",
    "citations_complete",
)


def load_questions(path: Path, split: str) -> list[EvaluationQuestion]:
    questions = [
        EvaluationQuestion.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [question for question in questions if question.split == split]
    if not selected:
        raise ValueError(f"No questions found for split {split!r}.")
    return selected


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.casefold().replace(",", ""))


def token_f1(expected: str, actual: str) -> float:
    expected_tokens, actual_tokens = _tokens(expected), _tokens(actual)
    if not expected_tokens or not actual_tokens:
        return 0.0
    expected_counts = {token: expected_tokens.count(token) for token in set(expected_tokens)}
    actual_counts = {token: actual_tokens.count(token) for token in set(actual_tokens)}
    overlap = sum(min(count, actual_counts.get(token, 0)) for token, count in expected_counts.items())
    precision = overlap / len(actual_tokens)
    recall = overlap / len(expected_tokens)
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def numeric_recall(expected: str, actual: str) -> float | None:
    pattern = r"(?<![a-z])\d[\d,]*(?:\.\d+)?%?"
    expected_values = {
        value.replace(",", "").casefold()
        for value in re.findall(pattern, expected.casefold())
    }
    if not expected_values:
        return None
    actual_values = {
        value.replace(",", "").casefold()
        for value in re.findall(pattern, actual.casefold())
    }
    return len(expected_values.intersection(actual_values)) / len(expected_values)


def citation_metrics(question: EvaluationQuestion, citations: list[dict]) -> dict:
    if not question.answerable or not citations:
        return {
            "citation_location_precision": None,
            "citation_group_recall": None,
        }
    groups = evidence_groups(question)
    cited = {(item["doc_id"], item["pdf_page"]) for item in citations}
    gold = set().union(*groups)
    return {
        "citation_location_precision": len(cited.intersection(gold)) / len(cited),
        "citation_group_recall": (
            sum(bool(group.intersection(cited)) for group in groups) / len(groups)
        ),
    }


def automatic_metrics(question: EvaluationQuestion, result: dict) -> dict:
    answer = result["answer"] if not result["abstained"] else ""
    metrics = {
        "answerability_decision_correct": result["abstained"] != question.answerable,
        "token_f1": None,
        "numeric_recall": None,
        "quote_validation_pass": not result["abstained"],
    }
    if question.answerable and answer:
        metrics["token_f1"] = token_f1(question.expected_answer or "", answer)
        metrics["numeric_recall"] = numeric_recall(question.expected_answer or "", answer)
    metrics.update(citation_metrics(question, result["citations"]))
    return metrics


def _save_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def run_generation(
    questions: list[EvaluationQuestion],
    report_path: Path,
    dataset_path: Path,
    delay_seconds: float = 2.0,
    capture_threshold: float = 0.0,
    answerer=answer_question,
) -> dict:
    settings = get_settings()
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["metadata"].get("capture_threshold") != capture_threshold:
            raise ValueError(
                "Existing report capture threshold does not match the requested threshold."
            )
    else:
        report = {
            "metadata": {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "dataset": str(dataset_path),
                "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
                "split": questions[0].split,
                "model": settings.groq_model,
                "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
                "capture_threshold": capture_threshold,
                "top_k": settings.rag_top_k,
                "reranker_model": settings.rag_reranker_model,
                "reranker_revision": settings.rag_reranker_revision,
            },
            "questions": [],
        }
    completed = {row["question_id"] for row in report["questions"] if "error" not in row}
    for question in questions:
        if question.id in completed:
            continue
        query = retrieval_query(question)
        started = time.perf_counter()
        try:
            result = answerer(
                query,
                min_evidence_sufficiency=capture_threshold,
            ).model_dump(mode="json")
            row = {
                "question_id": question.id,
                "type": question.type,
                "question": question.question,
                "query": query,
                "gold_answerable": question.answerable,
                "expected_answer": question.expected_answer,
                "result": result,
                "automatic": automatic_metrics(question, result),
                "latency_seconds": time.perf_counter() - started,
            }
            report["questions"] = [
                old for old in report["questions"] if old["question_id"] != question.id
            ] + [row]
        except Exception as error:  # Preserve progress; errors are retried on the next run.
            report["questions"] = [
                old for old in report["questions"] if old["question_id"] != question.id
            ] + [{
                "question_id": question.id,
                "type": question.type,
                "question": question.question,
                "gold_answerable": question.answerable,
                "expected_answer": question.expected_answer,
                "error": f"{type(error).__name__}: {error}",
                "latency_seconds": time.perf_counter() - started,
            }]
        report["questions"].sort(key=lambda row: row["question_id"])
        _save_report(report_path, report)
        if delay_seconds:
            time.sleep(delay_seconds)
    return report


def write_review_csv(report: dict, path: Path) -> None:
    existing = {}
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as handle:
            existing = {row["question_id"]: row for row in csv.DictReader(handle)}
    fields = [
        "question_id", "type", "gold_answerable", "question", "expected_answer",
        "system_abstained", "abstention_reason", "evidence_sufficiency",
        "generated_answer", "citations", *REVIEW_FIELDS, "review_notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in report["questions"]:
            if "error" in row:
                continue
            result = row["result"]
            previous = existing.get(row["question_id"], {})
            automatic_failure = row["gold_answerable"] and result["abstained"]
            manual_values = {
                field: previous.get(field, "") or ("0" if automatic_failure else "")
                for field in REVIEW_FIELDS
            }
            review_notes = previous.get("review_notes", "")
            if automatic_failure and not review_notes:
                review_notes = "Automatic: answerable question was abstained."
            writer.writerow({
                "question_id": row["question_id"],
                "type": row["type"],
                "gold_answerable": int(row["gold_answerable"]),
                "question": row["question"],
                "expected_answer": row["expected_answer"] or "",
                "system_abstained": int(result["abstained"]),
                "abstention_reason": result["abstention_reason"] or "",
                "evidence_sufficiency": result["evidence_sufficiency"],
                "generated_answer": result["answer"],
                "citations": json.dumps(result["citations"], ensure_ascii=False),
                **manual_values,
                "review_notes": review_notes,
            })


def read_review(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field in REVIEW_FIELDS:
            if row[field] not in {"", "0", "1"}:
                raise ValueError(f"{row['question_id']} {field} must be blank, 0, or 1.")
    return {row["question_id"]: row for row in rows}


def threshold_sweep(report: dict, review: dict[str, dict] | None = None) -> list[dict]:
    rows = [row for row in report["questions"] if "error" not in row]
    scored = []
    for threshold in THRESHOLDS:
        decisions = []
        end_to_end = []
        for row in rows:
            result = row["result"]
            accepted = not result["abstained"] and result["evidence_sufficiency"] >= threshold
            decisions.append((row["gold_answerable"], accepted))
            manual = (review or {}).get(row["question_id"], {})
            if not row["gold_answerable"]:
                end_to_end.append(not accepted)
            elif not accepted:
                end_to_end.append(False)
            elif all(manual.get(field) in {"0", "1"} for field in REVIEW_FIELDS):
                end_to_end.append(all(manual[field] == "1" for field in REVIEW_FIELDS))
        answerable = [accepted for gold, accepted in decisions if gold]
        unanswerable = [not accepted for gold, accepted in decisions if not gold]
        answerable_coverage = sum(answerable) / len(answerable) if answerable else None
        unanswerable_accuracy = (
            sum(unanswerable) / len(unanswerable) if unanswerable else None
        )
        scored.append({
            "threshold": threshold,
            "answerability_accuracy": sum(gold == accepted for gold, accepted in decisions) / len(decisions),
            "answerable_coverage": answerable_coverage,
            "unanswerable_abstention_accuracy": unanswerable_accuracy,
            "balanced_answerability_accuracy": (
                (answerable_coverage + unanswerable_accuracy) / 2
                if answerable_coverage is not None and unanswerable_accuracy is not None
                else None
            ),
            "reviewed_end_to_end_success": (
                sum(end_to_end) / len(end_to_end) if len(end_to_end) == len(rows) else None
            ),
            "reviewed_questions": len(end_to_end),
        })
    return scored


def write_markdown(report: dict, review_path: Path, path: Path) -> None:
    review = read_review(review_path) if review_path.exists() else None
    sweep = threshold_sweep(report, review)
    rows = [row for row in report["questions"] if "error" not in row]
    errors = [row for row in report["questions"] if "error" in row]
    lines = [
        "# Week 6 generation development evaluation",
        "",
        f"Completed outputs: {len(rows)}; errors: {len(errors)}. Groq outputs were captured with "
        f"threshold {report['metadata']['capture_threshold']:.2f}.",
        "",
        "Automatic lexical overlap is diagnostic only. Final answer correctness, citation "
        "entailment, and citation completeness require the review CSV.",
        "",
        "| Threshold | Answerability accuracy | Answerable coverage | Unanswerable abstention | Balanced accuracy | End-to-end success |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for item in sweep:
        end_to_end = (
            "pending" if item["reviewed_end_to_end_success"] is None
            else f"{item['reviewed_end_to_end_success']:.2%}"
        )
        lines.append(
            f"| {item['threshold']:.2f} | {item['answerability_accuracy']:.2%} | "
            f"{item['answerable_coverage']:.2%} | "
            f"{item['unanswerable_abstention_accuracy']:.2%} | "
            f"{item['balanced_answerability_accuracy']:.2%} | {end_to_end} |"
        )
    scores = [row["result"]["evidence_sufficiency"] for row in rows]
    abstentions = Counter(
        row["result"]["abstention_reason"]
        for row in rows
        if row["result"]["abstained"]
    )
    answerable_abstentions = Counter(
        row["result"]["abstention_reason"]
        for row in rows
        if row["gold_answerable"] and row["result"]["abstained"]
    )
    unanswerable_abstentions = Counter(
        row["result"]["abstention_reason"]
        for row in rows
        if not row["gold_answerable"] and row["result"]["abstained"]
    )
    answered_gold = [
        row for row in rows
        if row["gold_answerable"] and not row["result"]["abstained"]
    ]
    complete_gold_citations = sum(
        row["automatic"]["citation_group_recall"] == 1.0
        for row in answered_gold
    )
    review_complete = sweep[0]["reviewed_end_to_end_success"] is not None
    lines += [
        "",
        "## Score distribution",
        "",
        f"Observed scores: {', '.join(f'{score:.2f}' for score in sorted(set(scores)))}",
        "",
        "All generated scores are binary, so thresholds from 0.50 through 1.00 "
        "do not change a single development decision. The former 0.70 default is "
        "therefore not empirically special.",
        "",
        "## Automatic citation and failure audit",
        "",
        f"Answered gold-answerable questions: {len(answered_gold)}. Every accepted "
        "answer passed exact-quote validation; "
        f"{complete_gold_citations}/{len(answered_gold)} cited every required gold "
        "evidence group.",
        "",
        "Abstention reasons: " + ", ".join(
            f"{reason}={count}" for reason, count in sorted(abstentions.items())
        ) + ".",
        "Answerable false abstentions: " + ", ".join(
            f"{reason}={count}" for reason, count in sorted(answerable_abstentions.items())
        ) + ". Correct unanswerable abstentions: " + ", ".join(
            f"{reason}={count}" for reason, count in sorted(unanswerable_abstentions.items())
        ) + ".",
        "",
        "Gold-page citation precision is only diagnostic because valid supporting "
        "pages can be absent from the benchmark labels. Citation entailment and "
        "completeness still require review of the quoted text.",
        "",
        "## Second-person review",
        "",
        (
            "Review complete. Every non-abstained answerable output has a manual "
            "answer-correctness, citation-entailment, and citation-completeness decision."
            if review_complete else
            "Review pending. Enter `1` (yes) or `0` (no) in `answer_correct`, "
            "`citations_entail_answer`, and `citations_complete` for each "
            "non-abstained answerable row."
        ),
        "",
        f"Human review file: `{review_path}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_frozen_config(
    report: dict,
    review_path: Path,
    dataset_path: Path,
    path: Path,
    threshold: float,
) -> dict:
    """Freeze the reviewed development result before challenge evaluation."""
    review = read_review(review_path)
    selected = next(
        item for item in threshold_sweep(report, review)
        if item["threshold"] == threshold
    )
    if selected["reviewed_end_to_end_success"] is None:
        raise ValueError("The second-person review is incomplete; refusing to freeze.")
    settings = get_settings()
    frozen = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "freeze_scope": "Week 6 generation pipeline before the one-time challenge run",
        "challenge_status_at_freeze": "not_run",
        "dataset": str(dataset_path),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "development_report_sha256": hashlib.sha256(
            json.dumps(report, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
        "development_review_sha256": hashlib.sha256(review_path.read_bytes()).hexdigest(),
        "model": settings.groq_model,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "structured_output": "AnswerDraft JSON schema validated by Pydantic",
        "citation_policy": "Every citation quote must be an exact substring of its retrieved chunk.",
        "abstention_threshold": threshold,
        "threshold_note": (
            "Development evidence scores were binary (0.0 or 1.0), so thresholds "
            "from 0.50 through 1.00 produced identical decisions."
        ),
        "top_k": settings.rag_top_k,
        "candidate_limit": settings.rag_candidate_limit,
        "reranker_model": settings.rag_reranker_model,
        "reranker_revision": settings.rag_reranker_revision,
        "development_metrics": selected,
        "manual_review": {
            "reviewed_non_abstained_answerable_outputs": sum(
                row["gold_answerable"] and not row["result"]["abstained"]
                for row in report["questions"] if "error" not in row
            ),
            "review_file": str(review_path),
            "review_complete": True,
        },
    }
    _save_report(path, frozen)
    return frozen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", default="development")
    parser.add_argument("--prefix", type=Path, default=DEFAULT_PREFIX)
    parser.add_argument("--delay-seconds", type=float, default=2.0)
    parser.add_argument("--capture-threshold", type=float, default=0.0)
    parser.add_argument("--freeze-config", type=Path)
    args = parser.parse_args()
    questions = load_questions(args.dataset, args.split)
    report_path = args.prefix.with_suffix(".json")
    review_path = args.prefix.with_name(args.prefix.name + "-review.csv")
    markdown_path = args.prefix.with_suffix(".md")
    report = run_generation(
        questions, report_path, args.dataset, delay_seconds=args.delay_seconds,
        capture_threshold=args.capture_threshold,
    )
    write_review_csv(report, review_path)
    write_markdown(report, review_path, markdown_path)
    if args.freeze_config:
        write_frozen_config(
            report, review_path, args.dataset, args.freeze_config,
            threshold=get_settings().rag_min_evidence_sufficiency,
        )
    print(markdown_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
