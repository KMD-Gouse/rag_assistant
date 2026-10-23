"""Re-score ensemble ablations and nearby weights from saved model outputs."""

import argparse
import csv
import json
from pathlib import Path
import re

from evaluation.analyze_saved_fusion import evidence_groups, load_jsonl, score


STOPWORDS = {
    "a", "an", "and", "as", "at", "by", "did", "each", "for", "from",
    "how", "in", "it", "its", "of", "on", "report", "reported", "the",
    "to", "was", "were", "what", "which",
}


def bigrams(text: str) -> set[tuple[str, str]]:
    tokens = re.findall(r"[a-z0-9]+", text.casefold())
    return {
        (tokens[index], tokens[index + 1])
        for index in range(len(tokens) - 1)
        if tokens[index] not in STOPWORDS
        and tokens[index + 1] not in STOPWORDS
    }


def rank_ensemble(
    diagnostic: dict,
    fusion_weight: float,
    reranker_weight: float,
    phrase_weight: float,
    rrf_k: float = 60.0,
) -> list[dict]:
    pool = diagnostic["fused_candidates"][:50]
    payloads = {str(item["point_id"]): item["payload"] for item in pool}
    fused_ranks = {
        str(item["point_id"]): rank for rank, item in enumerate(pool, start=1)
    }
    reranked_ranks = {
        str(item["point_id"]): rank
        for rank, item in enumerate(diagnostic["cross_encoder_pool"], start=1)
        if str(item["point_id"]) in payloads
    }
    query_bigrams = bigrams(diagnostic["query"])
    scored = []
    for point_id, payload in payloads.items():
        reranker_rank = reranked_ranks[point_id]
        phrase_matches = len(query_bigrams.intersection(bigrams(payload["text"])))
        value = (
            fusion_weight / (rrf_k + fused_ranks[point_id])
            + reranker_weight / (rrf_k + reranker_rank)
            + phrase_weight * phrase_matches
        )
        scored.append((value, point_id, payload))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [payload for _, _, payload in scored]


def aggregate(
    questions: dict[str, dict],
    diagnostics: dict[str, dict],
    fusion_weight: float,
    reranker_weight: float,
    phrase_weight: float,
) -> dict[str, float]:
    metrics = []
    for question_id, question in questions.items():
        ranking = rank_ensemble(
            diagnostics[question_id],
            fusion_weight=fusion_weight,
            reranker_weight=reranker_weight,
            phrase_weight=phrase_weight,
        )
        metrics.append(score(question, ranking, 5))
    count = len(metrics)
    return {
        "evidence_recall_at_5": sum(item[0] for item in metrics) / count,
        "mrr_at_5": sum(item[1] for item in metrics) / count,
        "all_evidence_hit_at_5": sum(item[2] for item in metrics) / count,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("evaluation/results/benchmark-v2-development.json"),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evaluation/benchmark_v2.jsonl"),
    )
    parser.add_argument("--split", default="development")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("evaluation/results/benchmark-v2-development"),
    )
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    questions = {
        item["id"]: item
        for item in load_jsonl(args.dataset)
        if item["split"] == args.split and item["answerable"]
    }
    diagnostics = {
        item["question_id"]: item
        for item in report["diagnostics"]
        if item["question_id"] in questions
    }

    ablation_settings = [
        ("cross_encoder_only", 0.0, 1.0, 0.0),
        ("hybrid_plus_cross_encoder", 0.5, 4.0, 0.0),
        ("cross_encoder_plus_phrase", 0.0, 4.0, 0.005),
        ("hybrid_plus_phrase", 0.5, 0.0, 0.005),
        ("full_ensemble", 0.5, 4.0, 0.005),
    ]
    ablations = []
    for name, fusion_weight, reranker_weight, phrase_weight in ablation_settings:
        ablations.append({
            "variant": name,
            "fusion_weight": fusion_weight,
            "reranker_weight": reranker_weight,
            "phrase_weight": phrase_weight,
            **aggregate(
                questions,
                diagnostics,
                fusion_weight,
                reranker_weight,
                phrase_weight,
            ),
        })

    sensitivity = []
    for fusion_weight in (0.25, 0.5, 0.75):
        for reranker_weight in (3.0, 4.0, 5.0):
            for phrase_weight in (0.0025, 0.005, 0.0075):
                sensitivity.append({
                    "fusion_weight": fusion_weight,
                    "reranker_weight": reranker_weight,
                    "phrase_weight": phrase_weight,
                    **aggregate(
                        questions,
                        diagnostics,
                        fusion_weight,
                        reranker_weight,
                        phrase_weight,
                    ),
                })

    ablation_path = Path(f"{args.output_prefix}-ablation.csv")
    sensitivity_path = Path(f"{args.output_prefix}-weight-sensitivity.csv")
    write_csv(ablation_path, ablations)
    write_csv(sensitivity_path, sensitivity)
    print(json.dumps({
        "ablation_path": str(ablation_path),
        "sensitivity_path": str(sensitivity_path),
        "ablations": ablations,
        "sensitivity_runs": len(sensitivity),
        "perfect_recall_runs": sum(
            row["evidence_recall_at_5"] == 1.0 for row in sensitivity
        ),
        "perfect_all_hit_runs": sum(
            row["all_evidence_hit_at_5"] == 1.0 for row in sensitivity
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
