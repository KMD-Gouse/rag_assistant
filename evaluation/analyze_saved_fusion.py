"""Evaluate RRF weights and page caps from a saved Week 5 diagnostic."""

import argparse
import csv
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def evidence_groups(question: dict) -> list[set[tuple[str, int]]]:
    grouped: dict[str, set[tuple[str, int]]] = {}
    for index, evidence in enumerate(question["evidence"]):
        key = evidence.get("group") or f"__{index}"
        grouped.setdefault(key, set()).add(
            (evidence["doc_id"], evidence["pdf_page"])
        )
    return list(grouped.values())


def weighted_rrf(
    diagnostic: dict,
    dense_weight: float,
    bm25_weight: float,
    rrf_k: float,
    page_cap: int | None,
) -> list[dict]:
    payloads = {
        str(item["point_id"]): item["payload"]
        for item in diagnostic["fused_candidates"]
    }
    scores: dict[str, float] = {}
    for weight, ids in (
        (dense_weight, diagnostic["dense_ids"]),
        (bm25_weight, diagnostic["bm25_ids"]),
    ):
        for rank, point_id in enumerate(ids, start=1):
            point_id = str(point_id)
            if point_id in payloads:
                scores[point_id] = scores.get(point_id, 0.0) + weight / (
                    rrf_k + rank
                )
    ordered = sorted(scores, key=lambda point_id: (-scores[point_id], point_id))
    if page_cap is None:
        return [payloads[point_id] for point_id in ordered]

    page_counts: dict[tuple[str, int], int] = {}
    diversified = []
    for point_id in ordered:
        payload = payloads[point_id]
        page = (payload["doc_id"], payload["pdf_page"])
        if page_counts.get(page, 0) >= page_cap:
            continue
        page_counts[page] = page_counts.get(page, 0) + 1
        diversified.append(payload)
    return diversified


def score(question: dict, ranking: list[dict], k: int) -> tuple[float, float, float]:
    retrieved = [
        (item["doc_id"], item["pdf_page"])
        for item in ranking[:k]
    ]
    ranks = []
    for group in evidence_groups(question):
        rank = next(
            (
                index
                for index, location in enumerate(retrieved, start=1)
                if location in group
            ),
            None,
        )
        ranks.append(rank)
    recall = sum(rank is not None for rank in ranks) / len(ranks)
    relevant_ranks = [rank for rank in ranks if rank is not None]
    mrr = 1 / min(relevant_ranks) if relevant_ranks else 0.0
    return recall, mrr, float(all(rank is not None for rank in ranks))


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
    parser.add_argument("--rrf-k", type=float, default=60.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "evaluation/results/benchmark-v2-development-fusion-sensitivity.csv"
        ),
    )
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    questions = {
        item["id"]: item
        for item in load_jsonl(args.dataset)
        if item["split"] == "development" and item["answerable"]
    }
    diagnostics = {
        item["question_id"]: item
        for item in report["diagnostics"]
        if item["question_id"] in questions
    }

    rows = []
    for dense_weight in (0.0, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 1.0):
        for page_cap in (None, 1, 2):
            metrics = []
            pool_recalls = []
            for question_id, question in questions.items():
                ranking = weighted_rrf(
                    diagnostics[question_id],
                    dense_weight=dense_weight,
                    bm25_weight=1.0,
                    rrf_k=args.rrf_k,
                    page_cap=page_cap,
                )
                metrics.append(score(question, ranking, 5))
                pool_recalls.append(score(question, ranking, 50)[0])
            count = len(metrics)
            rows.append({
                "dense_weight": dense_weight,
                "bm25_weight": 1.0,
                "page_cap": page_cap or "none",
                "evidence_recall_at_5": sum(item[0] for item in metrics) / count,
                "mrr_at_5": sum(item[1] for item in metrics) / count,
                "all_evidence_hit_at_5": sum(item[2] for item in metrics) / count,
                "pool_recall_at_50": sum(pool_recalls) / count,
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} configurations to {args.output}")


if __name__ == "__main__":
    main()
