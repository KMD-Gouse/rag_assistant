"""Compare v1 dense, BM25 alone, and v2 RRF on a fixed dataset split."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean

from evaluation.metrics import grouped_document_recall_at_k, grouped_evidence_recall_at_k
from evaluation.validate_dataset import QUESTIONS_PATH, validate_dataset
from app.retrieval.hybrid import reciprocal_rank_fusion


def retrieval_query(question):
    """Use supplied conversation history consistently; never gold evidence."""
    return "\n".join([f"{message.role}: {message.content}" for message in question.history]
                     + [question.question])


def evidence_groups(question):
    """Return required facts as sets of interchangeable evidence locations."""

    grouped = {}
    for index, item in enumerate(question.evidence):
        grouped.setdefault(item.group or f"__location_{index}", set()).add(
            (item.doc_id, item.pdf_page)
        )
    return list(grouped.values())


def score_results(question, results, k):
    groups = evidence_groups(question)
    expected = set().union(*groups) if groups else set()
    retrieved = [(point.payload["doc_id"], point.payload["pdf_page"])
                 for point in results[:k]]
    recall = grouped_evidence_recall_at_k(groups, retrieved, k)
    first_relevant = next(
        (1.0 / rank for rank, key in enumerate(retrieved, 1)
         if any(key in group for group in groups)),
        0.0,
    )
    return {
        "evidence_recall": recall,
        "document_recall": grouped_document_recall_at_k(groups, retrieved, k),
        "mrr": first_relevant if groups else None,
        "all_evidence_hit": float(recall == 1.0) if expected else None,
    }


def evaluate(questions, dense_search, bm25, k=5, candidates=50, rrf_k=60):
    if k < 1 or candidates < k:
        raise ValueError("Require candidates >= k >= 1.")
    rows = []
    for question in questions:
        query = retrieval_query(question)
        dense = dense_search(query, limit=candidates)
        sparse = bm25.search(query, limit=candidates)
        rankings = {
            "v1_dense": dense[:k],
            "bm25": sparse[:k],
            "v2_hybrid": reciprocal_rank_fusion([dense, sparse], k, rrf_k),
        }
        for method, results in rankings.items():
            retrieved_keys = {
                (p.payload["doc_id"], p.payload["pdf_page"])
                for p in results
            }
            missing_groups = [
                group for group in evidence_groups(question)
                if not group.intersection(retrieved_keys)
            ]
            rows.append({
                "question_id": question.id, "type": question.type,
                "answerable": question.answerable, "method": method, "query": query,
                "metrics": score_results(question, results, k),
                "retrieved": [{"point_id": str(p.id), "score": p.score,
                               "payload": p.payload} for p in results],
                "missing_evidence": [
                    list(key) for group in missing_groups for key in sorted(group)
                ],
                "missing_evidence_groups": [
                    [list(key) for key in sorted(group)] for group in missing_groups
                ],
            })
    summaries = []
    for method in ("v1_dense", "bm25", "v2_hybrid"):
        for group in ["all"] + sorted({q.type for q in questions}):
            subset = [r for r in rows if r["method"] == method
                      and (group == "all" or r["type"] == group)]
            summary = {"method": method, "group": group, "questions": len(subset),
                       "answerable": sum(r["answerable"] for r in subset)}
            for metric in ("evidence_recall", "document_recall", "mrr", "all_evidence_hit"):
                values = [r["metrics"][metric] for r in subset if r["metrics"][metric] is not None]
                summary[metric] = mean(values) if values else None
            summaries.append(summary)
    return {"summary": summaries, "questions": rows}


def render_table(report):
    lines = ["| Method | Group | Answerable / total | Evidence recall | Document recall | MRR | All evidence hit |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in report["summary"]:
        values = ["N/A" if row[key] is None else f"{row[key]:.4f}" for key in
                  ("evidence_recall", "document_recall", "mrr", "all_evidence_hit")]
        lines.append(f"| {row['method']} | {row['group']} | {row['answerable']} / {row['questions']} | "
                     + " | ".join(values) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "held_out"], default="dev")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=50)
    parser.add_argument("--rrf-k", type=float, default=60)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        args.output = Path(f"evaluation/results/week4-{args.split}.json")
    if args.k < 1 or args.candidates < args.k:
        parser.error("Require candidates >= k >= 1.")
    reciprocal_rank_fusion([], args.k, args.rrf_k)
    questions = [q for q in validate_dataset() if q.split == args.split]
    from app.config import get_settings
    from app.retrieval.bm25 import BM25Index
    from app.retrieval.dense import search_chunks

    index = BM25Index.from_qdrant()
    # Reject incomplete collections instead of presenting misleading scores.
    from evaluation.validate_dataset import load_manifest_documents
    actual_docs = {(r.payload or {}).get("doc_id") for r in index.records}
    if actual_docs != set(load_manifest_documents()):
        raise ValueError("Collection documents differ from the manifest; index all reports first.")
    report = evaluate(questions, search_chunks, index, args.k, args.candidates, args.rrf_k)
    report["metadata"] = {
        "created_at": datetime.now(timezone.utc).isoformat(), "split": args.split,
        "k": args.k, "candidates_per_method": args.candidates, "rrf_k": args.rrf_k,
        "bm25_k1": index.k1, "bm25_b": index.b,
        "dataset_sha256": hashlib.sha256(QUESTIONS_PATH.read_bytes()).hexdigest(),
        "corpus_sha256": index.fingerprint, "corpus_chunks": len(index.records),
        "embedding_model": get_settings().embedding_model,
        "collection": get_settings().qdrant_collection,
        "query_policy": "Supplied history followed by question; no gold metadata filters.",
        "metric_unit": "Evidence is (doc_id, pdf_page); k counts chunks, including repeated pages.",
    }
    table = render_table(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.output.with_suffix(".md").write_text(
        f"# Week 4 retrieval evaluation: {args.split}, k={args.k}\n\n{table}\n\n"
        "Macro averages exclude unanswerable questions. These scores measure page retrieval, "
        "not answer correctness or abstention. See the JSON for configuration and per-question misses.\n",
        encoding="utf-8",
    )
    print(table)
    print(f"Saved report: {args.output}")


if __name__ == "__main__":
    main()

