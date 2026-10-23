"""Four-method ablation, candidate coverage, latency, and failure diagnostics."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from app.retrieval.hybrid import reciprocal_rank_fusion
from app.retrieval.reranker import (
    ENSEMBLE_FUSION_WEIGHT,
    ENSEMBLE_PHRASE_WEIGHT,
    ENSEMBLE_RERANK_WEIGHT,
    ENSEMBLE_RRF_K,
    ensemble_rank,
)
from evaluation.run_eval import evidence_groups, retrieval_query, score_results, render_table
from evaluation.validate_dataset import QUESTIONS_PATH, validate_dataset


METHODS = ("v1_dense", "bm25", "v2_hybrid", "v3_reranked")
METRICS = ("evidence_recall", "document_recall", "mrr", "all_evidence_hit")


def evidence_keys(points):
    return {(p.payload["doc_id"], p.payload["pdf_page"]) for p in points}


def diagnose(question, corpus_pages, union, pool, final):
    """Locate page-level losses; these labels do not prove semantic root causes."""
    stages = [corpus_pages, evidence_keys(union), evidence_keys(pool), evidence_keys(final)]
    labels = ["corpus_page_missing", "candidate_miss", "fusion_cutoff", "reranker_top_k_miss"]
    diagnosed = []
    for group in evidence_groups(question):
        stage = next(
            (label for keys, label in zip(stages, labels) if not group.intersection(keys)),
            "hit",
        )
        first_doc, first_page = sorted(group)[0]
        diagnosed.append({
            "doc_id": first_doc,
            "pdf_page": first_page,
            "alternatives": [list(key) for key in sorted(group)],
            "stage": stage,
        })
    return diagnosed


def serialize(points):
    return [{"point_id": str(p.id), "score": p.score, "payload": p.payload} for p in points]


def evaluate_week5(questions, dense_search, bm25, reranker,
                   k=5, candidates=50, pool_size=50, rrf_k=60,
                   auto_scope=False):
    if not 1 <= k <= pool_size <= candidates:
        raise ValueError("Require 1 <= k <= rerank pool <= candidates.")
    reciprocal_rank_fusion([], k, rrf_k)
    corpus_pages = evidence_keys(bm25.records)
    rows, diagnostics = [], []
    for question in questions:
        query = retrieval_query(question)
        filters = {}
        if auto_scope:
            from app.retrieval.scope import inferred_filters
            filters = {key: value for key, value in inferred_filters(query).items() if value}
        start = perf_counter()
        dense = dense_search(query, limit=candidates, **filters)
        dense_ms = (perf_counter() - start) * 1000
        start = perf_counter()
        sparse = bm25.search(query, limit=candidates, **filters)
        sparse_ms = (perf_counter() - start) * 1000
        start = perf_counter()
        union = reciprocal_rank_fusion([dense, sparse], 2 * candidates, rrf_k)
        pool = union[:pool_size]
        fusion_ms = (perf_counter() - start) * 1000
        start = perf_counter()
        cross_encoder_ranked = reranker.rerank(query, pool, limit=len(pool) or k)
        reranked = ensemble_rank(query, pool, cross_encoder_ranked, limit=len(pool) or k)
        rerank_ms = (perf_counter() - start) * 1000
        rankings = dict(zip(METHODS, [dense[:k], sparse[:k], pool[:k], reranked[:k]]))
        latencies = dict(zip(METHODS, [dense_ms, sparse_ms,
                         dense_ms + sparse_ms + fusion_ms,
                         dense_ms + sparse_ms + fusion_ms + rerank_ms]))
        for method, results in rankings.items():
            rows.append({"question_id": question.id, "type": question.type,
                         "answerable": question.answerable, "method": method,
                         "query": query, "metrics": score_results(question, results, k),
                         "latency_ms": latencies[method], "retrieved": serialize(results)})
        v2_keys, v3_keys = evidence_keys(pool[:k]), evidence_keys(reranked[:k])
        groups = evidence_groups(question)
        v2_group_hits = {index for index, group in enumerate(groups) if group.intersection(v2_keys)}
        v3_group_hits = {index for index, group in enumerate(groups) if group.intersection(v3_keys)}
        diagnostics.append({
            "question_id": question.id, "type": question.type, "query": query,
            "inferred_filters": filters,
            "evidence": [e.model_dump() for e in question.evidence],
            "failure_stages": diagnose(question, corpus_pages, union, pool, reranked[:k]),
            "union_evidence_recall": score_results(question, union, max(1, len(union)))["evidence_recall"],
            "pool_evidence_recall": score_results(question, pool, max(1, len(pool)))["evidence_recall"],
            "gained_pages": [
                list(key)
                for index in sorted(v3_group_hits - v2_group_hits)
                for key in sorted(groups[index].intersection(v3_keys))
            ],
            "lost_pages": [
                list(key)
                for index in sorted(v2_group_hits - v3_group_hits)
                for key in sorted(groups[index].intersection(v2_keys))
            ],
            "dense_ids": [str(p.id) for p in dense],
            "bm25_ids": [str(p.id) for p in sparse],
            "fused_candidates": serialize(union),
            "cross_encoder_pool": [{"point_id": str(p.id), "score": p.score} for p in cross_encoder_ranked],
            "reranked_pool": [{"point_id": str(p.id), "score": p.score} for p in reranked],
            "stage_ms": {"dense": dense_ms, "bm25": sparse_ms,
                         "fusion": fusion_ms, "reranker": rerank_ms},
        })
        print(f"Evaluated {question.id}", flush=True)
    summaries = []
    for method in METHODS:
        for group in ["all"] + sorted({q.type for q in questions}):
            subset = [r for r in rows if r["method"] == method and
                      (group == "all" or r["type"] == group)]
            summary = {"method": method, "group": group, "questions": len(subset),
                       "answerable": sum(r["answerable"] for r in subset)}
            for metric in METRICS:
                values = [r["metrics"][metric] for r in subset if r["metrics"][metric] is not None]
                summary[metric] = mean(values) if values else None
            timings = [r["latency_ms"] for r in subset]
            summary["median_latency_ms"] = median(timings) if timings else None
            summaries.append(summary)
    return {"summary": summaries, "questions": rows, "diagnostics": diagnostics}


def render_analysis(report):
    lines = ["# Week 5 retrieval ablation", "", render_table(report), "",
             "## Timing", "", "| Method | Median query latency (ms) |", "|---|---:|"]
    for row in report["summary"]:
        if row["group"] == "all":
            lines.append(f"| {row['method']} | {row['median_latency_ms']:.1f} |")
    lines += ["", "Timing excludes corpus loading, model loading and one warm-up query. "
              "Hybrid/v3 timings sum sequential stage times; these are single-run measurements, "
              "not a production throughput benchmark. All queries, including unanswerable ones, are timed.",
              "", "## Evidence loss stages", "", "| Stage | Evidence locations |", "|---|---:|"]
    counts = Counter(e["stage"] for d in report["diagnostics"] for e in d["failure_stages"])
    for stage in ("corpus_page_missing", "candidate_miss", "fusion_cutoff", "reranker_top_k_miss", "hit"):
        lines.append(f"| {stage} | {counts[stage]} |")
    for key in ("union_evidence_recall", "pool_evidence_recall"):
        values = [d[key] for d in report["diagnostics"] if d[key] is not None]
        if values:
            lines += ["", f"Macro {key}: {mean(values):.4f}."]
    lines += ["", "## Questions requiring review", ""]
    review_rows = []
    for item in report["diagnostics"]:
        misses = [f"{e['doc_id']} p{e['pdf_page']}: {e['stage']}" for e in item["failure_stages"] if e["stage"] != "hit"]
        if misses or item["lost_pages"]:
            review_rows.append(f"| {item['question_id']} | {item['type']} | {'; '.join(misses)} | "
                               f"{item['gained_pages']} | {item['lost_pages']} |")
    if review_rows:
        lines += ["| Question | Type | Missing evidence stages | Gained pages | Lost pages |",
                  "|---|---|---|---|---|", *review_rows]
    else:
        lines.append("No unresolved labeled-evidence losses on this split.")
    lines += ["", "Metrics use (document, PDF page) matches and exclude unanswerable questions. "
              "A page hit does not establish that the exact supporting fact was retrieved. "
              "Failure stages locate losses, not semantic causes: inspect the saved reference evidence "
              "and candidate text before labeling parsing, year confusion, truncation or lexical failures."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--dataset", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--auto-scope", action="store_true")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=50)
    parser.add_argument("--rerank-candidates", type=int, default=50)
    parser.add_argument("--rrf-k", type=float, default=60)
    parser.add_argument("--reranker-model")
    parser.add_argument("--reranker-revision", help="Pin a model commit for reproducible evaluation.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.k <= args.rerank_candidates <= args.candidates:
        parser.error("Require 1 <= k <= rerank-candidates <= candidates.")
    reciprocal_rank_fusion([], args.k, args.rrf_k)
    from app.config import get_settings
    from app.ingestion.embed import MODEL_NAME
    from app.retrieval.bm25 import BM25Index
    from app.retrieval.dense import search_chunks
    from app.retrieval.reranker import Reranker
    from evaluation.validate_dataset import load_manifest_documents

    if args.dataset.resolve() == QUESTIONS_PATH.resolve():
        all_questions = validate_dataset()
    else:
        from evaluation.schema import EvaluationQuestion
        all_questions = [
            EvaluationQuestion.model_validate_json(line)
            for line in args.dataset.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    questions = [q for q in all_questions if q.split == args.split]
    if not questions:
        parser.error(f"No questions use split {args.split!r} in {args.dataset}.")
    index = BM25Index.from_qdrant()
    documents = load_manifest_documents()
    if {(r.payload or {}).get("doc_id") for r in index.records} != set(documents):
        raise ValueError("Index all manifest reports before evaluation.")
    for record in index.records:
        payload = record.payload
        if payload.get("document_checksum") != documents[payload["doc_id"]]["sha256"].lower():
            raise ValueError("Indexed document checksum differs from manifest.")
    reranker = Reranker(args.reranker_model, args.batch_size, args.max_length,
                        revision=args.reranker_revision)
    warmup = "What did the company report?"
    warm_dense = search_chunks(warmup, limit=args.candidates)
    warm_sparse = index.search(warmup, limit=args.candidates)
    warm_pool = reciprocal_rank_fusion([warm_dense, warm_sparse], args.rerank_candidates, args.rrf_k)
    reranker.rerank(warmup, warm_pool, args.k)
    print("Warm-up complete; starting measured queries.", flush=True)
    report = evaluate_week5(questions, search_chunks, index, reranker,
                           args.k, args.candidates, args.rerank_candidates, args.rrf_k,
                           args.auto_scope)
    config = getattr(getattr(reranker.model, "model", None), "config", None)
    report["metadata"] = {
        "created_at": datetime.now(timezone.utc).isoformat(), "split": args.split,
        "k": args.k, "candidates_per_method": args.candidates,
        "rerank_candidates": args.rerank_candidates, "rrf_k": args.rrf_k,
        "reranker_model": reranker.model_name,
        "reranker_revision": args.reranker_revision or getattr(config, "_commit_hash", None),
        "device": str(reranker.model.device),
        "max_pair_tokens": args.max_length, "batch_size": args.batch_size,
        "embedding_model": MODEL_NAME, "bm25_k1": index.k1, "bm25_b": index.b,
        "corpus_sha256": index.fingerprint, "corpus_chunks": len(index.records),
        "dataset": str(args.dataset),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "collection": get_settings().qdrant_collection,
        "packages": {name: version(name) for name in ["sentence-transformers", "transformers", "torch", "qdrant-client"]},
        "query_policy": "Supplied history then question; no gold metadata filters.",
        "automatic_scope": args.auto_scope,
        "reranker_input": "Question/history paired with company, fiscal year, document and chunk text; model truncates long pairs.",
        "final_rank_ensemble": {
            "rrf_k": ENSEMBLE_RRF_K,
            "hybrid_rank_weight": ENSEMBLE_FUSION_WEIGHT,
            "cross_encoder_rank_weight": ENSEMBLE_RERANK_WEIGHT,
            "exact_content_bigram_weight": ENSEMBLE_PHRASE_WEIGHT,
        },
        "timing_policy": "Sequential stage sums after one warm-up; no corpus/model loading.",
    }
    default_prefix = "benchmark-v2" if args.dataset.resolve() != QUESTIONS_PATH.resolve() else "week5"
    output = args.output or Path(f"evaluation/results/{default_prefix}-{args.split}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output.with_suffix(".md").write_text(render_analysis(report), encoding="utf-8")
    print(render_table({"summary": [s for s in report["summary"] if s["group"] == "all"]}))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
