"""Retrieval CLI; answer generation is a later milestone."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--method", choices=["dense", "bm25", "hybrid", "reranked"], default="hybrid")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=50)
    parser.add_argument("--rrf-k", type=float, default=60)
    parser.add_argument("--reranker-model")
    parser.add_argument("--reranker-revision")
    parser.add_argument("--rerank-candidates", type=int, default=50)
    parser.add_argument("--doc-id")
    parser.add_argument("--company")
    parser.add_argument("--fiscal-year")
    args = parser.parse_args()
    from app.retrieval.dense import print_results, search_chunks
    from app.retrieval.bm25 import BM25Index
    from app.retrieval.hybrid import HybridRetriever

    search = search_chunks
    if args.method != "dense":
        index = BM25Index.from_qdrant()
        search = index.search if args.method == "bm25" else HybridRetriever(
            index, args.candidates, args.rrf_k
        ).search
        if args.method == "reranked":
            from app.retrieval.reranker import Reranker, RerankedRetriever
            search = RerankedRetriever(
                HybridRetriever(index, args.candidates, args.rrf_k),
                Reranker(args.reranker_model, revision=args.reranker_revision), args.rerank_candidates,
            ).search
    from app.retrieval.scope import inferred_filters
    automatic = inferred_filters(args.question)
    results = search(
        args.question,
        limit=args.limit,
        doc_id=args.doc_id,
        company=args.company or automatic["company"],
        fiscal_year=args.fiscal_year or automatic["fiscal_year"],
    )
    print_results(args.question, results)


if __name__ == "__main__":
    main()

