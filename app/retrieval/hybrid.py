"""Reciprocal-rank fusion without comparing incompatible raw scores."""

from math import isfinite

from app.retrieval.bm25 import BM25Index


def reciprocal_rank_fusion(rankings, limit: int = 5, rrf_k: float = 60):
    if limit < 1 or not isfinite(rrf_k) or rrf_k < 0:
        raise ValueError("Positive limit and finite nonnegative rrf_k required.")
    scores, points = {}, {}
    for ranking in rankings:
        seen = set()
        for rank, point in enumerate(ranking, start=1):
            key = str(point.id)
            if key in seen:
                continue
            seen.add(key)
            points.setdefault(key, point)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
    ordered = sorted(scores, key=lambda key: (-scores[key], key))[:limit]
    return [points[key].model_copy(update={"score": scores[key]}) for key in ordered]


class HybridRetriever:
    def __init__(self, bm25: BM25Index, candidate_limit: int = 50, rrf_k: float = 60):
        if candidate_limit < 1 or not isfinite(rrf_k) or rrf_k < 0:
            raise ValueError("Invalid fusion parameters.")
        self.bm25 = bm25
        self.candidate_limit = candidate_limit
        self.rrf_k = rrf_k

    def search(self, question: str, limit: int = 5, **filters):
        from app.retrieval.dense import search_chunks

        if limit < 1 or limit > self.candidate_limit:
            raise ValueError("limit must be between 1 and candidate_limit.")
        dense = search_chunks(question, limit=self.candidate_limit, **filters)
        sparse = self.bm25.search(question, limit=self.candidate_limit, **filters)
        return reciprocal_rank_fusion([dense, sparse], limit, self.rrf_k)
