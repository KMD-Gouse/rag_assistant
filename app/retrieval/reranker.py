"""Cross-encoder reranking of a fixed hybrid candidate pool."""

from functools import lru_cache

import numpy as np

from app.config import get_settings


@lru_cache(maxsize=2)
def load_reranker(model_name: str, max_length: int = 512, revision: str | None = None):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, max_length=max_length, revision=revision)


class Reranker:
    """Pairwise relevance scoring; scores are not answer probabilities."""

    def __init__(self, model_name=None, batch_size=8, max_length=512, model=None, revision=None):
        if batch_size < 1 or max_length < 1:
            raise ValueError("batch_size and max_length must be positive.")
        self.model_name = model_name or get_settings().reranker_model
        self.batch_size = batch_size
        self.max_length = max_length
        self.model = model
        self.revision = revision

    def rerank(self, question, candidates, limit=5):
        if not question.strip() or limit < 1:
            raise ValueError("A non-empty question and positive limit are required.")
        # Deduplicate by point ID, retaining the first candidate's payload.
        unique = {}
        for point in candidates:
            unique.setdefault(str(point.id), point)
        points = list(unique.values())
        if not points:
            return []
        texts = [(p.payload or {}).get("text") for p in points]
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Every candidate must contain non-empty text.")
        if self.model is None:
            self.model = load_reranker(self.model_name, self.max_length, self.revision)
        passages = []
        for point, text in zip(points, texts, strict=True):
            payload = point.payload or {}
            passages.append(
                f"Company: {payload.get('company', 'unknown')}\n"
                f"Fiscal year: {payload.get('fiscal_year', 'unknown')}\n"
                f"Document: {payload.get('doc_id', 'unknown')}\n{text}"
            )
        scores = np.asarray(self.model.predict(
            [(question, passage) for passage in passages],
            batch_size=self.batch_size, show_progress_bar=False,
            convert_to_numpy=True,
        ), dtype=float).reshape(-1)
        if len(scores) != len(points) or not np.isfinite(scores).all():
            raise ValueError("Reranker must produce one finite score per candidate.")
        order = sorted(range(len(points)), key=lambda i: (-scores[i], str(points[i].id)))
        return [points[i].model_copy(update={"score": float(scores[i])})
                for i in order[:limit]]


class RerankedRetriever:
    def __init__(self, hybrid, reranker, rerank_candidates=50):
        if not 1 <= rerank_candidates <= hybrid.candidate_limit:
            raise ValueError("Rerank pool must fit within the hybrid candidate limit.")
        self.hybrid = hybrid
        self.reranker = reranker
        self.rerank_candidates = rerank_candidates

    def search(self, question, limit=5, **filters):
        if not 1 <= limit <= self.rerank_candidates:
            raise ValueError("Final limit must fit within the rerank pool.")
        candidates = self.hybrid.search(question, limit=self.rerank_candidates, **filters)
        return self.reranker.rerank(question, candidates, limit)
