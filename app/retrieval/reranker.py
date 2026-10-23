"""Cross-encoder reranking of a fixed hybrid candidate pool."""

from functools import lru_cache
import re

import numpy as np

from app.config import get_settings


ENSEMBLE_RRF_K = 60
ENSEMBLE_FUSION_WEIGHT = 0.5
ENSEMBLE_RERANK_WEIGHT = 4.0
ENSEMBLE_PHRASE_WEIGHT = 0.005
PHRASE_STOPWORDS = {
    "a", "an", "and", "as", "at", "by", "did", "each", "for", "from",
    "how", "in", "it", "its", "of", "on", "report", "reported", "the",
    "to", "was", "were", "what", "which",
}


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


def _content_bigrams(text: str) -> set[tuple[str, str]]:
    tokens = re.findall(r"[a-z0-9]+", text.casefold())
    return {
        (tokens[index], tokens[index + 1])
        for index in range(len(tokens) - 1)
        if tokens[index] not in PHRASE_STOPWORDS
        and tokens[index + 1] not in PHRASE_STOPWORDS
    }


def ensemble_rank(question, fused, reranked, limit=5):
    """Blend retrieval and cross-encoder ranks with exact phrase evidence."""

    if not question.strip() or limit < 1:
        raise ValueError("A non-empty question and positive limit are required.")
    fused_by_id = {str(point.id): point for point in fused}
    fused_ranks = {str(point.id): rank for rank, point in enumerate(fused, start=1)}
    reranked_ranks = {str(point.id): rank for rank, point in enumerate(reranked, start=1)}
    query_bigrams = _content_bigrams(question)
    scored = []
    for point_id, point in fused_by_id.items():
        rerank_position = reranked_ranks.get(point_id)
        if rerank_position is None:
            continue
        text = (point.payload or {}).get("text", "")
        phrase_matches = len(query_bigrams.intersection(_content_bigrams(text)))
        score = (
            ENSEMBLE_FUSION_WEIGHT / (ENSEMBLE_RRF_K + fused_ranks[point_id])
            + ENSEMBLE_RERANK_WEIGHT / (ENSEMBLE_RRF_K + rerank_position)
            + ENSEMBLE_PHRASE_WEIGHT * phrase_matches
        )
        scored.append((score, point_id, point))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [
        point.model_copy(update={"score": score})
        for score, _, point in scored[:limit]
    ]


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
        reranked = self.reranker.rerank(question, candidates, len(candidates) or limit)
        return ensemble_rank(question, candidates, reranked, limit)
