"""In-memory BM25 over a snapshot of the dense collection's payloads."""

from collections import Counter, defaultdict
import hashlib
import json
import math
import re

from qdrant_client import models

from app.config import get_settings
from app.qdrant import get_qdrant_client


def tokenize(text: str) -> list[str]:
    """Case-fold words; normalize commas inside financial numbers."""
    return re.findall(r"\w+", re.sub(r"(?<=\d),(?=\d)", "", text.casefold()))


class BM25Index:
    """Okapi BM25 with positive IDF, k1=1.5 and b=0.75 by default."""

    def __init__(self, records, k1: float = 1.5, b: float = 0.75):
        if not math.isfinite(k1) or k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 requires finite k1 > 0 and 0 <= b <= 1.")
        self.k1, self.b = k1, b
        self.records = sorted(records, key=lambda record: str(record.id))
        if len({str(record.id) for record in self.records}) != len(self.records):
            raise ValueError("Duplicate point IDs in corpus.")
        self.postings = defaultdict(list)
        self.lengths = []
        for index, record in enumerate(self.records):
            payload = record.payload or {}
            if not isinstance(payload.get("text"), str) or not payload["text"].strip():
                raise ValueError(f"Point {record.id} has no searchable text.")
            terms = Counter(tokenize(payload["text"]))
            self.lengths.append(sum(terms.values()))
            for term, frequency in terms.items():
                self.postings[term].append((index, frequency))
        self.average_length = sum(self.lengths) / max(1, len(self.records))
        serialized = [(str(r.id), r.payload) for r in self.records]
        self.fingerprint = hashlib.sha256(
            json.dumps(serialized, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()

    @classmethod
    def from_qdrant(cls):
        """Read all pages once. Rebuild after ingestion changes the collection."""
        client = get_qdrant_client()
        records, offset = [], None
        while True:
            batch, offset = client.scroll(
                collection_name=get_settings().qdrant_collection,
                limit=256, offset=offset, with_payload=True, with_vectors=False,
            )
            records.extend(batch)
            if offset is None:
                break
        if not records:
            raise ValueError("The Qdrant collection is empty; run ingestion first.")
        return cls(records)

    def search(self, question: str, limit: int = 5, doc_id=None,
               company=None, fiscal_year=None) -> list[models.ScoredPoint]:
        if not question.strip() or limit < 1:
            raise ValueError("A non-empty question and positive limit are required.")
        filters = {"doc_id": doc_id, "company": company, "fiscal_year": fiscal_year}
        scores = defaultdict(float)
        for term in sorted(set(tokenize(question))):
            postings = self.postings.get(term, [])
            idf = math.log(1 + (len(self.records) - len(postings) + 0.5)
                           / (len(postings) + 0.5))
            for index, frequency in postings:
                payload = self.records[index].payload or {}
                if any(
                    value is not None
                    and payload.get(key) not in ((value,) if isinstance(value, str) else value)
                    for key, value in filters.items()
                ):
                    continue
                norm = 1 - self.b + self.b * self.lengths[index] / self.average_length
                scores[index] += idf * frequency * (self.k1 + 1) / (frequency + self.k1 * norm)
        ranked = sorted(scores, key=lambda i: (-scores[i], str(self.records[i].id)))
        return [models.ScoredPoint(id=self.records[i].id, version=0,
                                   score=scores[i], payload=self.records[i].payload)
                for i in ranked[:limit]]
