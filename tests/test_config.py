import pytest

from app.config import Settings


def test_default_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    # _env_file=None disables .env, but process environment still takes priority.
    for key in (
        "QDRANT_URL", "QDRANT_COLLECTION", "EMBEDDING_MODEL", "RERANKER_MODEL",
        "GROQ_MODEL", "RAG_MIN_EVIDENCE_SUFFICIENCY",
    ):
        monkeypatch.delenv(key, raising=False)
    settings = Settings(_env_file=None)

    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.qdrant_collection == "annual_report_chunks"
    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
    assert settings.reranker_model == "BAAI/bge-reranker-base"
    assert settings.groq_model == "openai/gpt-oss-20b"
    assert settings.rag_min_evidence_sufficiency == 1.0
    assert settings.rag_reranker_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
