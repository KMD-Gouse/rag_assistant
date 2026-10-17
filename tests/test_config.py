from app.config import Settings


def test_default_settings() -> None:
    settings = Settings(_env_file=None)

    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.qdrant_collection == "annual_report_chunks"
    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
    assert settings.reranker_model == "BAAI/bge-reranker-base"