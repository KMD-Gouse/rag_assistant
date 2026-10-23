"""Environment-backed application configuration."""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Configuration loaded from environment variables and .env."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "annual_report_chunks"

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"

    groq_api_key: SecretStr | None = None
    groq_model: str = "openai/gpt-oss-20b"
    groq_api_url: str = "https://api.groq.com/openai/v1/chat/completions"
    groq_timeout_seconds: float = 30.0

    rag_top_k: int = 5
    rag_candidate_limit: int = 50
    rag_min_evidence_sufficiency: float = 0.70
    rag_reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rag_reranker_revision: str = "233902d25c440f23af6f7d6e94d2946bac0bee0a"


@lru_cache
def get_settings() -> Settings:
    """Return one shared settings instance per process."""

    return Settings()
