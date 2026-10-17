"""Qdrant client creation and connectivity checks."""

from functools import lru_cache

from qdrant_client import QdrantClient

from app.config import get_settings


@lru_cache
def get_qdrant_client() -> QdrantClient:
    """Create and reuse one Qdrant client per process."""

    settings = get_settings()

    return QdrantClient(
        url=settings.qdrant_url,
        timeout=5.0,
    )


def check_qdrant_connection() -> bool:
    """Confirm that the Qdrant server responds successfully."""

    client = get_qdrant_client()
    client.get_collections()
    return True


if __name__ == "__main__":
    check_qdrant_connection()
    print("Successfully connected to Qdrant.")