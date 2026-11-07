"""Qdrant client creation and connectivity checks."""

from functools import lru_cache
from urllib.parse import urlsplit, urlunsplit

from qdrant_client import QdrantClient

from app.config import get_settings


@lru_cache
def get_qdrant_client() -> QdrantClient:
    """Create and reuse one Qdrant client per process."""

    settings = get_settings()
    api_key = (
        settings.qdrant_api_key.get_secret_value()
        if settings.qdrant_api_key is not None
        else None
    )
    # Docker's local service is bound to IPv4. On Windows, localhost can try
    # IPv6 first and add a connection delay to every paginated corpus read.
    url = urlsplit(settings.qdrant_url)
    if url.scheme == "http" and url.hostname == "localhost" and url.username is None:
        authority = "127.0.0.1" + (f":{url.port}" if url.port is not None else "")
        url = url._replace(netloc=authority)
    return QdrantClient(
        url=urlunsplit(url),
        api_key=api_key,   # None = no auth (local Docker); str = Qdrant Cloud
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
