import pytest

from app.qdrant import check_qdrant_connection


@pytest.mark.parametrize("configured,expected", [
    ("http://localhost:6333", "http://127.0.0.1:6333"),
    ("http://localhost:6333/prefix", "http://127.0.0.1:6333/prefix"),
    ("http://127.0.0.1:6333", "http://127.0.0.1:6333"),
    ("https://localhost:6333", "https://localhost:6333"),
    ("https://example.qdrant.io:6333", "https://example.qdrant.io:6333"),
])
def test_local_docker_uses_ipv4_without_changing_remote_hosts(monkeypatch, configured, expected):
    from types import SimpleNamespace
    from app import qdrant

    captured = {}
    monkeypatch.setattr(qdrant, "get_settings", lambda: SimpleNamespace(
        qdrant_url=configured, qdrant_api_key=None,
    ))
    monkeypatch.setattr(qdrant, "QdrantClient", lambda **kwargs: captured.update(kwargs))
    qdrant.get_qdrant_client.cache_clear()
    try:
        qdrant.get_qdrant_client()
        assert captured["url"] == expected
        assert captured["timeout"] == 5.0
    finally:
        qdrant.get_qdrant_client.cache_clear()


@pytest.mark.integration
def test_qdrant_connection() -> None:
    assert check_qdrant_connection() is True
