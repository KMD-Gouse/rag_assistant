import pytest

from app.qdrant import check_qdrant_connection


@pytest.mark.integration
def test_qdrant_connection() -> None:
    assert check_qdrant_connection() is True