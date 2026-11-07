from concurrent.futures import ThreadPoolExecutor
import sqlite3

import pytest

from app.config import Settings
from app.generation.models import AnswerResult, ValidatedCitation
from app.memory import HistoryRepository, SQLiteHistoryRepository


def answer(question="What was TCS revenue in FY2025?", *, abstained=False):
    return AnswerResult(
        question=question, answer="I don't know" if abstained else "Revenue was ₹100.",
        abstained=abstained,
        abstention_reason="no_retrieval_results" if abstained else None,
        evidence_sufficiency=0.0 if abstained else 1.0, threshold=1.0,
        citations=[] if abstained else [ValidatedCitation(
            evidence_id="E1", chunk_id="chunk-1", doc_id="tcs_fy2025",
            company="TCS", fiscal_year="2024-25", pdf_page=10,
            printed_page="8", quote="Revenue was ₹100.",
        )],
        retrieved_chunks=0 if abstained else 5, model="test-model",
    )


@pytest.fixture
def repo(tmp_path):
    return SQLiteHistoryRepository(tmp_path / "nested" / "history.sqlite3")


@pytest.mark.parametrize("abstained", [False, True])
def test_complete_turn_survives_reopening(repo, abstained):
    repository: HistoryRepository = repo
    result = answer(abstained=abstained)
    saved = repository.append_turn("session-1", "And this year?", result)
    reopened = SQLiteHistoryRepository(repo.db_path)
    assert reopened.get_history("session-1") == [saved]
    assert saved.original_question == "And this year?"
    assert saved.rewritten_question == result.question
    assert saved.result == result
    assert saved.created_at.utcoffset().total_seconds() == 0
    # Stored and returned data must not alias the caller's mutable list.
    result.citations.clear()
    assert reopened.get_history("session-1")[0] == saved


def test_recent_window_is_chronological_and_session_scoped(repo):
    saved = []
    for index in range(12):
        saved.append(repo.append_turn("a", f"Question {index}", answer()))
        repo.append_turn("b", "Unrelated", answer())
    assert repo.get_history("a") == saved[-10:]
    assert repo.get_history("a", limit=3) == saved[-3:]
    assert repo.get_history("a", limit=1) == saved[-1:]
    assert repo.get_history("a", limit=100) == saved
    assert repo.get_history("unknown") == []


def test_clear_is_isolated_and_ids_are_literal(repo):
    ids = ["Session", "session", "x'; DROP TABLE history_turns; --"]
    for session_id in ids:
        repo.append_turn(session_id, "Question", answer())
    assert repo.clear_session(ids[2]) == 1
    assert repo.clear_session(ids[2]) == 0
    assert repo.clear_session("unknown") == 0
    assert repo.get_history(ids[2]) == []
    for session_id in ids[:2]:
        assert len(repo.get_history(session_id)) == 1


@pytest.mark.parametrize("session_id", ["", " \n", None, 7])
def test_invalid_session_ids(repo, session_id):
    with pytest.raises(ValueError):
        repo.append_turn(session_id, "Question", answer())
    with pytest.raises(ValueError):
        repo.get_history(session_id)
    with pytest.raises(ValueError):
        repo.clear_session(session_id)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "2", None])
def test_invalid_history_limit(repo, limit):
    with pytest.raises(ValueError, match="positive integer"):
        repo.get_history("session", limit=limit)


def test_blank_questions_never_write(repo):
    with pytest.raises(ValueError):
        repo.append_turn("session", "  ", answer())
    with pytest.raises(ValueError):
        repo.append_turn("session", "Question", answer(" "))
    assert repo.get_history("session") == []


def test_failed_write_rolls_back_and_repository_remains_usable(repo):
    connection = sqlite3.connect(repo.db_path)
    try:
        connection.execute("""CREATE TRIGGER reject_turn BEFORE INSERT ON history_turns
                              BEGIN SELECT RAISE(ABORT, 'test failure'); END""")
        connection.commit()
        with pytest.raises(sqlite3.IntegrityError, match="test failure"):
            repo.append_turn("session", "Question", answer())
        assert repo.get_history("session") == []
        connection.execute("DROP TRIGGER reject_turn")
        connection.commit()
    finally:
        connection.close()
    repo.append_turn("session", "Question", answer())
    assert len(repo.get_history("session")) == 1


def test_concurrent_instances_do_not_lose_turns(repo):
    other = SQLiteHistoryRepository(repo.db_path)

    def append(index):
        target = repo if index % 2 else other
        return target.append_turn(f"session-{index % 3}", f"Question {index}", answer())

    with ThreadPoolExecutor(max_workers=4) as pool:
        saved = list(pool.map(append, range(24)))
    assert len({turn.id for turn in saved}) == 24
    for index in range(3):
        session_id = f"session-{index}"
        expected = sorted((t for t in saved if t.session_id == session_id), key=lambda t: t.id)
        assert repo.get_history(session_id) == expected


@pytest.mark.parametrize("path", ["", ":memory:"])
def test_requires_file_backed_database(path):
    with pytest.raises(ValueError, match="persistent SQLite file"):
        SQLiteHistoryRepository(path)


def test_configured_database_path(monkeypatch, tmp_path):
    path = tmp_path / "configured.sqlite3"
    monkeypatch.setenv("HISTORY_DB_PATH", str(path))
    settings = Settings(_env_file=None)
    monkeypatch.setattr("app.memory.repository.get_settings", lambda: settings)
    assert SQLiteHistoryRepository().db_path == path.resolve()
