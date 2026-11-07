"""Session-scoped history contract and durable SQLite implementation."""

from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from app.config import get_settings
from app.generation.models import AnswerResult


class HistoryTurn(BaseModel):
    """A completed turn; result.question is the rewritten retrieval question."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: int
    session_id: str
    original_question: str
    result: AnswerResult
    created_at: datetime

    @property
    def rewritten_question(self) -> str:
        return self.result.question


class HistoryRepository(Protocol):
    """Storage boundary for the follow-up rewriter and future API.

    Sessions are opaque, case-sensitive IDs supplied by the caller. Appending
    creates history implicitly. Reads/clears of unknown sessions are harmless.
    Implementations return the latest ``limit`` complete turns, oldest first.
    """

    def append_turn(
        self, session_id: str, original_question: str, result: AnswerResult,
    ) -> HistoryTurn: ...

    def get_history(self, session_id: str, *, limit: int = 10) -> list[HistoryTurn]: ...

    def clear_session(self, session_id: str) -> int: ...


def _nonblank(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonblank string")


class SQLiteHistoryRepository:
    """File-backed history, with a separate connection per operation.

    Safe to share between request threads; SQLite serializes short writes and
    waits up to five seconds for locks. Connections are always closed. Database
    errors propagate to callers; failed writes are rolled back, never hidden.
    This stores completed turns only, not in-flight requests or API retries.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        path = db_path if db_path is not None else get_settings().history_db_path
        if str(path) == ":memory:" or not str(path).strip():
            raise ValueError("db_path must name a persistent SQLite file")
        self.db_path = Path(path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection, connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS history_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    original_question TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            connection.execute("""
                CREATE INDEX IF NOT EXISTS history_turns_session_id_id
                ON history_turns (session_id, id)
            """)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        return connection

    def append_turn(
        self, session_id: str, original_question: str, result: AnswerResult,
    ) -> HistoryTurn:
        """Atomically persist a question and its complete answer, including citations."""
        _nonblank(session_id, "session_id")
        _nonblank(original_question, "original_question")
        # Round-trip validation also snapshots mutable nested citation lists.
        result = AnswerResult.model_validate_json(result.model_dump_json())
        _nonblank(result.question, "result.question")
        created_at = datetime.now(timezone.utc)
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                """INSERT INTO history_turns
                   (session_id, original_question, result_json, created_at)
                   VALUES (?, ?, ?, ?)""",
                (session_id, original_question, result.model_dump_json(), created_at.isoformat()),
            )
            turn = HistoryTurn(
                id=cursor.lastrowid, session_id=session_id,
                original_question=original_question, result=result, created_at=created_at,
            )
        return turn

    def get_history(self, session_id: str, *, limit: int = 10) -> list[HistoryTurn]:
        """Return the most recent turns in insertion order (not timestamp order)."""
        _nonblank(session_id, "session_id")
        if type(limit) is not int or limit < 1:
            raise ValueError("limit must be a positive integer")
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """SELECT id, session_id, original_question, result_json, created_at
                   FROM history_turns WHERE session_id = ? ORDER BY id DESC LIMIT ?""",
                (session_id, limit),
            ).fetchall()
        return [
            HistoryTurn(
                id=row["id"], session_id=row["session_id"],
                original_question=row["original_question"],
                result=AnswerResult.model_validate_json(row["result_json"]),
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in reversed(rows)
        ]

    def clear_session(self, session_id: str) -> int:
        """Delete only this session's history and return the number of deleted turns."""
        _nonblank(session_id, "session_id")
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                "DELETE FROM history_turns WHERE session_id = ?", (session_id,),
            )
            return cursor.rowcount
