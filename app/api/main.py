"""Local, single-worker HTTP wrapper around the frozen RAG pipeline."""

from contextlib import asynccontextmanager, contextmanager
import logging
import sqlite3
from threading import Lock
from typing import Callable

from fastapi import FastAPI, HTTPException, Path, Query

from app.api.models import ChatRequest, ClarificationResponse, CompletedResponse
from app.api.reports import report_router
from app.generation import AnswerResult, answer_question
from app.generation.groq import GroqGenerationError
from app.memory import (
    FollowUpRewriter, HistoryRepository, HistoryTurn, RewriteError, SQLiteHistoryRepository,
)
from app.memory.rewriter import RewriteGenerator


logger = logging.getLogger(__name__)


class SessionRequests:
    """Reject overlapping turns in one session without retaining idle locks."""

    def __init__(self):
        self._guard = Lock()
        self._active: set[str] = set()

    @contextmanager
    def acquire(self, session_id: str):
        with self._guard:
            if session_id in self._active:
                raise HTTPException(409, "A request is already running for this session.")
            self._active.add(session_id)
        try:
            yield
        finally:
            with self._guard:
                self._active.remove(session_id)


def create_app(
    *, history: HistoryRepository | None = None,
    rewrite_generator: RewriteGenerator | None = None,
    answerer: Callable[[str], AnswerResult] = answer_question,
) -> FastAPI:
    """Inject storage/model boundaries for tests without external services."""
    requests = SessionRequests()
    # Protect initial construction and inference on the shared local retriever.
    pipeline_lock = Lock()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        repository = history if history is not None else SQLiteHistoryRepository()
        application.state.history = repository
        application.state.rewriter = FollowUpRewriter(repository, rewrite_generator)
        yield

    application = FastAPI(title="Annual Report Q&A", version="0.1.0", lifespan=lifespan)
    application.include_router(report_router())

    @application.get("/health")
    def health():
        """Liveness only; does not probe Groq, Qdrant, or model readiness."""
        return {
            "status": "ok",
            "generation_pipeline": "source-excerpts-v1",
            "comparison_retrieval": "balanced-v1",
            "provider_policy": "fail-fast-429",
        }

    @application.post("/chat", response_model=CompletedResponse | ClarificationResponse)
    def chat(request: ChatRequest):
        # A sync route runs blocking SQLite, model inference, and HTTP in a worker thread.
        with requests.acquire(request.session_id):
            try:
                rewrite = application.state.rewriter.rewrite(request.session_id, request.question)
                if rewrite.needs_clarification:
                    return ClarificationResponse(session_id=request.session_id, rewrite=rewrite)
                with pipeline_lock:
                    result = answerer(rewrite.standalone_question)
                turn = application.state.history.append_turn(
                    request.session_id, request.question, result,
                )
                return CompletedResponse(session_id=request.session_id, rewrite=rewrite, turn=turn)
            except (RewriteError, GroqGenerationError) as error:
                logger.warning(
                    "Language-model request failed: %s%s",
                    error,
                    f" Provider: {error.provider_detail}" if getattr(error, "provider_detail", None) else "",
                )
                if error.status_code == 429:
                    raise HTTPException(
                        429, "The language-model rate limit is active. Wait before retrying.",
                    ) from error
                raise HTTPException(502, "The language-model service failed. Please retry.") from error
            except sqlite3.Error as error:
                raise HTTPException(503, "Session history is unavailable. Please retry.") from error
            except Exception as error:
                logger.error("Chat request failed (%s)", type(error).__name__)
                raise HTTPException(500, "The answer pipeline failed.") from error

    @application.get("/sessions/{session_id}/history", response_model=list[HistoryTurn])
    def get_history(
        session_id: str = Path(min_length=1, max_length=128, pattern=r"\S"),
        limit: int = Query(default=10, ge=1, le=100),
    ):
        try:
            return application.state.history.get_history(session_id, limit=limit)
        except sqlite3.Error as error:
            raise HTTPException(503, "Session history is unavailable. Please retry.") from error

    return application


app = create_app()
