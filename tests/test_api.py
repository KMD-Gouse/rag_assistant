from concurrent.futures import ThreadPoolExecutor
from functools import partial
import sqlite3
from threading import Event

from fastapi.testclient import TestClient
import pytest

from app.api.main import create_app
from app.generation.groq import GroqGenerationError
from app.generation.service import answer_question
from app.memory import RewriteError, SQLiteHistoryRepository
from tests.test_generation import FakeGenerator as AnswerGenerator, FakeRetriever, draft, point
from tests.test_memory import answer
from tests.test_rewriter import FakeGenerator


@pytest.fixture
def history(tmp_path):
    return SQLiteHistoryRepository(tmp_path / "history.sqlite3")


def test_chat_runs_pipeline_and_persists_original_and_rewritten_questions(history):
    rewrite_generator = FakeGenerator("What revenue did TCS report in FY2024-25?")
    retriever = FakeRetriever([point()])
    app = create_app(history=history, rewrite_generator=rewrite_generator,
                     answerer=partial(answer_question, retriever=retriever,
                                      generator=AnswerGenerator(draft())))
    with TestClient(app) as client:
        first = client.post("/chat", json={"session_id": "a", "question": "TCS revenue FY2024-25?"})
        assert first.status_code == 200
        assert first.json()["status"] == "completed"
        assert rewrite_generator.calls == []
        response = client.post("/chat", json={"session_id": "a", "question": "And this year?"})
        assert response.status_code == 200
        body = response.json()
        assert body["rewrite"]["history_turns"] == 1
        assert body["turn"]["original_question"] == "And this year?"
        assert body["turn"]["result"]["question"] == rewrite_generator.output.standalone_question
        assert body["turn"]["result"]["citations"][0]["doc_id"] == "tcs_fy2025"
        assert not body["turn"]["result"]["abstained"]
        stored = client.get("/sessions/a/history?limit=1").json()
        assert stored == [body["turn"]]
        assert client.get("/sessions/unknown/history").json() == []
    assert len(history.get_history("a")) == 2


def test_sessions_do_not_share_context(history):
    generator = FakeGenerator()
    with TestClient(create_app(history=history, rewrite_generator=generator, answerer=answer)) as client:
        for session in ["a", "b", "a"]:
            assert client.post("/chat", json={"session_id": session, "question": session}).status_code == 200
        assert len(generator.calls) == 1
        assert [t["original_question"] for t in generator.calls[0][1]["history"]] == ["a"]
        assert len(client.get("/sessions/b/history").json()) == 1


def test_clarification_does_not_generate_or_write(history):
    history.append_turn("a", "Compare both companies", answer())

    def unexpected(question):
        pytest.fail("Clarification must not invoke generation")

    with TestClient(create_app(history=history, rewrite_generator=FakeGenerator(ambiguous=True),
                               answerer=unexpected)) as client:
        response = client.post("/chat", json={"session_id": "a", "question": "Its margin?"})
        assert response.status_code == 200
        assert response.json()["status"] == "needs_clarification"
        assert "turn" not in response.json()
    assert len(history.get_history("a")) == 1


def test_abstentions_are_saved(history):
    with TestClient(create_app(history=history, answerer=partial(answer, abstained=True))) as client:
        response = client.post("/chat", json={"session_id": "a", "question": "Unknown?"})
        assert response.status_code == 200
        assert response.json()["turn"]["result"]["abstained"]
    assert history.get_history("a")[0].result.abstained


@pytest.mark.parametrize("payload", [
    {}, {"session_id": "a"}, {"session_id": "", "question": "Hi"},
    {"session_id": " ", "question": "Hi"}, {"session_id": 5, "question": "Hi"},
    {"session_id": "a", "question": " "}, {"session_id": "a", "question": None},
    {"session_id": "a", "question": "x" * 4001},
    {"session_id": "a" * 129, "question": "Hi"},
    {"session_id": "a", "question": "Hi", "model": "override"},
])
def test_invalid_requests_are_rejected(history, payload):
    with TestClient(create_app(history=history, answerer=answer)) as client:
        assert client.post("/chat", json=payload).status_code == 422
    assert history.get_history("a") == []


@pytest.mark.parametrize("error,code", [
    (GroqGenerationError("limited", status_code=429), 429),
    (GroqGenerationError("secret"), 502), (RuntimeError("secret"), 500),
    (sqlite3.OperationalError("secret"), 503),
])
def test_failures_are_sanitized_and_release_session(history, error, code):
    attempts = []

    def answerer(question):
        attempts.append(question)
        if len(attempts) == 1:
            raise error
        return answer(question)

    with TestClient(create_app(history=history, answerer=answerer)) as client:
        response = client.post("/chat", json={"session_id": "a", "question": "Question"})
        assert response.status_code == code
        assert "secret" not in response.text
        assert history.get_history("a") == []
        assert client.post("/chat", json={"session_id": "a", "question": "Retry"}).status_code == 200


def test_rewrite_failure_does_not_generate(history):
    class BrokenGenerator:
        def generate(self, *args):
            raise RewriteError("secret")

    def unexpected(question):
        pytest.fail("Rewrite failure must stop generation")

    history.append_turn("a", "Question", answer())
    with TestClient(create_app(history=history, rewrite_generator=BrokenGenerator(),
                               answerer=unexpected)) as client:
        response = client.post("/chat", json={"session_id": "a", "question": "How about that metric?"})
        assert response.status_code == 502
        assert "secret" not in response.text
    assert len(history.get_history("a")) == 1


def test_failed_persistence_never_returns_success(history, monkeypatch):
    def fail(*args):
        raise sqlite3.OperationalError("disk failure")

    monkeypatch.setattr(history, "append_turn", fail)
    with TestClient(create_app(history=history, answerer=answer)) as client:
        assert client.post("/chat", json={"session_id": "a", "question": "Question"}).status_code == 503
    assert history.get_history("a") == []


def test_overlapping_same_session_requests_conflict(history):
    entered, release = Event(), Event()

    def slow_answer(question):
        entered.set()
        assert release.wait(10)
        return answer(question)

    with TestClient(create_app(history=history, answerer=slow_answer)) as client:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(client.post, "/chat", json={"session_id": "a", "question": "First"})
            try:
                assert entered.wait(10)
                response = client.post("/chat", json={"session_id": "a", "question": "Second"})
                assert response.status_code == 409
            finally:
                release.set()
            assert future.result(timeout=10).status_code == 200
    assert len(history.get_history("a")) == 1


def test_health_openapi_and_history_validation(history):
    with TestClient(create_app(history=history, answerer=answer)) as client:
        assert client.get("/health").json() == {
            "status": "ok",
            "generation_pipeline": "source-excerpts-v1",
            "comparison_retrieval": "balanced-v1",
            "provider_policy": "fail-fast-429",
        }
        assert "/chat" in client.get("/openapi.json").json()["paths"]
        for limit in [0, -1, 101, "bad"]:
            assert client.get(f"/sessions/a/history?limit={limit}").status_code == 422
        assert client.get("/sessions/%20/history").status_code == 422


def test_citation_diagnostic_is_returned_logged_and_persisted(history, caplog):
    generated = draft(citations=[{"evidence_id": "E1", "quote": "A fabricated revenue value"}])
    app = create_app(history=history, answerer=partial(
        answer_question, retriever=FakeRetriever([point()]), generator=AnswerGenerator(generated),
    ))
    with TestClient(app) as client:
        response = client.post("/chat", json={"session_id": "a", "question": "Revenue?"})
        result = response.json()["turn"]["result"]
        assert result["abstention_reason"] == "invalid_citations"
        expected = "Quote for E1 is not present in the retrieved chunk."
        assert result["citation_validation_error"] == expected
        assert result["citations"] == []
        assert expected in caplog.text
        assert "fabricated revenue" not in caplog.text
        assert client.get("/sessions/a/history").json()[0]["result"] == result
    assert SQLiteHistoryRepository(history.db_path).get_history("a")[0].result.citation_validation_error == expected
