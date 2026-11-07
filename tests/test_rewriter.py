import io
import json
from urllib.error import HTTPError, URLError

import pytest

from app.config import Settings
from app.memory import FollowUpRewriter, RewriteError, SQLiteHistoryRepository
from app.memory.rewriter import GroqRewriteGenerator, RewriteDraft
from tests.test_memory import answer


class FakeGenerator:
    def __init__(self, question="What was Infosys revenue in FY2025?", ambiguous=False):
        self.output = RewriteDraft(standalone_question=question, needs_clarification=ambiguous)
        self.calls = []

    def generate(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, json.loads(user_prompt)))
        return self.output


@pytest.fixture
def history(tmp_path):
    return SQLiteHistoryRepository(tmp_path / "history.sqlite3")


def test_first_turn_preserved_without_model_call(history):
    generator = FakeGenerator()
    result = FollowUpRewriter(history, generator).rewrite("new", "What is revenue?")
    assert result.standalone_question == result.original_question == "What is revenue?"
    assert not result.rewritten
    assert not result.needs_clarification
    assert result.history_turns == 0
    assert generator.calls == []


def test_rewrite_uses_only_recent_session_history_without_writing(history):
    history.append_turn("other", "Private question", answer())
    for index in range(3):
        history.append_turn("mine", f"Original {index}", answer(f"Resolved {index}"))
    before = history.get_history("mine")
    generator = FakeGenerator()
    result = FollowUpRewriter(history, generator, history_limit=2).rewrite("mine", "And Infosys?")
    assert result.original_question == "And Infosys?"
    assert result.standalone_question == generator.output.standalone_question
    assert result.rewritten and result.history_turns == 2
    _, payload = generator.calls[0]
    assert payload["current_question"] == "And Infosys?"
    assert [t["standalone_question"] for t in payload["history"]] == ["Resolved 1", "Resolved 2"]
    assert [t["original_question"] for t in payload["history"]] == ["Original 1", "Original 2"]
    assert "Private" not in json.dumps(payload)
    assert history.get_history("mine") == before


def test_abstention_answer_not_used_as_context(history):
    history.append_turn("s", "Revenue?", answer(abstained=True))
    generator = FakeGenerator()
    FollowUpRewriter(history, generator).rewrite("s", "How about that metric?")
    assert generator.calls[0][1]["history"][0]["answer"] is None


def test_ambiguous_output_preserves_original(history):
    history.append_turn("s", "Compare TCS and Infosys", answer())
    generator = FakeGenerator("A speculative interpretation", ambiguous=True)
    result = FollowUpRewriter(history, generator).rewrite("s", "What about its margin?")
    assert result.needs_clarification
    assert result.standalone_question == "What about its margin?"
    assert not result.rewritten


def test_standalone_question_can_remain_unchanged(history):
    history.append_turn("s", "Old topic", answer())
    question = "How many employees did TCS report in FY2024-25?"
    generator = FakeGenerator(question)
    result = FollowUpRewriter(history, generator).rewrite("s", question)
    assert not result.rewritten
    assert generator.calls == []


@pytest.mark.parametrize(("previous", "follow_up", "expected"), [
    ("What revenue did TCS report in FY2024-25?", "And Infosys?",
     "What revenue did Infosys report in FY2024-25?"),
    ("How many employees did Infosys report in FY2024-25?", "What about TCS?",
     "How many employees did TCS report in FY2024-25?"),
    ("What revenue did Infosys report in FY2024-25?", "And the previous year?",
     "What revenue did Infosys report in FY2023-24?"),
])
def test_common_follow_ups_are_local(history, previous, follow_up, expected):
    history.append_turn("s", "Prior", answer(previous))
    generator = FakeGenerator("should not be called")
    result = FollowUpRewriter(history, generator).rewrite("s", follow_up)
    assert result.standalone_question == expected
    assert generator.calls == []


def test_repository_isolation_violation_stops_before_provider(history):
    turn = history.append_turn("other", "Private", answer())

    class BrokenRepository:
        def get_history(self, session_id, *, limit):
            return [turn]

    generator = FakeGenerator()
    with pytest.raises(RewriteError, match="another session"):
        FollowUpRewriter(BrokenRepository(), generator).rewrite("mine", "And then?")
    assert generator.calls == []


@pytest.mark.parametrize("question", ["", "  ", None, 3, "x" * 4001])
def test_invalid_questions(history, question):
    with pytest.raises(ValueError):
        FollowUpRewriter(history).rewrite("s", question)


@pytest.mark.parametrize("limit", [0, -1, True, 1.2])
def test_invalid_window(history, limit):
    with pytest.raises(ValueError):
        FollowUpRewriter(history, history_limit=limit)


def test_invalid_model_output_is_not_used(history):
    history.append_turn("s", "Question", answer())
    with pytest.raises(RewriteError, match="Invalid rewrite"):
        FollowUpRewriter(history, FakeGenerator("   ")).rewrite("s", "How about that metric?")


@pytest.fixture
def settings(monkeypatch):
    config = Settings(_env_file=None, groq_api_key="test-secret")
    monkeypatch.setattr("app.memory.rewriter.get_settings", lambda: config)
    return config


def test_groq_request_and_structured_response(settings):
    def opener(request, timeout):
        body = json.loads(request.data)
        assert body["model"] == settings.groq_model
        assert body["messages"] == [
            {"role": "system", "content": "Rules"}, {"role": "user", "content": "Data"},
        ]
        assert body["response_format"]["json_schema"]["strict"] is True
        assert timeout == settings.groq_timeout_seconds
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": json.dumps({
            "standalone_question": "What was TCS revenue?", "needs_clarification": False,
        })}}]}).encode())

    draft = GroqRewriteGenerator(opener=opener).generate("Rules", "Data")
    assert draft.standalone_question == "What was TCS revenue?"


def test_groq_rewrite_retries_transient_errors(settings):
    calls, sleeps = [], []

    def opener(request, timeout):
        calls.append(request)
        if len(calls) < 3:
            raise HTTPError("url", 503, "temporary", {}, None)
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": json.dumps({
            "standalone_question": "What was TCS revenue?", "needs_clarification": False,
        })}}]}).encode())

    result = GroqRewriteGenerator(
        opener=opener, sleeper=sleeps.append, transient_retries=2,
    ).generate("Rules", "Data")
    assert result.standalone_question == "What was TCS revenue?"
    assert len(calls) == 3
    assert sleeps == [1.0, 2.0]


@pytest.mark.parametrize("payload", [
    {}, {"choices": []}, {"choices": [{"message": {"content": "not json"}}]},
    {"choices": [{"message": {"content": '{"standalone_question":"x"}'}}]},
    {"choices": [{"message": {"content": '{"standalone_question":"x","needs_clarification":"false"}'}}]},
])
def test_invalid_provider_responses(settings, payload):
    with pytest.raises(RewriteError, match="invalid rewrite response"):
        GroqRewriteGenerator(opener=lambda *a, **kw: io.BytesIO(json.dumps(payload).encode())).generate("", "")


@pytest.mark.parametrize("error", [
    HTTPError("url", 429, "test-secret", {}, None), URLError("test-secret"), TimeoutError("test-secret"),
])
def test_provider_errors_are_sanitized_and_propagated(history, settings, error):
    def opener(*args, **kwargs):
        raise error

    history.append_turn("s", "Question", answer())
    with pytest.raises(RewriteError) as caught:
        FollowUpRewriter(history, GroqRewriteGenerator(opener=opener)).rewrite(
            "s", "How about that metric?",
        )
    assert "test-secret" not in str(caught.value)
    assert len(history.get_history("s")) == 1


def test_missing_key_only_required_when_rewriting(history, settings):
    settings.groq_api_key = None
    rewriter = FollowUpRewriter(history)
    assert rewriter.rewrite("s", "First question").history_turns == 0
    history.append_turn("s", "Question", answer())
    with pytest.raises(RewriteError, match="GROQ_API_KEY"):
        rewriter.rewrite("s", "How about that metric?")
