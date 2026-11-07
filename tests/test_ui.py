from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from ui.client import APIClient, APIError, Completed, Turn


def turn(session="s", question="Revenue?", *, abstained=False):
    return {
        "id": 1, "session_id": session, "original_question": question,
        "result": {
            "question": "What was TCS revenue in FY2024-25?",
            "answer": "I don't know" if abstained else "Revenue was ₹100.",
            "abstained": abstained,
            "citations": [] if abstained else [{
                "evidence_id": "E1", "doc_id": "tcs_fy2025", "company": "TCS", "fiscal_year": "2024-25",
                "pdf_page": 10, "printed_page": "8", "quote": "Revenue was ₹100.",
            }],
        },
    }


def completed(session="s", question="Revenue?"):
    return {
        "status": "completed", "session_id": session,
        "rewrite": {"original_question": question, "standalone_question": "TCS revenue?"},
        "turn": turn(session, question),
    }


def test_http_client_contract():
    requests = []

    def handler(request):
        requests.append(request)
        if request.method == "GET":
            assert request.url.params["limit"] == "100"
            return httpx.Response(200, json=[turn()])
        assert request.content == b'{"session_id":"s","question":"Revenue?"}'
        return httpx.Response(200, json=completed())

    client = APIClient("http://api", transport=httpx.MockTransport(handler))
    assert client.history("s")[0].result.citations[0].pdf_page == 10
    assert isinstance(client.chat("s", "Revenue?"), Completed)
    assert [request.url.path for request in requests] == ["/sessions/s/history", "/chat"]


@pytest.mark.parametrize("status", [409, 422, 429, 500, 502, 503])
def test_http_errors_do_not_expose_raw_bodies(status):
    client = APIClient("http://api", transport=httpx.MockTransport(
        lambda request: httpx.Response(status, text="secret internal information"),
    ))
    with pytest.raises(APIError) as caught:
        client.chat("s", "Question")
    assert "secret" not in str(caught.value)


def test_timeout_is_not_retried():
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("secret")

    client = APIClient("http://api", transport=httpx.MockTransport(handler))
    with pytest.raises(APIError, match="Refresh history before resending"):
        client.chat("s", "Question")
    assert len(calls) == 1


@pytest.mark.parametrize("payload", [{}, completed("other")])
def test_invalid_or_cross_session_chat_response(payload):
    client = APIClient("http://api", transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json=payload),
    ))
    with pytest.raises(APIError, match="invalid chat response"):
        client.chat("s", "Question")


def test_cross_session_history_rejected():
    client = APIClient("http://api", transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json=[turn("other")]),
    ))
    with pytest.raises(APIError, match="invalid session history"):
        client.history("s")


APP_PATH = Path(__file__).resolve().parents[1] / "ui" / "streamlit_app.py"


def test_streamlit_chat_rerun_and_new_session(monkeypatch):
    calls = []

    def request(self, method, path, **kwargs):
        calls.append((method, kwargs))
        if method == "GET":
            return []
        return completed(kwargs["json"]["session_id"], kwargs["json"]["question"])

    monkeypatch.setattr(APIClient, "_request", request)
    app = AppTest.from_file(str(APP_PATH)).run()
    assert not app.exception
    original_session = app.session_state["session_id"]
    app.chat_input[0].set_value("Revenue?").run()
    assert not app.exception
    assert len(app.chat_message) == 2
    assert any(exp.label == "Sources (1)" for exp in app.expander)
    app.run()
    assert len([call for call in calls if call[0] == "POST"]) == 1
    app.sidebar.button[0].click().run()
    assert app.session_state["session_id"] != original_session
    assert len(app.chat_message) == 0


def test_streamlit_resume_history(monkeypatch):
    def request(self, method, path, **kwargs):
        return [turn("saved", abstained=True)] if "/saved/" in path else []

    monkeypatch.setattr(APIClient, "_request", request)
    app = AppTest.from_file(str(APP_PATH)).run()
    app.sidebar.text_input[0].set_value("saved")
    app.sidebar.button[1].click().run()
    assert not app.exception
    assert app.session_state["session_id"] == "saved"
    assert len(app.chat_message) == 2
    assert any(info.value == "I don't know" for info in app.info)


def test_streamlit_clarification_and_failure(monkeypatch):
    calls = []

    def request(self, method, path, **kwargs):
        if method == "GET":
            return []
        calls.append(kwargs)
        if len(calls) > 1:
            raise APIError("Could not reach the API")
        data = kwargs["json"]
        return {"status": "needs_clarification", "session_id": data["session_id"],
                "rewrite": {"original_question": data["question"], "standalone_question": data["question"]},
                "message": "Which company do you mean?"}

    monkeypatch.setattr(APIClient, "_request", request)
    app = AppTest.from_file(str(APP_PATH)).run()
    app.chat_input[0].set_value("Its margin?").run()
    assert any(info.value == "Which company do you mean?" for info in app.info)
    app.chat_input[0].set_value("TCS").run()
    assert not app.exception
    assert any("Could not reach" in error.value for error in app.error)
    app.run()
    assert len(calls) == 2


def test_streamlit_history_failure_blocks_submission(monkeypatch):
    def request(*args, **kwargs):
        raise APIError("Could not reach the API")

    monkeypatch.setattr(APIClient, "_request", request)
    app = AppTest.from_file(str(APP_PATH)).run()
    assert not app.exception
    assert len(app.chat_input) == 0
    assert len(app.error) == 1


@pytest.mark.parametrize("diagnostic", [None, "Quote for E1 is not present in the retrieved chunk."])
def test_streamlit_explains_citation_failure(monkeypatch, diagnostic):
    def request(self, method, path, **kwargs):
        session = path.split("/")[2]
        saved = turn(session, abstained=True)
        saved["result"].update(abstention_reason="invalid_citations", retrieved_chunks=5)
        if diagnostic:
            saved["result"]["citation_validation_error"] = diagnostic
        return [saved]

    monkeypatch.setattr(APIClient, "_request", request)
    app = AppTest.from_file(str(APP_PATH)).run()
    assert not app.exception
    assert any(exp.label == "Why was this answer withheld?" for exp in app.expander)
    text = " ".join(element.value for element in app.text)
    assert "Retrieved passages: 5" in text
    assert diagnostic in text if diagnostic else "no detailed validation error" in text


def test_citation_click_opens_exact_report_and_navigates(monkeypatch):
    import pymupdf

    pages = []
    monkeypatch.setattr(APIClient, "history", lambda self, session: [
        Turn.model_validate(turn(session))
    ])
    monkeypatch.setattr(APIClient, "report", lambda self, doc: {
        "doc_id": doc, "title": "TCS Annual Report", "pdf_pages": 20,
    })

    def page(self, doc, number):
        pages.append((doc, number))
        with pymupdf.open() as pdf:
            return pdf.new_page(width=100, height=100).get_pixmap().tobytes("png")

    monkeypatch.setattr(APIClient, "report_page", page)
    app = AppTest.from_file(str(APP_PATH)).run()
    next(button for button in app.button if button.label.startswith("[E1]")).click().run()
    assert not app.exception
    assert pages[-1] == ("tcs_fy2025", 10)
    app.number_input[0].set_value(11).run()
    assert not app.exception
    assert pages[-1] == ("tcs_fy2025", 11)


def test_report_binary_client():
    seen = []

    def handler(request):
        seen.append(request.url.path)
        return httpx.Response(200, content=b"binary-report")

    client = APIClient("http://api", transport=httpx.MockTransport(handler))
    assert client.report_page("tcs_fy2025", 10) == b"binary-report"
    assert client.report_pdf("tcs_fy2025") == b"binary-report"
    assert seen == ["/reports/tcs_fy2025/pages/10", "/reports/tcs_fy2025/pdf"]
