"""HTTP-only boundary for the Streamlit UI, independent of backend modules."""

from typing import Literal
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError


class APIModel(BaseModel):
    model_config = ConfigDict(strict=True)


class Citation(APIModel):
    evidence_id: str
    doc_id: str | None = None
    company: str
    fiscal_year: str
    pdf_page: int
    printed_page: str | int | None = None
    quote: str


class Answer(APIModel):
    question: str
    answer: str
    abstained: bool
    citations: list[Citation]
    abstention_reason: str | None = None
    citation_validation_error: str | None = None
    retrieved_chunks: int | None = None


class Turn(APIModel):
    id: int
    session_id: str
    original_question: str
    result: Answer


class Rewrite(APIModel):
    original_question: str
    standalone_question: str


class Completed(APIModel):
    status: Literal["completed"]
    session_id: str
    rewrite: Rewrite
    turn: Turn


class Clarification(APIModel):
    status: Literal["needs_clarification"]
    session_id: str
    rewrite: Rewrite
    message: str


CHAT_RESPONSE = TypeAdapter(Completed | Clarification)
HISTORY_RESPONSE = TypeAdapter(list[Turn])


class APIError(RuntimeError):
    """A user-facing failure with no raw server body or credentials."""


class APIClient:
    def __init__(self, base_url: str, *, transport=None):
        self.base_url = base_url.rstrip("/")
        self.transport = transport

    def _request(self, method: str, path: str, *, binary: bool = False, **kwargs):
        try:
            with httpx.Client(
                base_url=self.base_url, transport=self.transport,
                timeout=httpx.Timeout(300.0, connect=5.0),
            ) as client:
                response = client.request(method, path, **kwargs)
                response.raise_for_status()
                return response.content if binary else response.json()
        except httpx.TimeoutException as error:
            raise APIError(
                "The API timed out. The request may still finish. Refresh history before resending."
            ) from error
        except httpx.HTTPStatusError as error:
            messages = {
                404: "The cited report or PDF page is unavailable.",
                429: "The Groq rate limit is active. Wait before sending another question.",
                409: "This conversation already has a request running. Wait, then refresh history.",
                422: "The API rejected the input. Use a nonblank question of at most 4,000 characters.",
                502: "The language-model service is unavailable. Please try again later.",
                503: "Session history is unavailable. Please try again later.",
            }
            raise APIError(messages.get(error.response.status_code, "The API could not complete the request.")) from error
        except httpx.RequestError as error:
            raise APIError(
                "Could not reach the API. Check that FastAPI is running. Refresh history before resending."
            ) from error
        except ValueError as error:
            raise APIError("The API returned an invalid response.") from error

    def history(self, session_id: str) -> list[Turn]:
        payload = self._request("GET", f"/sessions/{quote(session_id, safe='')}/history", params={"limit": 100})
        try:
            turns = HISTORY_RESPONSE.validate_python(payload)
            if any(turn.session_id != session_id for turn in turns):
                raise ValueError("Session mismatch")
            return turns
        except (ValueError, ValidationError) as error:
            raise APIError("The API returned invalid session history.") from error

    def report(self, doc_id: str) -> dict:
        data = self._request("GET", f"/reports/{quote(doc_id, safe='')}")
        if not isinstance(data, dict) or data.get("doc_id") != doc_id or not isinstance(data.get("title"), str) or type(data.get("pdf_pages")) is not int or data["pdf_pages"] < 1:
            raise APIError("The API returned invalid report metadata.")
        return data

    def report_page(self, doc_id: str, page: int) -> bytes:
        return self._request("GET", f"/reports/{quote(doc_id, safe='')}/pages/{page}", binary=True)

    def report_pdf(self, doc_id: str) -> bytes:
        return self._request("GET", f"/reports/{quote(doc_id, safe='')}/pdf", binary=True)

    def chat(self, session_id: str, question: str) -> Completed | Clarification:
        payload = self._request("POST", "/chat", json={"session_id": session_id, "question": question})
        try:
            result = CHAT_RESPONSE.validate_python(payload)
            if result.session_id != session_id or (
                isinstance(result, Completed) and result.turn.session_id != session_id
            ):
                raise ValueError("Session mismatch")
            return result
        except (ValueError, ValidationError) as error:
            raise APIError("The API returned an invalid chat response. Refresh history before resending.") from error
