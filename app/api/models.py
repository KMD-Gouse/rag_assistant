"""Public HTTP request and response contracts."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.memory import HistoryTurn, RewriteResult


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    session_id: str = Field(min_length=1, max_length=128)
    question: str = Field(min_length=1, max_length=4000)

    @field_validator("session_id", "question")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Must not be blank")
        return value


class CompletedResponse(BaseModel):
    status: Literal["completed"] = "completed"
    session_id: str
    rewrite: RewriteResult
    turn: HistoryTurn


class ClarificationResponse(BaseModel):
    status: Literal["needs_clarification"] = "needs_clarification"
    session_id: str
    rewrite: RewriteResult
    message: str = "Please specify the company, fiscal year, or subject you mean."
