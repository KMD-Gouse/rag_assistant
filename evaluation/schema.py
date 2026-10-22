"""Pydantic models for the manually verified evaluation dataset."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


QuestionType = Literal[
    "fact",
    "table",
    "comparison",
    "follow_up",
    "unanswerable",
]

DatasetSplit = Literal["dev", "held_out", "development", "challenge"]
MessageRole = Literal["user", "assistant"]


class HistoryMessage(BaseModel):
    """One earlier message needed to understand a follow-up question."""

    model_config = ConfigDict(extra="forbid")

    role: MessageRole
    content: str = Field(min_length=1)


class Evidence(BaseModel):
    """One manually verified evidence location."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(pattern=r"^[a-z0-9_]+$")
    pdf_page: int = Field(ge=1)
    printed_page: str | None = None
    evidence_text: str = Field(min_length=1)
    # Locations sharing a group are alternatives for the same required fact.
    # A missing group preserves the original one-location-per-fact behavior.
    group: str | None = Field(default=None, pattern=r"^[a-z0-9_]+$")


class EvaluationQuestion(BaseModel):
    """One question and its manually verified reference answer."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^q\d{3}$")
    question: str = Field(min_length=1)
    type: QuestionType
    answerable: bool
    expected_answer: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    history: list[HistoryMessage] = Field(default_factory=list)
    split: DatasetSplit

    @model_validator(mode="after")
    def validate_question_consistency(self) -> "EvaluationQuestion":
        if self.answerable:
            if not self.expected_answer:
                raise ValueError(
                    "Answerable questions require expected_answer."
                )

            if not self.evidence:
                raise ValueError(
                    "Answerable questions require at least one evidence item."
                )
        else:
            if self.expected_answer is not None:
                raise ValueError(
                    "Unanswerable questions must use expected_answer=null."
                )

            if self.evidence:
                raise ValueError(
                    "Unanswerable questions must use evidence=[]."
                )

        if self.type == "unanswerable" and self.answerable:
            raise ValueError(
                "Questions with type='unanswerable' cannot be answerable."
            )

        if not self.answerable and self.type != "unanswerable":
            raise ValueError(
                "Questions with answerable=false must use "
                "type='unanswerable'."
            )

        if self.type == "follow_up" and not self.history:
            raise ValueError(
                "Follow-up questions require conversation history."
            )

        return self
