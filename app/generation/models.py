"""Strict models for untrusted model output and trusted public results."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DraftCitation(BaseModel):
    """A reference emitted by the LLM before it is checked."""

    model_config = ConfigDict(extra="forbid", strict=True)

    evidence_id: str = Field(pattern=r"^E[1-9][0-9]*$")
    quote: str = Field(min_length=1, max_length=1000)


class AnswerDraft(BaseModel):
    """Groq's structured response. Semantic checks happen in the service."""

    model_config = ConfigDict(extra="forbid", strict=True)

    answerable: bool
    evidence_sufficiency: float = Field(ge=0.0, le=1.0)
    answer: str = Field(max_length=4000)
    citations: list[DraftCitation] = Field(max_length=10)


class ValidatedCitation(BaseModel):
    """A model citation resolved against trusted retrieval metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str
    chunk_id: str
    doc_id: str
    company: str
    fiscal_year: str
    pdf_page: int
    printed_page: str | int | None = None
    quote: str


AbstentionReason = Literal[
    "no_retrieval_results",
    "model_says_unanswerable",
    "below_evidence_threshold",
    "invalid_citations",
]


class AnswerResult(BaseModel):
    """The stable return type of :func:`answer_question`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    question: str
    answer: str
    abstained: bool
    abstention_reason: AbstentionReason | None
    evidence_sufficiency: float
    threshold: float
    citations: list[ValidatedCitation]
    retrieved_chunks: int
    model: str | None
