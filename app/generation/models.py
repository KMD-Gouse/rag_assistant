"""Strict models for untrusted model output and trusted public results."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DraftCitation(BaseModel):
    """A reference emitted by the LLM before it is checked."""

    model_config = ConfigDict(extra="forbid", strict=True)

    evidence_id: str = Field(pattern=r"^E[1-9][0-9]*$")
    quote: str = Field(min_length=1, max_length=1000)


class ComparableFact(BaseModel):
    """One validated side of a comparison (filled by the LLM, checked by service)."""

    model_config = ConfigDict(extra="forbid", strict=True)

    company: str = Field(min_length=1, max_length=200)
    fiscal_year: str = Field(min_length=1, max_length=20)   # e.g. "2024-25"
    metric: str = Field(min_length=1, max_length=200)        # e.g. "revenue"
    definition: str = Field(min_length=1, max_length=500)    # e.g. "consolidated revenue from operations"
    value_text: str = Field(min_length=1, max_length=100)    # exact text: "1,62,990" or "21.1%"
    unit: str = Field(min_length=1, max_length=100)          # "crore INR", "percent", "employees"
    evidence_id: str = Field(pattern=r"^E[1-9][0-9]*$")     # which chunk contains this fact
    supporting_quote: str = Field(min_length=1, max_length=1000)  # exact excerpt with this number


class AnswerDraft(BaseModel):
    """Groq's structured response. Semantic checks happen in the service."""

    model_config = ConfigDict(extra="forbid", strict=True)

    answerable: bool
    evidence_sufficiency: float = Field(ge=0.0, le=1.0)
    answer: str = Field(max_length=4000)
    citations: list[DraftCitation] = Field(max_length=10)
    # Optional: present only for comparison questions; omitted for single-side answers
    comparable_facts: list[ComparableFact] | None = None


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
    citation_validation_error: str | None = None
