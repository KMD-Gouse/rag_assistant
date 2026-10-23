import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.generation.citations import CitationValidationError, validate_citations
from app.generation.groq import GroqGenerationError, GroqGenerator
from app.generation.models import AnswerDraft, DraftCitation
from app.generation.service import UNKNOWN_ANSWER, answer_question


def point(text: str = "Revenue from operations was ₹13,90,853 million."):
    return SimpleNamespace(
        id="point-1",
        payload={
            "chunk_id": "tcs_fy2025_p10_c01",
            "doc_id": "tcs_fy2025",
            "company": "TCS",
            "fiscal_year": "2024-25",
            "pdf_page": 10,
            "printed_page": "8",
            "section": "Financial highlights",
            "text": text,
        },
    )


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, question, limit=5, **filters):
        self.calls.append((question, limit, filters))
        return self.results


class FakeGenerator:
    model = "fake-groq"

    def __init__(self, generated):
        self.generated = generated
        self.calls = []

    def generate(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, json.loads(user_prompt)))
        return self.generated


def draft(**updates):
    values = {
        "answerable": True,
        "evidence_sufficiency": 0.91,
        "answer": "TCS reported revenue from operations of ₹13,90,853 million.",
        "citations": [{
            "evidence_id": "E1",
            "quote": "Revenue from operations was ₹13,90,853 million.",
        }],
    }
    values.update(updates)
    return AnswerDraft.model_validate(values)


def test_answer_question_returns_validated_metadata_and_applies_scope():
    retriever = FakeRetriever([point()])
    generator = FakeGenerator(draft())
    result = answer_question(
        "What revenue did TCS report in FY2024-25?",
        retriever=retriever,
        generator=generator,
    )
    assert not result.abstained
    assert result.citations[0].chunk_id == "tcs_fy2025_p10_c01"
    assert result.citations[0].pdf_page == 10
    assert retriever.calls[0][2] == {"company": ("TCS",), "fiscal_year": ("2024-25",)}
    sent = generator.calls[0][1]
    assert sent["evidence"][0]["evidence_id"] == "E1"
    assert sent["evidence"][0]["text"].startswith("Revenue")


def test_empty_retrieval_abstains_without_calling_generator():
    generator = FakeGenerator(draft())
    result = answer_question("Unknown fact?", retriever=FakeRetriever([]), generator=generator)
    assert result.answer == UNKNOWN_ANSWER
    assert result.abstention_reason == "no_retrieval_results"
    assert generator.calls == []


@pytest.mark.parametrize(
    ("generated", "reason"),
    [
        (draft(answerable=False, evidence_sufficiency=0.1, answer="", citations=[]),
         "model_says_unanswerable"),
        (draft(evidence_sufficiency=0.69), "below_evidence_threshold"),
        (draft(citations=[{"evidence_id": "E2", "quote": "Some fabricated quotation"}]),
         "invalid_citations"),
        (draft(citations=[{"evidence_id": "E1", "quote": "A fabricated revenue value"}]),
         "invalid_citations"),
        (draft(answer="", citations=[]), "invalid_citations"),
    ],
)
def test_abstention_paths(generated, reason):
    result = answer_question(
        "Question",
        retriever=FakeRetriever([point()]),
        generator=FakeGenerator(generated),
    )
    assert result.abstained
    assert result.answer == UNKNOWN_ANSWER
    assert result.abstention_reason == reason
    assert result.citations == []


def test_threshold_is_inclusive():
    result = answer_question(
        "Question",
        retriever=FakeRetriever([point()]),
        generator=FakeGenerator(draft(evidence_sufficiency=0.70)),
        min_evidence_sufficiency=0.70,
    )
    assert not result.abstained


def test_draft_models_forbid_extra_fields_and_bad_evidence_ids():
    with pytest.raises(ValidationError):
        AnswerDraft.model_validate({
            "answerable": False,
            "evidence_sufficiency": 0.0,
            "answer": "",
            "citations": [],
            "unexpected": True,
        })
    with pytest.raises(ValidationError):
        DraftCitation(evidence_id="page-1", quote="A sufficiently long quote")


def test_citation_normalizes_unicode_and_whitespace_but_rejects_duplicates():
    evidence = {"E1": point("Total\u00a0revenue was ₹100 million")}
    citation = DraftCitation(evidence_id="E1", quote="total revenue was ₹100 million")
    assert validate_citations([citation], evidence)[0].doc_id == "tcs_fy2025"
    with pytest.raises(CitationValidationError, match="Duplicate"):
        validate_citations([citation, citation], evidence)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def read(self):
        return json.dumps(self.payload).encode()


def test_groq_request_uses_strict_json_schema_and_parses_response():
    captured = {}

    def opener(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse({
            "choices": [{"message": {"content": draft().model_dump_json()}}]
        })

    generator = GroqGenerator(
        "secret-key",
        "openai/gpt-oss-20b",
        "https://api.groq.com/openai/v1/chat/completions",
        timeout=12,
        opener=opener,
    )
    result = generator.generate("system", "user")
    body = json.loads(captured["request"].data)
    assert result.answerable
    assert body["response_format"]["json_schema"]["strict"] is True
    schema = body["response_format"]["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["DraftCitation"]["additionalProperties"] is False
    assert "pattern" not in schema["$defs"]["DraftCitation"]["properties"]["evidence_id"]
    assert "maximum" not in schema["properties"]["evidence_sufficiency"]
    assert captured["request"].get_header("Authorization") == "Bearer secret-key"
    assert captured["request"].get_header("User-agent") == "annual-report-qa/0.1"
    assert captured["timeout"] == 12


def test_groq_invalid_response_is_sanitized():
    generator = GroqGenerator(
        "secret-key", "model", "https://example.test",
        opener=lambda *_args, **_kwargs: FakeResponse({"wrong": "shape"}),
    )
    with pytest.raises(GroqGenerationError, match="invalid structured response") as error:
        generator.generate("system", "user")
    assert "secret-key" not in str(error.value)


@pytest.mark.parametrize("top_k, threshold", [(0, 0.7), (5, -0.1), (5, 1.1)])
def test_answer_question_rejects_invalid_controls(top_k, threshold):
    with pytest.raises(ValueError):
        answer_question(
            "Question",
            retriever=FakeRetriever([]),
            generator=FakeGenerator(draft()),
            top_k=top_k,
            min_evidence_sufficiency=threshold,
        )
