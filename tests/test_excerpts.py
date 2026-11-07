import json

import pytest

from app.generation.citations import CitationValidationError, validate_citations
from app.generation.excerpts import ExcerptGenerator, source_excerpts
from app.generation.service import answer_question
from tests.test_generation import FakeGenerator, FakeRetriever, draft, point


def test_excerpts_are_exact_bounded_and_cover_source():
    text = "Currency ₹ 100 cr. Revenue ` 200 cr. " * 100
    spans = list(source_excerpts(text))
    assert len(spans) > 1
    assert all(span in text and 8 <= len(span) <= 650 for span in spans)
    assert spans[0].startswith("Currency")
    assert text.rstrip().endswith(spans[-1])


def test_model_selects_excerpt_server_supplies_exact_quote():
    text = "Revenue from operations 1,36,592 1,28,933 5.9 1,62,990 1,53,670 6.1"
    raw = FakeGenerator(draft(citations=[{"evidence_id": "E1", "quote": "E1:S1"}]))
    result = answer_question("Revenue?", retriever=FakeRetriever([point(text)]), generator=ExcerptGenerator(raw))
    assert not result.abstained
    assert result.citations[0].quote == text
    assert "text" not in raw.calls[0][1]["evidence"][0]
    assert raw.calls[0][1]["evidence"][0]["excerpts"][0] == {"excerpt_id": "E1:S1", "text": text}


def test_only_most_question_relevant_excerpts_are_sent():
    text = (
        "Background about governance and offices. " * 30
        + "Revenue from operations was 1,62,990 crore. " * 15
        + "Background about sustainability and communities. " * 30
    )
    raw = FakeGenerator(draft(answerable=False, answer="", citations=[]))
    answer_question(
        "What revenue was reported?", retriever=FakeRetriever([point(text)]),
        generator=ExcerptGenerator(raw),
    )
    excerpts = raw.calls[0][1]["evidence"][0]["excerpts"]
    assert 1 <= len(excerpts) <= 2
    assert any("Revenue from operations" in excerpt["text"] for excerpt in excerpts)


def test_large_comparison_sends_one_excerpt_per_evidence():
    raw = FakeGenerator(draft(answerable=False, answer="", citations=[]))
    payload = {
        "question": "Compare revenue",
        "comparison_scope": {"companies": ["Infosys", "TCS"], "fiscal_years": ["2024-25"]},
        "evidence": [
            {"evidence_id": f"E{index}", "text": "Revenue evidence. " * 100}
            for index in range(1, 7)
        ],
    }
    ExcerptGenerator(raw).generate("rules", json.dumps(payload))
    assert all(len(item["excerpts"]) == 1 for item in raw.calls[0][1]["evidence"])


@pytest.mark.parametrize("selection", ["E1:S999", "E2:S1", "Revenue was 100 crore."])
def test_unknown_cross_evidence_and_free_text_selections_rejected(selection):
    raw = FakeGenerator(draft(citations=[{"evidence_id": "E1", "quote": selection}]))
    payload = {"question": "Revenue?", "evidence": [
        {"evidence_id": "E1", "text": "Revenue was 100 crore."},
        {"evidence_id": "E2", "text": "Revenue was 200 crore."},
    ]}
    generated = ExcerptGenerator(raw).generate("rules", json.dumps(payload))
    with pytest.raises(CitationValidationError):
        validate_citations(generated.citations, {"E1": point("Revenue was 100 crore.")})


def test_default_application_uses_excerpt_generator(monkeypatch):
    raw = FakeGenerator(draft(citations=[{"evidence_id": "E1", "quote": "E1:S1"}]))
    monkeypatch.setattr("app.generation.service.build_default_generator", lambda: raw)
    result = answer_question("Revenue?", retriever=FakeRetriever([point()]))
    assert not result.abstained
    assert result.citations[0].quote == point().payload["text"]


def test_verbatim_mode_preserves_original_generator(monkeypatch):
    raw = FakeGenerator(draft())
    monkeypatch.setattr("app.generation.service.build_default_generator", lambda: raw)
    result = answer_question("Revenue?", retriever=FakeRetriever([point()]), citation_mode="verbatim", citation_retries=0)
    assert not result.abstained
    assert "text" in raw.calls[0][1]["evidence"][0]
