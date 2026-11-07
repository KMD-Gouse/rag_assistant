import json
from types import SimpleNamespace
from urllib.error import HTTPError

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


class ScopedRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, question, limit=5, **filters):
        self.calls.append((question, limit, filters))
        return self.results.get((filters.get("company"), filters.get("fiscal_year")), [])[:limit]


def scoped_point(point_id, company, fiscal_year, text):
    value = point(text)
    value.id = point_id
    value.payload.update(
        chunk_id=point_id, doc_id=f"{company.casefold()}_{fiscal_year}",
        company=company, fiscal_year=fiscal_year,
    )
    return value


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
        "evidence_sufficiency": 1.0,
        "answer": "TCS reported revenue from operations of ₹13,90,853 million.",
        "citations": [{
            "evidence_id": "E1",
            "quote": "Revenue from operations was ₹13,90,853 million.",
        }],
    }
    values.update(updates)
    return AnswerDraft.model_validate(values)


def comparison_fact(company, fiscal_year, metric, definition, value, unit, evidence_id, quote):
    return {
        "company": company, "fiscal_year": fiscal_year, "metric": metric,
        "definition": definition, "value_text": value, "unit": unit,
        "evidence_id": evidence_id, "supporting_quote": quote,
    }


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


def test_company_comparison_retrieves_and_cites_each_company_separately():
    infosys = scoped_point(
        "infosys-revenue", "Infosys", "2024-25",
        "Our revenues in fiscal 2025 were ₹1,62,990 crore.",
    )
    tcs = scoped_point(
        "tcs-revenue", "TCS", "2024-25",
        "Consolidated revenue from operations was ₹2,55,324 crore.",
    )
    retriever = ScopedRetriever({
        ("Infosys", "2024-25"): [infosys], ("TCS", "2024-25"): [tcs],
    })
    generator = FakeGenerator(draft(
        answer="Infosys reported ₹1,62,990 crore and TCS reported ₹2,55,324 crore.",
        citations=[
            {"evidence_id": "E1", "quote": infosys.payload["text"]},
            {"evidence_id": "E2", "quote": tcs.payload["text"]},
        ],
        comparable_facts=[
            comparison_fact("Infosys", "2024-25", "revenue", "consolidated revenue from operations", "1,62,990", "crore", "E1", infosys.payload["text"]),
            comparison_fact("TCS", "2024-25", "revenue", "consolidated revenue from operations", "2,55,324", "crore", "E2", tcs.payload["text"]),
        ],
    ))
    result = answer_question(
        "Compare Infosys and TCS revenue in FY2024-25.",
        retriever=retriever, generator=generator,
    )
    assert not result.abstained
    assert {citation.company for citation in result.citations} == {"Infosys", "TCS"}
    assert [call[2] for call in retriever.calls] == [
        {"company": "Infosys", "fiscal_year": "2024-25"},
        {"company": "TCS", "fiscal_year": "2024-25"},
    ]
    assert [call[0] for call in retriever.calls] == [
        "Infosys revenue total operations FY2024-25",
        "TCS revenue total operations FY2024-25",
    ]
    assert all(call[1] == 5 for call in retriever.calls)
    assert generator.calls[0][1]["comparison_scope"] == {
        "companies": ["Infosys", "TCS"], "fiscal_years": ["2024-25"],
    }
    assert [item["company"] for item in generator.calls[0][1]["evidence"]] == [
        "Infosys", "TCS",
    ]


def test_company_comparison_recovers_currency_scale_from_report_heading():
    infosys = scoped_point(
        "infosys-revenue", "Infosys", "2024-25",
        "Our total revenues in fiscal 2025 were ₹1,62,990 crore.",
    )
    wipro = scoped_point(
        "wipro-revenue", "Wipro", "2024-25",
        "Consolidated Statements of Profit and Loss. Revenue from operations "
        "890,884 897,603. (₹ in millions, except per share data).",
    )
    retriever = ScopedRetriever({
        ("Infosys", "2024-25"): [infosys], ("Wipro", "2024-25"): [wipro],
    })
    generated = draft(
        answer="Infosys reported ₹1,62,990 crore and Wipro reported ₹890,884.",
        citations=[
            {"evidence_id": "E1", "quote": infosys.payload["text"]},
            {"evidence_id": "E2", "quote": wipro.payload["text"]},
        ],
        comparable_facts=[
            comparison_fact(
                "Infosys", "2024-25", "revenue", "consolidated revenue",
                "1,62,990", "₹ crore", "E1", infosys.payload["text"],
            ),
            comparison_fact(
                "Wipro", "2024-25", "revenue", "consolidated revenue from operations",
                "890,884", "₹ crore", "E2", wipro.payload["text"],
            ),
        ],
    )

    result = answer_question(
        "Compare Infosys and Wipro revenue in FY2024-25.",
        retriever=retriever, generator=FakeGenerator(generated),
    )

    assert not result.abstained, result.citation_validation_error
    assert "890,884 ₹ million" in result.answer
    assert "89,088.4 ₹ crore" in result.answer
    assert "73,901.6 ₹ crore" in result.answer


def test_company_revenue_comparison_falls_back_when_model_refuses():
    infosys = scoped_point(
        "infosys-revenue", "Infosys", "2024-25",
        "Total revenue from operations 1,62,990 1,53,670 (In ₹ crore).",
    )
    wipro = scoped_point(
        "wipro-revenue", "Wipro", "2024-25",
        "Revenue from operations 22 890,884 897,603. "
        "Consolidated Statements of Profit and Loss (₹ in millions).",
    )
    retriever = ScopedRetriever({
        ("Infosys", "2024-25"): [infosys], ("Wipro", "2024-25"): [wipro],
    })
    refused = draft(
        answerable=False, evidence_sufficiency=0.0, answer="", citations=[],
        comparable_facts=None,
    )

    result = answer_question(
        "“Compare Wipro and Infosys revenue in FY2024-25.",
        retriever=retriever, generator=FakeGenerator(refused),
    )

    assert not result.abstained
    assert "Infosys (FY2024-25)" in result.answer
    assert "Wipro (FY2024-25)" in result.answer
    assert "89,088.4 ₹ crore" in result.answer
    assert "73,901.6 ₹ crore" in result.answer


def test_company_comparison_returns_grounded_partial_answer_when_one_side_is_missing():
    infosys = scoped_point(
        "infosys-revenue", "Infosys", "2024-25",
        "Our revenues in fiscal 2025 were ₹1,62,990 crore.",
    )
    retriever = ScopedRetriever({("Infosys", "2024-25"): [infosys]})
    generator = FakeGenerator(draft(
        answer="Infosys was found; TCS was not found in the retrieved evidence.",
        citations=[{"evidence_id": "E1", "quote": infosys.payload["text"]}],
        comparable_facts=[comparison_fact(
            "Infosys", "2024-25", "revenue", "consolidated revenue",
            "1,62,990", "crore", "E1", infosys.payload["text"],
        )],
    ))
    result = answer_question(
        "Compare Infosys and TCS revenue in FY2024-25.",
        retriever=retriever, generator=generator,
    )
    assert not result.abstained
    assert "Infosys" in result.answer and "1,62,990" in result.answer
    assert "No validated figure was found for TCS FY2024-25" in result.answer
    assert "Calculated" not in result.answer
    assert {citation.company for citation in result.citations} == {"Infosys"}


def test_year_comparison_retrieves_each_year_separately():
    older = scoped_point("older", "Infosys", "2023-24", "Employees globally 3,17,240")
    newer = scoped_point("newer", "Infosys", "2024-25", "Employees globally 3,23,578")
    retriever = ScopedRetriever({
        ("Infosys", "2023-24"): [older], ("Infosys", "2024-25"): [newer],
    })
    generator = FakeGenerator(draft(
        answer="Employees increased from 3,17,240 to 3,23,578.",
        citations=[
            {"evidence_id": "E1", "quote": older.payload["text"]},
            {"evidence_id": "E2", "quote": newer.payload["text"]},
        ],
        comparable_facts=[
            comparison_fact("Infosys", "2023-24", "employees", "Employees globally", "3,17,240", "employees", "E1", older.payload["text"]),
            comparison_fact("Infosys", "2024-25", "employees", "Employees globally", "3,23,578", "employees", "E2", newer.payload["text"]),
        ],
    ))
    result = answer_question(
        "How did Infosys employees change from FY2023-24 to FY2024-25?",
        retriever=retriever, generator=generator,
    )
    assert not result.abstained
    assert {citation.fiscal_year for citation in result.citations} == {"2023-24", "2024-25"}
    assert [call[2]["fiscal_year"] for call in retriever.calls] == ["2023-24", "2024-25"]
    assert all("globally permanent" in call[0] for call in retriever.calls)


def test_two_company_two_year_comparison_guarantees_all_four_groups():
    results = {}
    for company in ("Infosys", "TCS"):
        for fiscal_year in ("2023-24", "2024-25"):
            results[(company, fiscal_year)] = [scoped_point(
                f"{company}-{fiscal_year}", company, fiscal_year,
                f"{company} metric for {fiscal_year} was 100.",
            )]
    retriever = ScopedRetriever(results)
    generator = FakeGenerator(draft(answerable=False, answer="", citations=[]))
    result = answer_question(
        "Compare Infosys and TCS from FY2023-24 to FY2024-25.",
        retriever=retriever, generator=generator, top_k=3,
    )
    assert result.abstained
    assert len(generator.calls[0][1]["evidence"]) == 4
    assert len(retriever.calls) == 4


def test_comparison_merges_full_top_k_per_side_up_to_ten_evidence():
    results = {
        (company, "2024-25"): [
            scoped_point(
                f"{company}-{rank}", company, "2024-25",
                f"{company} revenue evidence rank {rank}.",
            )
            for rank in range(1, 6)
        ]
        for company in ("Infosys", "TCS")
    }
    retriever = ScopedRetriever(results)
    generator = FakeGenerator(draft(answerable=False, answer="", citations=[]))
    result = answer_question(
        "Compare Infosys and TCS revenue in FY2024-25.",
        retriever=retriever, generator=generator,
    )
    assert result.abstained
    evidence = generator.calls[0][1]["evidence"]
    assert len(evidence) == 10
    assert [item["company"] for item in evidence[:4]] == ["Infosys", "TCS"] * 2


def test_balanced_comparisons_can_be_disabled_for_frozen_evaluation():
    retriever = FakeRetriever([point()])
    generator = FakeGenerator(draft(answerable=False, answer="", citations=[]))
    answer_question(
        "Compare Infosys and TCS revenue in FY2024-25.",
        retriever=retriever,
        generator=generator,
        balanced_comparisons=False,
    )
    assert retriever.calls == [(
        "Compare Infosys and TCS revenue in FY2024-25.",
        5,
        {"company": ("Infosys", "TCS"), "fiscal_year": ("2024-25",)},
    )]


def test_company_comparison_rejects_segment_revenue_as_company_total():
    infosys = scoped_point(
        "infosys-total", "Infosys", "2024-25",
        "Our revenues in fiscal 2025 were ₹1,62,990 crore.",
    )
    tcs_segment = scoped_point(
        "tcs-segment", "TCS", "2024-25",
        "Segment Revenue was ₹94,597 crore.",
    )
    retriever = ScopedRetriever({
        ("Infosys", "2024-25"): [infosys],
        ("TCS", "2024-25"): [tcs_segment],
    })
    generator = FakeGenerator(draft(
        answer="Infosys reported ₹1,62,990 crore and TCS reported ₹94,597 crore.",
        citations=[
            {"evidence_id": "E1", "quote": infosys.payload["text"]},
            {"evidence_id": "E2", "quote": tcs_segment.payload["text"]},
        ],
    ))
    result = answer_question(
        "Compare Infosys and TCS revenue in FY2024-25.",
        retriever=retriever, generator=generator, citation_retries=0,
    )
    assert result.abstained
    assert result.abstention_reason == "invalid_citations"
    assert "company-wide revenue" in result.citation_validation_error


def test_generic_revenue_comparison_filters_mixed_and_segment_passages():
    infosys_mixed = scoped_point(
        "infosys-mixed", "Infosys", "2024-25",
        "Standalone Consolidated Revenue from operations 1,36,592 1,62,990.",
    )
    infosys_total = scoped_point(
        "infosys-total", "Infosys", "2024-25",
        "Total revenue from operations was ₹1,62,990 crore.",
    )
    tcs_segment = scoped_point(
        "tcs-segment", "TCS", "2024-25",
        "Segment Revenue was ₹94,597 crore; Segment result was ₹25,000 crore.",
    )
    tcs_total = scoped_point(
        "tcs-total", "TCS", "2024-25",
        "Consolidated revenue from operations was ₹2,55,324 crore.",
    )
    retriever = ScopedRetriever({
        ("Infosys", "2024-25"): [infosys_mixed, infosys_total],
        ("TCS", "2024-25"): [tcs_segment, tcs_total],
    })
    generator = FakeGenerator(draft(
        answer="Infosys reported ₹1,62,990 crore; TCS reported ₹2,55,324 crore.",
        citations=[
            {"evidence_id": "E1", "quote": infosys_total.payload["text"]},
            {"evidence_id": "E2", "quote": tcs_total.payload["text"]},
        ],
        comparable_facts=[
            comparison_fact("Infosys", "2024-25", "revenue", "consolidated revenue from operations", "1,62,990", "crore", "E1", infosys_total.payload["text"]),
            comparison_fact("TCS", "2024-25", "revenue", "consolidated revenue from operations", "2,55,324", "crore", "E2", tcs_total.payload["text"]),
        ],
    ))
    result = answer_question(
        "Compare Infosys and TCS revenue in FY2024-25.",
        retriever=retriever, generator=generator,
    )
    assert not result.abstained
    assert [item["chunk_id"] for item in generator.calls[0][1]["evidence"]] == [
        "infosys-total", "tcs-total",
    ]


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


class SequenceGenerator(FakeGenerator):
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def generate(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, json.loads(user_prompt)))
        return next(self.outputs)


def test_repairs_quote_from_extracted_infosys_text_without_relaxing_validation():
    text = "Business highlights Revenues ₹ 1,62,990 cr 6.1% growth Y-o-Y"
    evidence = point(text)
    evidence.payload.update(company="Infosys", doc_id="infosys_fy2025")
    bad = draft(answer="Infosys reported ₹1,62,990 crore.", citations=[{
        "evidence_id": "E1", "quote": "Revenue was ₹1,62,990 crore",
    }])
    good = draft(answer=bad.answer, citations=[{
        "evidence_id": "E1", "quote": "Revenues ₹ 1,62,990 cr",
    }])
    generator = SequenceGenerator([bad, good])
    retriever = FakeRetriever([evidence])
    result = answer_question("Infosys revenue FY2024-25?", retriever=retriever, generator=generator)
    assert not result.abstained
    assert result.citations[0].quote in text
    assert result.citations[0].company == "Infosys"
    assert result.citation_validation_error is None
    assert len(retriever.calls) == 1
    assert len(generator.calls) == 2
    repair = generator.calls[1][1]
    assert repair["evidence"] == generator.calls[0][1]["evidence"]
    assert repair["previous_response"] == bad.model_dump()
    assert "not present" in repair["validation_error"]


@pytest.mark.parametrize("repair,reason", [
    (draft(citations=[{"evidence_id": "E1", "quote": "Still fabricated"}]), "invalid_citations"),
    (draft(answerable=False), "model_says_unanswerable"),
    (draft(evidence_sufficiency=0.5), "below_evidence_threshold"),
])
def test_repair_is_bounded_and_rechecks_all_gates(repair, reason):
    bad = draft(citations=[{"evidence_id": "E1", "quote": "Fabricated quote"}])
    generator = SequenceGenerator([bad, repair])
    result = answer_question("Question", retriever=FakeRetriever([point()]), generator=generator)
    assert result.abstention_reason == reason
    assert result.citations == []
    assert len(generator.calls) == 2


def test_frozen_single_pass_can_disable_repair():
    generator = SequenceGenerator([draft(citations=[])])
    result = answer_question("Question", retriever=FakeRetriever([point()]), generator=generator, citation_retries=0)
    assert result.abstained
    assert len(generator.calls) == 1


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


def test_groq_does_not_retry_rate_limit():
    calls = []
    sleeps = []

    def opener(request, timeout):
        calls.append(request)
        raise HTTPError(request.full_url, 429, "limited", {"Retry-After": "2"}, None)

    generator = GroqGenerator(
        "secret-key", "model", "https://example.test",
        opener=opener, sleeper=sleeps.append,
    )
    with pytest.raises(GroqGenerationError) as caught:
        generator.generate("system", "user")
    assert caught.value.status_code == 429
    assert len(calls) == 1
    assert sleeps == []


def test_groq_retries_transient_server_failure():
    calls, sleeps = [], []

    def opener(request, timeout):
        calls.append(request)
        if len(calls) == 1:
            raise HTTPError(request.full_url, 503, "temporary", {}, None)
        return FakeResponse({"choices": [{"message": {"content": draft().model_dump_json()}}]})

    generator = GroqGenerator(
        "secret-key", "model", "https://example.test", opener=opener, sleeper=sleeps.append,
    )
    assert generator.generate("system", "user").answerable
    assert len(calls) == 2
    assert sleeps == [1.0]


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
