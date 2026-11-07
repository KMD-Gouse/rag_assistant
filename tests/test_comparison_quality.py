"""Regression tests for comparison fact quality: metric alignment, figure
validation, deterministic calculations, and fiscal-year parsing.

Run with:
    .venv\\Scripts\\python.exe -m pytest tests/test_comparison_quality.py -v
"""

from __future__ import annotations

import types
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_point(company: str, fiscal_year: str, text: str, pid: int = 1):
    """Build a fake Qdrant ScoredPoint-like object."""
    point = MagicMock()
    point.id = pid
    point.payload = {
        "company": company,
        "fiscal_year": fiscal_year,
        "text": text,
        "chunk_id": f"chunk-{pid}",
        "doc_id": f"{company.lower()}_fy{fiscal_year.replace('-', '')}",
        "pdf_page": 42,
        "printed_page": None,
        "section": "Financials",
    }
    return point


# ════════════════════════════════════════════════════════════════════════════
# FIX 1 — Fiscal-year normalisation
# ════════════════════════════════════════════════════════════════════════════

class TestFiscalYearNormalisation:
    def test_fy_with_2digit_suffix(self):
        from app.retrieval.scope import normalize_fiscal_year
        assert normalize_fiscal_year("FY2023-24") == "2023-24"

    def test_fy_with_4digit_suffix(self):
        from app.retrieval.scope import normalize_fiscal_year
        assert normalize_fiscal_year("FY2023-2024") == "2023-24"

    def test_bare_with_2digit_suffix(self):
        from app.retrieval.scope import normalize_fiscal_year
        assert normalize_fiscal_year("2023-24") == "2023-24"

    def test_bare_with_4digit_suffix(self):
        from app.retrieval.scope import normalize_fiscal_year
        assert normalize_fiscal_year("2023-2024") == "2023-24"

    def test_fy2025_with_4digit(self):
        from app.retrieval.scope import normalize_fiscal_year
        assert normalize_fiscal_year("FY2024-2025") == "2024-25"

    def test_scope_parses_4digit_suffix(self):
        from app.retrieval.scope import infer_query_scope
        scope = infer_query_scope("Infosys revenue in FY2023-2024")
        assert scope.fiscal_years == ("2023-24",)

    def test_scope_parses_2digit_suffix(self):
        from app.retrieval.scope import infer_query_scope
        scope = infer_query_scope("TCS employees in FY2024-25")
        assert scope.fiscal_years == ("2024-25",)

    def test_scope_4digit_equals_2digit(self):
        """FY2023-2024 and FY2023-24 must resolve to the same tuple."""
        from app.retrieval.scope import infer_query_scope
        s1 = infer_query_scope("Infosys FY2023-24")
        s2 = infer_query_scope("Infosys FY2023-2024")
        assert s1.fiscal_years == s2.fiscal_years


# ════════════════════════════════════════════════════════════════════════════
# FIX 3 — Deterministic calculations
# ════════════════════════════════════════════════════════════════════════════

class TestCalculateComparison:
    def test_employee_count(self):
        from app.generation.comparison import calculate_comparison
        result = calculate_comparison("3,17,240", "3,23,578")
        assert result is not None
        assert result["difference"] == Decimal("6338")
        assert result["pct_change"] == Decimal("2.00")
        assert result["direction"] == "increase"

    def test_revenue_crore(self):
        from app.generation.comparison import calculate_comparison
        result = calculate_comparison("1,53,670", "1,62,990")
        assert result is not None
        assert result["difference"] == Decimal("9320")
        assert result["direction"] == "increase"

    def test_converts_million_to_crore_before_comparing(self):
        from app.generation.comparison import calculate_comparison
        result = calculate_comparison(
            "1,62,990", "890,884", "₹ crore", "₹ million",
        )
        assert result is not None
        assert result["rhs_value"] == Decimal("89088.4")
        assert result["difference"] == Decimal("-73901.6")
        assert result["pct_change"] == Decimal("-45.34")
        assert result["unit"] == "₹ crore"

    def test_margin_percentage(self):
        from app.generation.comparison import calculate_comparison
        result = calculate_comparison("20.7%", "21.1%")
        assert result is not None
        assert result["difference"] == Decimal("0.4")
        assert result["direction"] == "increase"

    def test_decrease(self):
        from app.generation.comparison import calculate_comparison
        result = calculate_comparison("1,62,990", "1,53,670")
        assert result is not None
        assert result["direction"] == "decrease"
        assert result["difference"] == Decimal("-9320")

    def test_unchanged(self):
        from app.generation.comparison import calculate_comparison
        result = calculate_comparison("100", "100")
        assert result is not None
        assert result["direction"] == "unchanged"
        assert result["pct_change"] == Decimal("0.00")

    def test_returns_none_for_unparseable(self):
        from app.generation.comparison import calculate_comparison
        assert calculate_comparison("N/A", "100") is None
        assert calculate_comparison("100", "N/A") is None

    def test_returns_none_for_zero_base(self):
        from app.generation.comparison import calculate_comparison
        assert calculate_comparison("0", "100") is None


# ════════════════════════════════════════════════════════════════════════════
# FIX 5 — Figure validation: answer number must appear in cited quote
# ════════════════════════════════════════════════════════════════════════════

class TestFigureValidation:
    """_validate_comparison_citations should reject answers where large numbers
    in the prose are absent from the citation quotes."""

    def _run_validate(self, answer_text: str, quote_text: str, companies=("TCS", "Infosys")):
        from app.generation.service import _validate_comparison_citations
        from app.generation.models import AnswerDraft, DraftCitation, ComparableFact
        from app.retrieval.scope import QueryScope

        scope = QueryScope(companies=companies, fiscal_years=("2024-25",))

        # Two citations, one per company, both with the same quote_text for simplicity
        citations = [
            DraftCitation(evidence_id="E1", quote=quote_text),
            DraftCitation(evidence_id="E2", quote=quote_text),
        ]
        draft = AnswerDraft(
            answerable=True,
            evidence_sufficiency=0.9,
            answer=answer_text,
            citations=citations,
            comparable_facts=[
                ComparableFact(
                    company=company, fiscal_year="2024-25", metric="employees",
                    definition="permanent employees", value_text="3,23,578",
                    unit="employees", evidence_id=f"E{index}",
                    supporting_quote=quote_text,
                )
                for index, company in enumerate(companies, start=1)
            ],
        )

        point_tcs = _make_point("TCS", "2024-25", quote_text, pid=1)
        point_inf = _make_point("Infosys", "2024-25", quote_text, pid=2)
        lookup = {"E1": point_tcs, "E2": point_inf}

        return draft, scope, lookup

    def test_correct_figures_pass(self):
        from app.generation.service import _validate_comparison_citations
        from app.generation.citations import CitationValidationError

        draft, scope, lookup = self._run_validate(
            answer_text="TCS had 323578 employees vs Infosys 323578.",
            quote_text="permanent employees: 3,23,578 as of March 2025",
        )
        # Should NOT raise
        _validate_comparison_citations("Compare TCS and Infosys employees FY2024-25", scope, draft, lookup)

    def test_model_arithmetic_is_ignored_when_validated_facts_are_present(self):
        from app.generation.service import _validate_comparison_citations
        from app.generation.citations import CitationValidationError

        draft, scope, lookup = self._run_validate(
            # Answer mentions 340687 but quotes only contain 323578
            answer_text="TCS had 340687 employees.",
            quote_text="permanent employees: 3,23,578 as of March 2025",
        )
        _validate_comparison_citations(
            "Compare TCS and Infosys employees FY2024-25", scope, draft, lookup
        )


# ════════════════════════════════════════════════════════════════════════════
# FIX 5 — Metric definition alignment check
# ════════════════════════════════════════════════════════════════════════════

class TestDefinitionAlignment:
    def test_mismatched_definitions_rejected(self):
        from app.generation.service import _validate_comparison_citations
        from app.generation.citations import CitationValidationError
        from app.generation.models import AnswerDraft, DraftCitation, ComparableFact
        from app.retrieval.scope import QueryScope

        scope = QueryScope(companies=("TCS", "Infosys"), fiscal_years=("2024-25",))
        quote = "employees 3,23,578"

        facts = [
            ComparableFact(
                company="TCS", fiscal_year="2024-25", metric="employees",
                definition="permanent employees",
                value_text="3,40,687", unit="employees",
                evidence_id="E1", supporting_quote=quote,
            ),
            ComparableFact(
                company="Infosys", fiscal_year="2024-25", metric="employees",
                definition="total workforce including contractors",
                value_text="3,23,578", unit="employees",
                evidence_id="E2", supporting_quote="permanent employees: 323578",
            ),
        ]
        draft = AnswerDraft(
            answerable=True,
            evidence_sufficiency=0.9,
            answer="TCS 340687, Infosys 323578",
            citations=[
                DraftCitation(evidence_id="E1", quote=quote),
                DraftCitation(evidence_id="E2", quote=quote),
            ],
            comparable_facts=facts,
        )
        point_tcs = _make_point("TCS", "2024-25", "employees 3,40,687", pid=1)
        point_inf = _make_point("Infosys", "2024-25", "employees 3,23,578", pid=2)
        lookup = {"E1": point_tcs, "E2": point_inf}

        with pytest.raises(CitationValidationError, match="not aligned"):
            _validate_comparison_citations(
                "Compare TCS and Infosys employees FY2024-25", scope, draft, lookup
            )

    def test_matching_definitions_pass(self):
        from app.generation.service import _validate_comparison_citations
        from app.generation.citations import CitationValidationError
        from app.generation.models import AnswerDraft, DraftCitation, ComparableFact
        from app.retrieval.scope import QueryScope

        scope = QueryScope(companies=("TCS", "Infosys"), fiscal_years=("2024-25",))
        quote = "permanent employees: 317240"

        facts = [
            ComparableFact(
                company="TCS", fiscal_year="2024-25", metric="employees",
                definition="permanent employees",
                value_text="317240", unit="employees",
                evidence_id="E1", supporting_quote=quote,
            ),
            ComparableFact(
                company="Infosys", fiscal_year="2024-25", metric="employees",
                definition="permanent employees",
                value_text="323578", unit="employees",
                evidence_id="E2", supporting_quote="permanent employees: 323578",
            ),
        ]
        tcs_quote = "permanent employees: 317240"
        inf_quote = "permanent employees: 323578"
        draft = AnswerDraft(
            answerable=True,
            evidence_sufficiency=0.9,
            answer="TCS 317240 vs Infosys 323578",
            citations=[
                DraftCitation(evidence_id="E1", quote=tcs_quote),
                DraftCitation(evidence_id="E2", quote=inf_quote),
            ],
            comparable_facts=facts,
        )
        point_tcs = _make_point("TCS", "2024-25", tcs_quote, pid=1)
        point_inf = _make_point("Infosys", "2024-25", inf_quote, pid=2)
        lookup = {"E1": point_tcs, "E2": point_inf}

        # Should not raise
        _validate_comparison_citations(
            "Compare TCS and Infosys employees FY2024-25", scope, draft, lookup
        )


# ════════════════════════════════════════════════════════════════════════════
# FIX 6 — Evidence recovery for missing comparison side
# ════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════
# FIX 4 — Per-fact integrity validation
# ════════════════════════════════════════════════════════════════════════════

class TestPerFactIntegrity:
    """Each ComparableFact must be validated against the chunk it references."""

    def _make_draft_with_facts(self, facts, tcs_text, inf_text):
        from app.generation.models import AnswerDraft, DraftCitation, ComparableFact
        from app.retrieval.scope import QueryScope

        scope = QueryScope(companies=("TCS", "Infosys"), fiscal_years=("2024-25",))
        tcs_quote = tcs_text[:40]
        inf_quote = inf_text[:40]

        draft = AnswerDraft(
            answerable=True,
            evidence_sufficiency=0.9,
            answer="TCS 317240, Infosys 323578",
            citations=[
                DraftCitation(evidence_id="E1", quote=tcs_quote),
                DraftCitation(evidence_id="E2", quote=inf_quote),
            ],
            comparable_facts=facts,
        )
        point_tcs = _make_point("TCS", "2024-25", tcs_text, pid=1)
        point_inf = _make_point("Infosys", "2024-25", inf_text, pid=2)
        lookup = {"E1": point_tcs, "E2": point_inf}
        return draft, scope, lookup

    def test_wrong_company_in_fact_rejected(self):
        from app.generation.service import _validate_comparison_citations
        from app.generation.citations import CitationValidationError
        from app.generation.models import ComparableFact

        # fact says company=Infosys but evidence_id=E1 belongs to TCS
        facts = [
            ComparableFact(
                company="Infosys",        # ← wrong: E1 is TCS
                fiscal_year="2024-25",
                metric="employees",
                definition="permanent employees",
                value_text="317240",
                unit="employees",
                evidence_id="E1",
                supporting_quote="permanent employees: 317240",
            ),
            ComparableFact(
                company="Infosys",
                fiscal_year="2024-25",
                metric="employees",
                definition="permanent employees",
                value_text="323578",
                unit="employees",
                evidence_id="E2",
                supporting_quote="permanent employees: 323578",
            ),
        ]
        draft, scope, lookup = self._make_draft_with_facts(
            facts,
            tcs_text="TCS permanent employees: 317240 as of March 2025",
            inf_text="Infosys permanent employees: 323578 as of March 2025",
        )
        with pytest.raises(CitationValidationError, match="does not match"):
            _validate_comparison_citations(
                "Compare TCS and Infosys employees FY2024-25", scope, draft, lookup
            )

    def test_value_not_in_chunk_rejected(self):
        from app.generation.service import _validate_comparison_citations
        from app.generation.citations import CitationValidationError
        from app.generation.models import ComparableFact

        facts = [
            ComparableFact(
                company="TCS",
                fiscal_year="2024-25",
                metric="employees",
                definition="permanent employees",
                value_text="999999",   # ← invented; not in chunk
                unit="employees",
                evidence_id="E1",
                supporting_quote="permanent employees: 317240",
            ),
            ComparableFact(
                company="Infosys",
                fiscal_year="2024-25",
                metric="employees",
                definition="permanent employees",
                value_text="323578",
                unit="employees",
                evidence_id="E2",
                supporting_quote="permanent employees: 323578",
            ),
        ]
        draft, scope, lookup = self._make_draft_with_facts(
            facts,
            tcs_text="TCS permanent employees: 317240 as of March 2025",
            inf_text="Infosys permanent employees: 323578 as of March 2025",
        )
        with pytest.raises(CitationValidationError, match="value_text"):
            _validate_comparison_citations(
                "Compare TCS and Infosys employees FY2024-25", scope, draft, lookup
            )

    def test_valid_facts_pass(self):
        from app.generation.service import _validate_comparison_citations
        from app.generation.citations import CitationValidationError
        from app.generation.models import ComparableFact

        facts = [
            ComparableFact(
                company="TCS",
                fiscal_year="2024-25",
                metric="employees",
                definition="permanent employees",
                value_text="317240",
                unit="employees",
                evidence_id="E1",
                supporting_quote="permanent employees: 317240",
            ),
            ComparableFact(
                company="Infosys",
                fiscal_year="2024-25",
                metric="employees",
                definition="permanent employees",
                value_text="323578",
                unit="employees",
                evidence_id="E2",
                supporting_quote="permanent employees: 323578",
            ),
        ]
        draft, scope, lookup = self._make_draft_with_facts(
            facts,
            tcs_text="TCS permanent employees: 317240 as of March 2025",
            inf_text="Infosys permanent employees: 323578 as of March 2025",
        )
        # Should not raise
        _validate_comparison_citations(
            "Compare TCS and Infosys employees FY2024-25", scope, draft, lookup
        )

    def test_wrong_year_column_in_same_quote_rejected(self):
        from app.generation.service import _validate_comparison_citations
        from app.generation.citations import CitationValidationError
        from app.generation.models import AnswerDraft, DraftCitation, ComparableFact
        from app.retrieval.scope import QueryScope

        quote = "Revenue from operations 2,55,324 2,40,893"
        draft = AnswerDraft(
            answerable=True, evidence_sufficiency=0.9,
            answer="TCS revenue was 240893 crore.",
            citations=[DraftCitation(evidence_id="E1", quote=quote)],
            comparable_facts=[ComparableFact(
                company="TCS", fiscal_year="2024-25", metric="revenue",
                definition="Revenue from operations", value_text="2,40,893",
                unit="crore", evidence_id="E1", supporting_quote=quote,
            )],
        )
        lookup = {"E1": _make_point("TCS", "2024-25", quote, pid=1)}
        with pytest.raises(CitationValidationError, match="nearest the requested metric row"):
            _validate_comparison_citations(
                "Compare TCS revenue from FY2023-24 to FY2024-25",
                QueryScope(companies=("TCS",), fiscal_years=("2023-24", "2024-25")),
                draft, lookup,
            )

    def test_total_revenue_row_is_preferred_over_component_rows(self):
        from app.generation.service import _validate_comparison_citations
        from app.generation.models import AnswerDraft, DraftCitation, ComparableFact
        from app.retrieval.scope import QueryScope

        quote = (
            "Revenue from operations for the year is as follows. "
            "Revenue from software services 1,55,395 Revenue from products 7,595 "
            "Total revenue from operations 1,62,990 1,53,670"
        )
        draft = AnswerDraft(
            answerable=True, evidence_sufficiency=0.9,
            answer="Infosys revenue was 1,62,990 crore.",
            citations=[DraftCitation(evidence_id="E1", quote=quote)],
            comparable_facts=[ComparableFact(
                company="Infosys", fiscal_year="2024-25", metric="revenue",
                definition="consolidated revenue from operations", value_text="1,62,990",
                unit="crore", evidence_id="E1", supporting_quote=quote,
            )],
        )
        lookup = {"E1": _make_point("Infosys", "2024-25", quote, pid=1)}

        _validate_comparison_citations(
            "Compare Infosys revenue from FY2023-24 to FY2024-25",
            QueryScope(companies=("Infosys",), fiscal_years=("2023-24", "2024-25")),
            draft, lookup,
        )

    def test_joined_financial_statement_note_is_removed_from_value(self):
        from app.generation.service import _validate_comparison_citations
        from app.generation.models import AnswerDraft, DraftCitation, ComparableFact
        from app.retrieval.scope import QueryScope

        quote = "Revenue from operations 22 890,884 897,603 (₹ in millions)"
        fact = ComparableFact(
            company="Wipro", fiscal_year="2024-25", metric="revenue",
            definition="consolidated revenue from operations", value_text="22,890,884",
            unit="₹", evidence_id="E1", supporting_quote=quote,
        )
        draft = AnswerDraft(
            answerable=True, evidence_sufficiency=0.9,
            answer="Wipro revenue was ₹22,890,884.",
            citations=[DraftCitation(evidence_id="E1", quote=quote)],
            comparable_facts=[fact],
        )
        lookup = {"E1": _make_point("Wipro", "2024-25", quote, pid=1)}

        _validate_comparison_citations(
            "Compare Wipro revenue from FY2023-24 to FY2024-25",
            QueryScope(companies=("Wipro",), fiscal_years=("2023-24", "2024-25")),
            draft, lookup,
        )
        assert fact.value_text == "890,884"


# ════════════════════════════════════════════════════════════════════════════
# FIX 5 — Evidence recovery: only when balanced_comparisons=True
# ════════════════════════════════════════════════════════════════════════════

class TestEvidenceRecovery:
    def _build_scenario(self):
        tcs_point = _make_point("TCS", "2024-25",
            "TCS consolidated revenue from operations: 2,55,324 crore", pid=1)
        infosys_point = _make_point("Infosys", "2024-25",
            "Infosys consolidated revenue from operations: 1,62,990 crore", pid=2)
        return tcs_point, infosys_point

    def _make_answer_draft(self):
        from app.generation.models import AnswerDraft, DraftCitation, ComparableFact
        # Balanced retrieval follows canonical company order: Infosys=E1, TCS=E2.
        return AnswerDraft(
            answerable=True,
            evidence_sufficiency=0.9,
            answer=(
                "Infosys consolidated revenue from operations: 162990 crore; "
                "TCS consolidated revenue from operations: 255324 crore."
            ),
            citations=[
                DraftCitation(evidence_id="E1",
                    quote="Infosys consolidated revenue from operations: 1,62,990 crore"),
                DraftCitation(evidence_id="E2",
                    quote="TCS consolidated revenue from operations: 2,55,324 crore"),
            ],
            comparable_facts=[
                ComparableFact(
                    company="Infosys", fiscal_year="2024-25", metric="revenue",
                    definition="consolidated revenue from operations",
                    value_text="1,62,990", unit="crore", evidence_id="E1",
                    supporting_quote="Infosys consolidated revenue from operations: 1,62,990 crore",
                ),
                ComparableFact(
                    company="TCS", fiscal_year="2024-25", metric="revenue",
                    definition="consolidated revenue from operations",
                    value_text="2,55,324", unit="crore", evidence_id="E2",
                    supporting_quote="TCS consolidated revenue from operations: 2,55,324 crore",
                ),
            ],
        )

    def test_recovery_triggers_when_balanced(self):
        """balanced_comparisons=True → recovery fetches the missing Infosys side."""
        from app.generation.service import answer_question
        tcs_point, infosys_point = self._build_scenario()
        call_count = {"n": 0}

        def mock_search(question, limit=5, **filters):
            call_count["n"] += 1
            if filters.get("company") == "Infosys":
                return [infosys_point]
            # Balanced retrieval calls per-group; first call is TCS only
            return [tcs_point]

        mock_retriever = MagicMock()
        mock_retriever.search.side_effect = mock_search
        mock_generator = MagicMock()
        mock_generator.model = "test-model"
        mock_generator.generate.return_value = self._make_answer_draft()

        result = answer_question(
            "Compare TCS and Infosys revenue in FY2024-25",
            retriever=mock_retriever,
            generator=mock_generator,
            citation_mode="verbatim",
            balanced_comparisons=True,
            min_evidence_sufficiency=0.0,
        )
        assert not result.abstained, f"Unexpected abstention: {result.abstention_reason}"
        cited_companies = {c.company for c in result.citations}
        assert "TCS" in cited_companies
        assert "Infosys" in cited_companies

    def test_no_recovery_when_balanced_false(self):
        """balanced_comparisons=False → recovery never runs; only initial search used."""
        from app.generation.service import answer_question
        tcs_point, infosys_point = self._build_scenario()
        search_calls = []

        def mock_search(question, limit=5, **filters):
            search_calls.append(filters)
            return [tcs_point]   # only ever returns TCS

        mock_retriever = MagicMock()
        mock_retriever.search.side_effect = mock_search
        mock_generator = MagicMock()
        mock_generator.model = "test-model"
        # Generator abstains because Infosys evidence is absent
        from app.generation.models import AnswerDraft
        mock_generator.generate.return_value = AnswerDraft(
            answerable=False,
            evidence_sufficiency=0.0,
            answer="",
            citations=[],
            comparable_facts=None,
        )

        answer_question(
            "Compare TCS and Infosys revenue in FY2024-25",
            retriever=mock_retriever,
            generator=mock_generator,
            citation_mode="verbatim",
            balanced_comparisons=False,
            min_evidence_sufficiency=0.0,
        )
        # With balanced_comparisons=False there is only 1 search call;
        # recovery must NOT have added any Infosys-targeted extra calls.
        infosys_recovery_calls = [
            c for c in search_calls if c.get("company") == "Infosys"
        ]
        assert infosys_recovery_calls == [], (
            "Recovery must not run when balanced_comparisons=False"
        )


# ════════════════════════════════════════════════════════════════════════════
# FIX 2 — calculate_comparison is wired into the answer pipeline
# ════════════════════════════════════════════════════════════════════════════

class TestCalculationPipeline:
    def test_calculation_appended_to_answer(self):
        """When comparable_facts has 2 entries, the final answer includes the
        calculated difference and percentage."""
        from app.generation.service import answer_question
        from app.generation.models import AnswerDraft, DraftCitation, ComparableFact

        tcs_text = "TCS permanent employees: 317240 as of March 2025"
        inf_text  = "Infosys permanent employees: 323578 as of March 2025"
        tcs_point = _make_point("TCS",     "2024-25", tcs_text, pid=1)
        inf_point  = _make_point("Infosys", "2024-25", inf_text,  pid=2)

        def mock_search(question, limit=5, **filters):
            company = filters.get("company")
            if company == "Infosys":
                return [inf_point]
            return [tcs_point]

        mock_retriever = MagicMock()
        mock_retriever.search.side_effect = mock_search

        draft = AnswerDraft(
            answerable=True,
            evidence_sufficiency=0.9,
            answer="TCS had 317240 employees; Infosys had 323578 employees.",
            citations=[
                DraftCitation(evidence_id="E1", quote="permanent employees: 323578"),
                DraftCitation(evidence_id="E2", quote="permanent employees: 317240"),
            ],
            comparable_facts=[
                ComparableFact(
                    company="TCS", fiscal_year="2024-25", metric="employees",
                    definition="permanent employees", value_text="317240",
                    unit="employees", evidence_id="E2",
                    supporting_quote="permanent employees: 317240",
                ),
                ComparableFact(
                    company="Infosys", fiscal_year="2024-25", metric="employees",
                    definition="permanent employees", value_text="323578",
                    unit="employees", evidence_id="E1",
                    supporting_quote="permanent employees: 323578",
                ),
            ],
        )
        mock_generator = MagicMock()
        mock_generator.model = "test-model"
        mock_generator.generate.return_value = draft

        result = answer_question(
            "Compare TCS and Infosys employees in FY2024-25",
            retriever=mock_retriever,
            generator=mock_generator,
            citation_mode="verbatim",
            balanced_comparisons=True,
            min_evidence_sufficiency=0.0,
        )
        assert not result.abstained, result.abstention_reason
        # The calculated sentence must appear in the final answer
        assert "Calculated" in result.answer or "increase" in result.answer, (
            f"Expected calculation in answer; got:\n{result.answer}"
        )
        assert "6338" in result.answer.replace(",", ""), (
            f"Expected difference 6338 in answer; got:\n{result.answer}"
        )
