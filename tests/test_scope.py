"""Query scope inference and multi-value retrieval filters."""

from app.retrieval.dense import build_search_filter
from app.retrieval.scope import infer_query_scope
from qdrant_client import models


def test_scope_normalizes_fiscal_year_forms() -> None:
    scope = infer_query_scope(
        "Compare Infosys from FY2023-24 with TCS in FY2025 and March 31, 2024."
    )
    assert scope.companies == ("Infosys", "TCS")
    assert scope.fiscal_years == ("2023-24", "2024-25")


def test_scope_recognizes_future_year_without_inventing_corpus_match() -> None:
    assert infer_query_scope("TCS revenue in FY2025-26").fiscal_years == ("2025-26",)


def test_scope_recognizes_company_aliases_and_fiscal_wording() -> None:
    scope = infer_query_scope(
        "Compare Infosys Limited in fiscal year 2025 with Tata Consultancy Services."
    )
    assert scope.companies == ("Infosys", "TCS")
    assert scope.fiscal_years == ("2024-25",)


def test_scope_resolves_previous_year_follow_up_from_history() -> None:
    scope = infer_query_scope(
        "user: What rate did TCS report for FY2024-25?\n"
        "assistant: It reported 13.3%.\n"
        "What was the rate in the previous fiscal year?"
    )
    assert scope.companies == ("TCS",)
    assert scope.fiscal_years == ("2023-24",)


def test_dense_filter_uses_match_any_for_comparisons() -> None:
    query_filter = build_search_filter(
        company=("Infosys", "TCS"),
        fiscal_year=("2023-24", "2024-25"),
    )
    assert query_filter is not None
    conditions = {condition.key: condition for condition in query_filter.must or []}
    assert isinstance(conditions["company"].match, models.MatchAny)
    assert conditions["company"].match.any == ["Infosys", "TCS"]
    assert conditions["fiscal_year"].match.any == ["2023-24", "2024-25"]
