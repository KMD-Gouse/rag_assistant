"""Infer explicit company and fiscal-year constraints from a retrieval query."""

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class QueryScope:
    companies: tuple[str, ...] = ()
    fiscal_years: tuple[str, ...] = ()


def _ending_year_to_fiscal_year(ending_year: int) -> str:
    return f"{ending_year - 1}-{ending_year % 100:02d}"


def _previous_fiscal_year(fiscal_year: str) -> str:
    starting_year = int(fiscal_year[:4]) - 1
    return f"{starting_year}-{(starting_year + 1) % 100:02d}"


def normalize_fiscal_year(fy: str) -> str:
    """Normalize a fiscal-year string to ``YYYY-YY`` format.

    Accepts:
      - ``"FY2023-24"``  → ``"2023-24"``
      - ``"FY2023-2024"`` → ``"2023-24"``
      - ``"2023-2024"``  → ``"2023-24"``
      - ``"2023-24"``    → ``"2023-24"``
    """
    fy = fy.strip()
    # Strip leading FY / fy prefix
    fy = re.sub(r"^(?:FY|fiscal\s+year)\s*", "", fy, flags=re.IGNORECASE)
    match = re.match(r"^((?:20)?\d{2})\s*[-–]\s*(\d{2,4})$", fy)
    if match:
        start_number = int(match.group(1))
        start = str(start_number if start_number >= 2000 else 2000 + start_number)
        suffix = str(int(match.group(2)) % 100).zfill(2)
        return f"{start}-{suffix}"
    # Bare 4-digit year with no suffix
    match = re.match(r"^(20\d{2})$", fy)
    if match:
        return _ending_year_to_fiscal_year(int(match.group(1)))
    return fy


def infer_query_scope(query: str) -> QueryScope:
    """Return constraints explicitly stated in the question or supplied history."""

    companies = tuple(
        company
        for company, pattern in (
            ("Infosys", r"\bInfosys(?:\s+Limited)?\b"),
            ("TCS", r"\b(?:TCS|Tata\s+Consultancy\s+Services(?:\s+Limited)?)\b"),
            ("Wipro", r"\bWipro(?:\s+Limited)?\b"),
        )
        if re.search(pattern, query, re.IGNORECASE)
    )
    years: set[str] = set()
    # Short conversational form: "year 24-25", "FY24-25", or
    # "fiscal year 24-25". Annual-report years are interpreted as 20xx.
    for match in re.finditer(
        r"\b(?:FY|(?:fiscal\s+)?year)\s*(\d{2})\s*[-–]\s*(\d{2})\b",
        query,
        re.IGNORECASE,
    ):
        years.add(normalize_fiscal_year(f"{match.group(1)}-{match.group(2)}"))
    # Primary pattern: FY2023-24 or FY2023-2024 or FY2024 (with optional 2- or 4-digit suffix)
    for match in re.finditer(
        r"\b(?:FY|fiscal(?:\s+year)?)\s*(20\d{2})(?:\s*[-–]\s*(\d{2,4}))?\b",
        query,
        re.IGNORECASE,
    ):
        first = int(match.group(1))
        suffix = match.group(2)
        if suffix:
            normalized_suffix = str(int(suffix) % 100).zfill(2)
            years.add(f"{first}-{normalized_suffix}")
        else:
            years.add(_ending_year_to_fiscal_year(first))
    # Bare-year pattern: 2023-24 or 2023-2024
    for match in re.finditer(r"\b(20\d{2})\s*[-–]\s*(\d{2,4})\b", query):
        first = match.group(1)
        suffix = str(int(match.group(2)) % 100).zfill(2)
        years.add(f"{first}-{suffix}")
    for match in re.finditer(r"\bMarch\s+31,?\s+(20\d{2})\b", query, re.IGNORECASE):
        years.add(_ending_year_to_fiscal_year(int(match.group(1))))
    final_question = next(
        (line for line in reversed(query.splitlines()) if line.strip()),
        query,
    )
    if re.search(r"\bprevious\s+(?:fiscal\s+)?year\b", final_question, re.IGNORECASE):
        years = {_previous_fiscal_year(year) for year in years}
    return QueryScope(companies=companies, fiscal_years=tuple(sorted(years)))


def inferred_filters(query: str) -> dict[str, str | tuple[str, ...] | None]:
    """Convert explicit query scope to retriever-compatible filters."""

    scope = infer_query_scope(query)
    return {
        "company": scope.companies or None,
        "fiscal_year": scope.fiscal_years or None,
    }
