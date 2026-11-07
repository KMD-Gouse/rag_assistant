"""Deterministic arithmetic for validated comparison facts.

All difference and percentage-change calculations are done here in Python.
The LLM is never asked to compute numbers — it only supplies raw value strings.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation


def _parse_number(text: str) -> Decimal | None:
    """Parse a localised number string into a Decimal.

    Handles:
    - Indian comma grouping: ``"1,62,990"`` → ``162990``
    - Percentage suffix:     ``"21.1%"``    → ``21.1``
    - Plain integer/float:   ``"3,23,578"`` → ``323578``
    """
    cleaned = re.sub(r"[,%\s₹$\u20b9]", "", text.strip())
    if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", cleaned):
        return None
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def _normalise_unit_value(value: Decimal, unit: str) -> tuple[Decimal, str]:
    """Normalise compatible Indian-report currency units to INR crore."""
    lowered = unit.casefold().replace("₹", " ").replace("inr", " ").strip()
    if "crore" in lowered:
        return value, "₹ crore"
    if "million" in lowered:
        return value / Decimal("10"), "₹ crore"
    if "billion" in lowered:
        return value * Decimal("100"), "₹ crore"
    return value, unit


def calculate_comparison(
    lhs_text: str,
    rhs_text: str,
    lhs_unit: str | None = None,
    rhs_unit: str | None = None,
) -> dict | None:
    """Compute arithmetic comparison between two value strings.

    Parameters
    ----------
    lhs_text:
        The first (base / older) value as text, e.g. ``"3,17,240"`` or ``"20.7%"``.
    rhs_text:
        The second (newer / compared) value as text.

    Returns
    -------
    dict with keys:
        ``lhs_value``  – parsed Decimal
        ``rhs_value``  – parsed Decimal
        ``difference`` – rhs_value − lhs_value
        ``pct_change`` – (difference / lhs_value) × 100, rounded to 2 dp
        ``direction``  – ``"increase"`` | ``"decrease"`` | ``"unchanged"``

    Returns ``None`` when either value cannot be parsed or lhs is zero
    (to avoid division-by-zero).
    """
    lhs = _parse_number(lhs_text)
    rhs = _parse_number(rhs_text)
    result_unit = lhs_unit or ""
    if lhs is None or rhs is None:
        return None
    if lhs_unit is not None and rhs_unit is not None:
        lhs, normalised_lhs_unit = _normalise_unit_value(lhs, lhs_unit)
        rhs, normalised_rhs_unit = _normalise_unit_value(rhs, rhs_unit)
        if normalised_lhs_unit.casefold() != normalised_rhs_unit.casefold():
            return None
        result_unit = normalised_lhs_unit
    if lhs == 0:
        return None
    diff = rhs - lhs
    pct = (diff / lhs * 100).quantize(Decimal("0.01"))
    if diff > 0:
        direction = "increase"
    elif diff < 0:
        direction = "decrease"
    else:
        direction = "unchanged"
    return {
        "lhs_value": lhs,
        "rhs_value": rhs,
        "difference": diff,
        "pct_change": pct,
        "direction": direction,
        "unit": result_unit,
    }


def format_comparison_sentence(
    lhs_label: str,
    rhs_label: str,
    calc: dict,
    unit: str = "",
) -> str:
    """Return a human-readable sentence from a ``calculate_comparison`` result.

    Example::

        "TCS: 317,240 vs Infosys: 323,578; a difference of +6,338 (+2.00%)."
    """
    unit_str = f" {unit}" if unit else ""
    difference_unit = " percentage points" if unit.casefold() in {"%", "percent", "percentage"} else unit_str
    return (
        f"{lhs_label}: {calc['lhs_value']:,}{unit_str} vs "
        f"{rhs_label}: {calc['rhs_value']:,}{unit_str}; "
        f"a difference of {calc['difference']:+,}{difference_unit} "
        f"(second minus first; {calc['pct_change']:+}% relative to the first)."
    )
