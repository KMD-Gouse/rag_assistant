"""Validation of LLM citations against the retrieved evidence only."""

import re
import unicodedata

from app.generation.models import DraftCitation, ValidatedCitation


class CitationValidationError(ValueError):
    """Raised when a generated citation cannot be proved to be retrieved."""


def _normalized_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", " ", value).strip().casefold()


def validate_citations(
    citations: list[DraftCitation],
    evidence: dict[str, object],
) -> list[ValidatedCitation]:
    """Resolve references and require each quote to occur in its cited chunk."""

    if not citations:
        raise CitationValidationError("An answer must include at least one citation.")

    validated: list[ValidatedCitation] = []
    seen: set[str] = set()
    for citation in citations:
        if citation.evidence_id in seen:
            raise CitationValidationError(f"Duplicate citation {citation.evidence_id}.")
        seen.add(citation.evidence_id)
        point = evidence.get(citation.evidence_id)
        if point is None:
            raise CitationValidationError(f"Unknown citation {citation.evidence_id}.")
        payload = getattr(point, "payload", None) or {}
        text = payload.get("text")
        quote = _normalized_text(citation.quote)
        if len(quote) < 8:
            raise CitationValidationError("Citation quotes must contain at least eight characters.")
        if not isinstance(text, str) or quote not in _normalized_text(text):
            raise CitationValidationError(
                f"Quote for {citation.evidence_id} is not present in the retrieved chunk."
            )
        required = ("chunk_id", "doc_id", "company", "fiscal_year", "pdf_page")
        missing = [key for key in required if payload.get(key) is None]
        if missing:
            raise CitationValidationError(
                f"Retrieved evidence {citation.evidence_id} lacks metadata: {', '.join(missing)}."
            )
        validated.append(ValidatedCitation(
            evidence_id=citation.evidence_id,
            chunk_id=str(payload["chunk_id"]),
            doc_id=str(payload["doc_id"]),
            company=str(payload["company"]),
            fiscal_year=str(payload["fiscal_year"]),
            pdf_page=int(payload["pdf_page"]),
            printed_page=payload.get("printed_page"),
            quote=citation.quote.strip(),
        ))
    return validated
