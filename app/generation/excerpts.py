"""Resolve model-selected excerpt IDs to exact source text on the server."""

import json
import re

from app.generation.models import DraftCitation


def source_excerpts(text: str, *, size: int = 650, overlap: int = 120):
    """Yield contiguous, overlapping source spans without editing their text."""
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + size // 2, end)
            if boundary > start:
                end = boundary
        span = text[start:end].strip()
        if len(span) >= 8:
            yield span
        if end == len(text):
            break
        start = max(start + 1, end - overlap)


class ExcerptGenerator:
    """The model chooses source spans; it never needs to transcribe citations.

    Uses the existing provider's structured AnswerDraft contract. In the wire
    response, quote contains an excerpt ID. Only known IDs belonging to the
    cited evidence are resolved. Invalid selections fail normal validation.
    """

    def __init__(self, generator, *, max_excerpts_per_evidence: int = 2):
        if max_excerpts_per_evidence < 1:
            raise ValueError("max_excerpts_per_evidence must be positive")
        self.generator = generator
        self.model = generator.model
        self.max_excerpts_per_evidence = max_excerpts_per_evidence

    def generate(self, system_prompt: str, user_prompt: str):
        payload = json.loads(user_prompt)
        lookup = {}
        evidence = []
        per_evidence_limit = self.max_excerpts_per_evidence
        if payload.get("comparison_scope") and len(payload["evidence"]) > 5:
            per_evidence_limit = 1
        query_terms = {
            token for token in re.findall(r"[a-z0-9]+", payload["question"].casefold())
            if len(token) > 2 and token not in {
                "and", "did", "for", "from", "how", "its", "report", "reported",
                "the", "was", "were", "what", "which",
            }
        }
        for item in payload["evidence"]:
            candidates = []
            for index, text in enumerate(source_excerpts(item["text"]), start=1):
                excerpt_id = f"{item['evidence_id']}:S{index}"
                lowered = text.casefold()
                score = sum(lowered.count(term) for term in query_terms)
                candidates.append((score, index, {"excerpt_id": excerpt_id, "text": text}))
            selected = sorted(candidates, key=lambda row: (-row[0], row[1]))[
                :per_evidence_limit
            ]
            excerpts = [row[2] for row in sorted(selected, key=lambda row: row[1])]
            for excerpt in excerpts:
                lookup[excerpt["excerpt_id"]] = (item["evidence_id"], excerpt["text"])
            evidence.append({key: value for key, value in item.items() if key != "text"} | {"excerpts": excerpts})
        payload["evidence"] = evidence
        # Previous responses contain resolved source quotes, not wire-format IDs.
        payload.pop("previous_response", None)
        instructions = """
CITATION OUTPUT FORMAT: Source text is supplied as numbered excerpts. For each
citation put the selected excerpt_id (for example E1:S2) in the quote field,
and its parent evidence_id (E1) in evidence_id. Do NOT copy or rewrite source
text into quote. The server inserts the exact source text for that ID.
Choose excerpts that directly support every material claim, including the
company, metric, amount, and period. Do not cite unrelated matching numbers.
Use at most one excerpt per evidence_id. The answer itself must remain normal
prose, never an excerpt ID. If the evidence does not support the answer, abstain.
These ID-selection instructions replace instructions to copy literal quotes.
For each comparable_fact, also put its selected excerpt_id in supporting_quote.
Use the same excerpt for that fact's citation.
"""
        draft = self.generator.generate(system_prompt + instructions, json.dumps(payload, ensure_ascii=False))
        citations = []
        for citation in draft.citations:
            selected = lookup.get(citation.quote.strip())
            if selected is not None and selected[0] == citation.evidence_id:
                citations.append(DraftCitation(evidence_id=citation.evidence_id, quote=selected[1]))
            else:
                # Never accept free text or an ID from a different evidence item.
                citations.append(DraftCitation(evidence_id=citation.evidence_id, quote="INVALID"))
        facts = []
        for fact in draft.comparable_facts or []:
            selected = lookup.get(fact.supporting_quote.strip())
            quote = selected[1] if selected and selected[0] == fact.evidence_id else "INVALID"
            facts.append(fact.model_copy(update={"supporting_quote": quote}))
        return draft.model_copy(update={"citations": citations, "comparable_facts": facts or None})
