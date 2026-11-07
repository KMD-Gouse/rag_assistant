"""Small Groq Chat Completions client with strict structured output."""

import json
import time
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import ValidationError

from app.generation.models import AnswerDraft


class GroqGenerationError(RuntimeError):
    """A sanitized generation failure safe to show without leaking secrets."""

    def __init__(
        self, message: str, *, status_code: int | None = None,
        provider_detail: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.provider_detail = provider_detail


def _groq_strict_schema() -> dict:
    """Hand-craft a Groq-compatible strict JSON schema for AnswerDraft.

    Groq's strict mode requires:
    1. Every property listed in ``required`` — including nullable ones.
    2. ``additionalProperties: false`` at every object level.
    3. No unsupported keywords (``default``, ``pattern``, min/max variants, etc.).

    We build the schema manually rather than transforming Pydantic's output so
    that nullable fields like ``comparable_facts`` stay in ``required`` with an
    explicit ``["array", "null"]`` type union, which Groq accepts.
    """
    comparable_fact = {
        "type": "object",
        "properties": {
            "company":           {"type": "string"},
            "fiscal_year":       {"type": "string"},
            "metric":            {"type": "string"},
            "definition":        {"type": "string"},
            "value_text":        {"type": "string"},
            "unit":              {"type": "string"},
            "evidence_id":       {"type": "string"},
            "supporting_quote":  {"type": "string"},
        },
        "required": [
            "company", "fiscal_year", "metric", "definition",
            "value_text", "unit", "evidence_id", "supporting_quote",
        ],
        "additionalProperties": False,
    }

    draft_citation = {
        "type": "object",
        "properties": {
            "evidence_id": {"type": "string"},
            "quote":       {"type": "string"},
        },
        "required": ["evidence_id", "quote"],
        "additionalProperties": False,
    }

    return {
        "type": "object",
        "$defs": {
            "DraftCitation": draft_citation,
            "ComparableFact": comparable_fact,
        },
        "properties": {
            "answerable":          {"type": "boolean"},
            "evidence_sufficiency": {"type": "number"},
            "answer":              {"type": "string"},
            "citations": {
                "type": "array",
                "items": {"$ref": "#/$defs/DraftCitation"},
            },
            # Required but nullable: null for single-side answers,
            # array of exactly 2 ComparableFact objects for comparisons.
            "comparable_facts": {
                "type": ["array", "null"],
                "items": {"$ref": "#/$defs/ComparableFact"},
            },
        },
        "required": [
            "answerable",
            "evidence_sufficiency",
            "answer",
            "citations",
            "comparable_facts",   # always required; value may be null
        ],
        "additionalProperties": False,
    }


class GroqGenerator:
    """Call Groq without adding a second SDK around its OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        api_url: str,
        timeout: float = 30.0,
        opener: Callable = urlopen,
        sleeper: Callable[[float], None] = time.sleep,
        rate_limit_retries: int = 1,
    ):
        if not api_key.strip():
            raise ValueError("GROQ_API_KEY is required for live answer generation.")
        if timeout <= 0:
            raise ValueError("Groq timeout must be positive.")
        self.api_key = api_key
        self.model = model
        self.api_url = api_url
        self.timeout = timeout
        self.opener = opener
        self.sleeper = sleeper
        self.rate_limit_retries = rate_limit_retries

    def generate(self, system_prompt: str, user_prompt: str) -> AnswerDraft:
        schema = _groq_strict_schema()
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "annual_report_answer",
                    "strict": True,
                    "schema": schema,
                },
            },
            "reasoning_effort": "low",
            "temperature": 0,
            "max_completion_tokens": 800,
        }
        request = Request(
            self.api_url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "annual-report-qa/0.1",
            },
            method="POST",
        )
        for attempt in range(self.rate_limit_retries + 1):
            try:
                with self.opener(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                content = payload["choices"][0]["message"]["content"]
                return AnswerDraft.model_validate_json(content)
            except HTTPError as error:
                if error.code in {500, 502, 503, 504} and attempt < self.rate_limit_retries:
                    retry_after = error.headers.get("Retry-After")
                    try:
                        delay = float(retry_after) if retry_after else 2 ** attempt
                    except ValueError:
                        delay = 2 ** attempt
                    self.sleeper(max(1.0, min(delay, 5.0)))
                    continue
                provider_detail = None
                try:
                    error_payload = json.loads(error.read().decode("utf-8"))
                    candidate = error_payload.get("error", {}).get("message")
                    if isinstance(candidate, str):
                        provider_detail = candidate.replace(self.api_key, "[redacted]")[:500]
                except (UnicodeDecodeError, json.JSONDecodeError, AttributeError, TypeError):
                    pass
                raise GroqGenerationError(
                    f"Groq request failed with HTTP {error.code}.",
                    status_code=error.code, provider_detail=provider_detail,
                ) from error
            except URLError as error:
                raise GroqGenerationError("Could not reach the Groq API.") from error
            except (
                KeyError,
                IndexError,
                TypeError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                ValidationError,
            ) as error:
                raise GroqGenerationError("Groq returned an invalid structured response.") from error

        raise AssertionError("Unreachable Groq retry state.")
