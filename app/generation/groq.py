"""Small Groq Chat Completions client with strict structured output."""

import json
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import ValidationError

from app.generation.models import AnswerDraft


class GroqGenerationError(RuntimeError):
    """A sanitized generation failure safe to show without leaking secrets."""


def _groq_strict_schema() -> dict:
    """Keep Pydantic's shape while avoiding undocumented schema constraints."""

    unsupported = {
        "default",
        "maxItems",
        "maxLength",
        "maximum",
        "minItems",
        "minLength",
        "minimum",
        "pattern",
    }

    def clean(value):
        if isinstance(value, dict):
            return {
                key: clean(item)
                for key, item in value.items()
                if key not in unsupported
            }
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    return clean(AnswerDraft.model_json_schema())


class GroqGenerator:
    """Call Groq without adding a second SDK around its OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        api_url: str,
        timeout: float = 30.0,
        opener: Callable = urlopen,
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
            "max_completion_tokens": 1200,
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
        try:
            with self.opener(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            content = payload["choices"][0]["message"]["content"]
            return AnswerDraft.model_validate_json(content)
        except HTTPError as error:
            raise GroqGenerationError(f"Groq request failed with HTTP {error.code}.") from error
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
