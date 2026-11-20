"""Shared parsing helpers for model-produced JSON objects."""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)


class StructuredOutputError(ValueError):
    """Raised when a model response is empty, malformed, or not a JSON object."""


def parse_json_object(raw: Any) -> dict[str, Any]:
    """Parse one complete JSON object, allowing a surrounding code fence or prose."""
    if isinstance(raw, BaseModel):
        raw = raw.model_dump(mode="python")
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        raise StructuredOutputError("The model returned an empty structured response.")

    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, count=1, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text, count=1).strip()
    decoder = json.JSONDecoder()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
        start = text.find("{")
        if start >= 0:
            try:
                parsed, _ = decoder.raw_decode(text, start)
            except json.JSONDecodeError as exc:
                reason = "truncated" if exc.pos >= len(text) - 1 else "invalid"
                raise StructuredOutputError(
                    f"The model returned {reason} JSON; a complete object is required."
                ) from exc
        else:
            raise StructuredOutputError("The model did not return a JSON object.")

    if not isinstance(parsed, dict):
        raise StructuredOutputError("The model response must be a JSON object.")
    return parsed


def validate_json_object(raw: Any, model: type[ModelT]) -> ModelT:
    """Parse a model response and validate its fields using a Pydantic model."""
    return model.model_validate(parse_json_object(raw))
