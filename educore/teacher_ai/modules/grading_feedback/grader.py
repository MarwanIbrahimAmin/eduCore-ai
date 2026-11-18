
import json
import re
from typing import Any

from educore.teacher_ai.core.llm_client import llm_client

from .schemas import GradingRequest, GradingResult
from .prompts import (
    GRADING_SYSTEM_PROMPT,
    build_grading_prompt,
)
from .validator import validate_grading_result


def _parse_json_response(response: Any) -> dict:
    """Parse a JSON object from the LLM response."""

    if isinstance(response, dict):
        return response

    if not isinstance(response, str):
        raise ValueError("AI response must be a string or dictionary.")

    cleaned = response.strip()

    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)

        if not match:
            raise ValueError("AI did not return a valid JSON object.")

        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Could not parse AI grading response: {exc}"
            ) from exc

    if not isinstance(parsed, dict):
        raise ValueError("AI response must contain a JSON object.")

    return parsed


def grade_submission(
    request: GradingRequest,
    max_tokens: int = 12000,
) -> GradingResult:
    """Grade and validate a student submission."""

    if not isinstance(request, GradingRequest):
        request = GradingRequest.model_validate(request)

    request_data = request.model_dump(mode="json")
    user_prompt = build_grading_prompt(request_data)

    response = llm_client.generate(
        system_prompt=GRADING_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.1,
        max_tokens=max_tokens,
    )

    parsed_response = _parse_json_response(response)
    result = GradingResult.model_validate(parsed_response)

    errors = validate_grading_result(request, result)

    if errors:
        raise ValueError(
            "AI grading validation failed:\n- "
            + "\n- ".join(errors)
        )

    return result
