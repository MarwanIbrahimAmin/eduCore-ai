"""
OpenRouter LLM client for EduCore teacher tools.

Public interface
----------------
    llm_client.generate(system_prompt, user_prompt, temperature, max_tokens) -> str
        Backwards-compatible. Returns the text of a COMPLETE response and raises
        a specific LLMError subclass otherwise (never returns truncated text).

    llm_client.generate_detailed(...) -> LLMResponse
        Same call, but returns the text together with finish reason, token usage,
        attempt count and latency.

Behaviour
---------
* The SDK client is created lazily, so importing this module needs no API key.
* The SDK's own retries are disabled (max_retries=0); this class owns the retry
  policy so that the number of calls is predictable and observable.
* Only retryable failures are retried: timeouts, connection errors, HTTP 408/409,
  429 (when the requested wait is short), 5xx, and an empty response.
  Authentication, bad-request, request-too-large and not-found errors are
  permanent and raised immediately.
* finish_reason == "length" means the output was cut off at the token limit. It
  raises LLMTruncatedError and is never retried unchanged, because the same
  request would be cut off again.
* Error text and logs are redacted: API keys never appear in them.
"""

from __future__ import annotations

import logging
import random
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional

from educore.teacher_ai.core import config

logger = logging.getLogger(__name__)

_SECRET_PATTERNS = (
    re.compile(r"gsk_[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{8,}"),
)


def redact(text: Any, extra_secrets: tuple = (), limit: int = 600) -> str:
    """Return ``text`` with API keys removed and length capped."""
    value = "" if text is None else str(text)
    for secret in extra_secrets:
        if secret and len(secret) >= 6:
            value = value.replace(secret, "[REDACTED]")
    for pattern in _SECRET_PATTERNS:
        value = pattern.sub("[REDACTED]", value)
    return value if len(value) <= limit else value[:limit] + "..."


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class LLMError(Exception):
    """Base class. ``retryable`` says whether the same call may succeed later."""

    retryable = False

    def __init__(self, message: str, *, status_code: Optional[int] = None,
                 detail: Any = None, retry_after: Optional[float] = None,
                 retryable: Optional[bool] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.detail = redact(detail) if detail else None
        self.retry_after = retry_after
        if retryable is not None:
            self.retryable = retryable


class LLMConfigurationError(LLMError):
    """Missing API key, missing SDK, or invalid client settings."""


class LLMAuthenticationError(LLMError):
    """HTTP 401/403: the API key was rejected."""


class LLMRequestError(LLMError):
    """Permanent client-side failure (HTTP 400/404/413/422 and similar)."""


class LLMRateLimitError(LLMError):
    """HTTP 429. Retryable only when the required wait is short."""

    retryable = True


class LLMTimeoutError(LLMError):
    retryable = True


class LLMConnectionError(LLMError):
    retryable = True


class LLMServerError(LLMError):
    """HTTP 5xx (or 408/409) from the provider."""

    retryable = True


class LLMEmptyResponseError(LLMError):
    """The API answered successfully but returned no text."""

    retryable = True


class LLMResponseError(LLMError):
    """The response had an unexpected shape or finish reason."""


class LLMTruncatedError(LLMError):
    """The output hit the token limit (finish_reason == 'length')."""

    def __init__(self, message: str, *, partial_content: str = "",
                 completion_tokens: Optional[int] = None,
                 max_tokens: Optional[int] = None):
        super().__init__(message)
        self.partial_content = partial_content
        self.completion_tokens = completion_tokens
        self.max_tokens = max_tokens


# ---------------------------------------------------------------------------
# Response object
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    content: str
    finish_reason: Optional[str]
    model: Optional[str]
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    attempts: int = 1
    latency_seconds: float = 0.0
    request_id: Optional[str] = None

    def to_metadata(self) -> dict:
        data = asdict(self)
        data.pop("content", None)
        data["finish_reason_missing"] = self.finish_reason is None
        return data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RETRY_IN_PATTERNS = (
    (re.compile(r"try again in\s+(\d+(?:\.\d+)?)\s*h\s*(\d+(?:\.\d+)?)\s*m", re.I), lambda m: float(m[1]) * 3600 + float(m[2]) * 60),
    (re.compile(r"try again in\s+(\d+(?:\.\d+)?)\s*m\s*(\d+(?:\.\d+)?)\s*s", re.I), lambda m: float(m[1]) * 60 + float(m[2])),
    (re.compile(r"try again in\s+(\d+(?:\.\d+)?)\s*ms", re.I), lambda m: float(m[1]) / 1000.0),
    (re.compile(r"try again in\s+(\d+(?:\.\d+)?)\s*s", re.I), lambda m: float(m[1])),
    (re.compile(r"try again in\s+(\d+(?:\.\d+)?)\s*m\b", re.I), lambda m: float(m[1]) * 60),
)


def parse_retry_after(headers: Any, message: str = "") -> Optional[float]:
    """Seconds to wait, from a Retry-After header or the provider's message text."""
    try:
        raw = headers.get("retry-after") if headers is not None else None
    except Exception:  # headers object of an unexpected type
        raw = None
    if raw is not None:
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            pass
    for pattern, convert in _RETRY_IN_PATTERNS:
        match = pattern.search(message or "")
        if match:
            return convert(match)
    return None


def _error_text(exc: Exception) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        inner = body.get("error")
        if isinstance(inner, dict) and inner.get("message"):
            return str(inner["message"])
        if body.get("message"):
            return str(body["message"])
    return str(getattr(exc, "message", None) or exc)


def _int_or_none(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


class LLMClient:
    """Thin, well-behaved wrapper around OpenRouter chat completions."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        sdk_client: Any = None,
        max_retries: Optional[int] = None,
        base_delay: Optional[float] = None,
        max_delay: Optional[float] = None,
        max_retry_after: Optional[float] = None,
        timeout: Optional[float] = None,
        model_max_completion_tokens: Optional[int] = None,
        json_mode: Optional[bool] = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: Callable[[], float] = random.random,
    ):
        self._api_key = api_key
        self.model = model or config.OPENROUTER_MODEL
        self._sdk = sdk_client
        self.max_retries = config.OPENROUTER_MAX_RETRIES if max_retries is None else max_retries
        self.base_delay = config.OPENROUTER_RETRY_BASE_DELAY if base_delay is None else base_delay
        self.max_delay = config.OPENROUTER_RETRY_MAX_DELAY if max_delay is None else max_delay
        self.max_retry_after = config.OPENROUTER_MAX_RETRY_AFTER if max_retry_after is None else max_retry_after
        self.timeout = config.OPENROUTER_TIMEOUT_SECONDS if timeout is None else timeout
        self.model_max_completion_tokens = (
            config.OPENROUTER_MODEL_MAX_COMPLETION_TOKENS
            if model_max_completion_tokens is None else model_max_completion_tokens
        )
        self.json_mode = config.OPENROUTER_JSON_MODE if json_mode is None else json_mode
        self._sleep = sleep
        self._rng = rng
        # OpenRouter recommends max_completion_tokens; retain a compatibility
        # fallback for older SDK versions.
        self._token_param = "max_completion_tokens"

    # -- SDK access ---------------------------------------------------------
    def _secrets(self) -> tuple:
        return (self._api_key or config.OPENROUTER_API_KEY or "",)

    def _get_sdk(self) -> Any:
        if self._sdk is None:
            key = self._api_key or config.require_openrouter_api_key()
            if not key or not key.strip():
                raise LLMConfigurationError(
                    "OPENROUTER_API_KEY is missing. Add it to EduCore's .env file."
                )
            try:
                from educore.provider import create_sync_client
            except ImportError:
                raise LLMConfigurationError(
                    "The 'openai' package is not installed. Install EduCore dependencies."
                ) from None
            self._sdk = create_sync_client(
                api_key=key.strip(), timeout=self.timeout, max_retries=0
            )
        return self._sdk

    # -- Public API ---------------------------------------------------------
    def generate(self, system_prompt: str, user_prompt: str,
                 temperature: float = 0.4, max_tokens: Optional[int] = None,
                 *, model: Optional[str] = None) -> str:
        """Return the text of a complete response (see module docstring)."""
        return self.generate_detailed(
            system_prompt, user_prompt, temperature, max_tokens, model=model
        ).content

    def generate_detailed(self, system_prompt: str, user_prompt: str,
                          temperature: float = 0.4, max_tokens: Optional[int] = None,
                          *, model: Optional[str] = None) -> LLMResponse:
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("system_prompt must be a non-empty string.")
        if not isinstance(user_prompt, str) or not user_prompt.strip():
            raise ValueError("user_prompt must be a non-empty string.")

        requested = config.OPENROUTER_GENERATION_MAX_TOKENS if max_tokens is None else int(max_tokens)
        if requested < 1:
            raise ValueError("max_tokens must be a positive integer.")
        limit = min(requested, self.model_max_completion_tokens)
        if limit < requested:
            logger.warning("max_tokens %s exceeds the model limit; using %s.", requested, limit)
        temp = max(0.0, min(float(temperature), config.MAX_TEMPERATURE))
        model_name = model or self.model

        started = time.monotonic()
        total_attempts = self.max_retries + 1
        for attempt in range(1, total_attempts + 1):
            try:
                raw = self._call(model_name, system_prompt, user_prompt, temp, limit)
                response = self._to_response(raw, model_name, limit)
            except LLMError as exc:
                error = exc
            except Exception as exc:  # SDK / transport exceptions
                error = self._classify(exc)
            else:
                response.attempts = attempt
                response.latency_seconds = round(time.monotonic() - started, 3)
                logger.info(
                    "LLM call ok: model=%s finish=%s prompt_tokens=%s completion_tokens=%s attempts=%s latency=%.1fs",
                    response.model, response.finish_reason, response.prompt_tokens,
                    response.completion_tokens, attempt, response.latency_seconds,
                )
                return response

            if not error.retryable or attempt == total_attempts:
                logger.warning("LLM call failed (%s) after %s attempt(s): %s",
                               type(error).__name__, attempt, redact(error.message, self._secrets()))
                raise error
            delay = self._backoff(attempt, error.retry_after)
            logger.info("LLM call retry %s/%s in %.1fs after %s", attempt, self.max_retries,
                        delay, type(error).__name__)
            self._sleep(delay)
        raise LLMError("LLM call failed.")  # pragma: no cover - loop always returns or raises

    # -- Internals ----------------------------------------------------------
    def _backoff(self, attempt: int, retry_after: Optional[float]) -> float:
        base = min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))
        delay = base + self._rng() * 0.25 * base
        if retry_after is not None:
            delay = max(delay, retry_after)
        return delay

    def _call(self, model: str, system_prompt: str, user_prompt: str,
              temperature: float, max_tokens: int) -> Any:
        sdk = self._get_sdk()

        def build() -> dict:
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                self._token_param: max_tokens,
            }
            if self.json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            return kwargs

        try:
            return sdk.chat.completions.create(**build())
        except TypeError as exc:
            # An old SDK may not know max_completion_tokens; retry once with max_tokens.
            if self._token_param == "max_completion_tokens" and "max_completion_tokens" in str(exc):
                logger.warning("SDK rejected max_completion_tokens; falling back to max_tokens.")
                self._token_param = "max_tokens"
                return sdk.chat.completions.create(**build())
            raise

    def _to_response(self, raw: Any, requested_model: str, max_tokens: int) -> LLMResponse:
        choices = getattr(raw, "choices", None)
        if not choices:
            raise LLMResponseError("The AI response contained no choices.")
        choice = choices[0]
        message = getattr(choice, "message", None)
        content = getattr(message, "content", None)
        finish = getattr(choice, "finish_reason", None)
        usage = getattr(raw, "usage", None)
        completion_tokens = _int_or_none(getattr(usage, "completion_tokens", None))

        response = LLMResponse(
            content=content if isinstance(content, str) else "",
            finish_reason=finish if isinstance(finish, str) else None,
            model=getattr(raw, "model", None) or requested_model,
            prompt_tokens=_int_or_none(getattr(usage, "prompt_tokens", None)),
            completion_tokens=completion_tokens,
            total_tokens=_int_or_none(getattr(usage, "total_tokens", None)),
            max_tokens=max_tokens,
            request_id=getattr(raw, "id", None),
        )

        if response.finish_reason == "length":
            raise LLMTruncatedError(
                "The AI response was cut off at the output token limit "
                f"({completion_tokens if completion_tokens is not None else 'unknown'} of "
                f"{max_tokens} tokens used). The output is incomplete.",
                partial_content=response.content,
                completion_tokens=completion_tokens,
                max_tokens=max_tokens,
            )
        if not response.content.strip():
            raise LLMEmptyResponseError(
                f"The AI returned an empty response (finish_reason={response.finish_reason}).")
        if response.finish_reason not in (None, "stop"):
            raise LLMResponseError(
                f"The AI stopped for an unexpected reason: {response.finish_reason!r}.")
        return response

    def _classify(self, exc: Exception) -> LLMError:
        """Map an SDK/transport exception to an LLMError (without leaking secrets)."""
        name = type(exc).__name__
        status = getattr(exc, "status_code", None)
        status = status if isinstance(status, int) else None
        text = redact(_error_text(exc), self._secrets())
        headers = getattr(getattr(exc, "response", None), "headers", None)

        if "Timeout" in name:
            return LLMTimeoutError("The AI service timed out.", detail=text)
        if "Connection" in name:
            return LLMConnectionError("Could not connect to the AI service.", detail=text)
        if status == 429:
            wait = parse_retry_after(headers, text)
            if wait is not None and wait > self.max_retry_after:
                return LLMRateLimitError(
                    f"Rate limit reached; the provider asks to wait about {int(wait)}s. "
                    "Try again later or check your plan limits.",
                    status_code=status, detail=text, retry_after=wait, retryable=False)
            return LLMRateLimitError("Rate limit reached.", status_code=status,
                                     detail=text, retry_after=wait)
        if status in (401, 403):
            return LLMAuthenticationError(
                "The AI service rejected the API key. Check OPENROUTER_API_KEY.",
                status_code=status, detail=text)
        if status == 413:
            return LLMRequestError(
                "The request was too large for the model or your plan's token limit. "
                "Lower OPENROUTER_GENERATION_MAX_TOKENS or the lesson size.",
                status_code=status, detail=text)
        if status in (400, 404, 422):
            return LLMRequestError(
                f"The AI service rejected the request (HTTP {status}). Check the OpenRouter model ID and settings.",
                status_code=status, detail=text)
        if status is not None and (status >= 500 or status in (408, 409)):
            return LLMServerError(f"The AI service had a temporary problem (HTTP {status}).",
                                  status_code=status, detail=text)
        return LLMError(f"Unexpected error while calling the AI service ({name}).", detail=text)


llm_client = LLMClient()
