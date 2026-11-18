"""LLM generation through OpenRouter's OpenAI-compatible API."""

from __future__ import annotations

import logging
from typing import AsyncGenerator

from educore.config import settings
from educore.provider import create_async_client

logger = logging.getLogger(__name__)


class OpenRouterGenerator:
    """Async wrapper for regular, streaming, and opt-in web-search responses."""

    def __init__(self, model_id: str | None = None, reasoning_effort: str = "medium"):
        self.model_id = model_id or settings.OPENROUTER_CHAT_MODEL
        self.reasoning_effort = reasoning_effort  # retained for existing call sites
        self.client = create_async_client()

    @staticmethod
    def _max_tokens(value: int | None) -> int:
        result = settings.OPENROUTER_MAX_TOKENS if value is None else value
        if result <= 0:
            raise ValueError("max_tokens must be greater than zero")
        return result

    @staticmethod
    def _temperature(value: float | None) -> float:
        result = settings.OPENROUTER_TEMPERATURE if value is None else value
        if not 0 <= result <= 2:
            raise ValueError("temperature must be between 0 and 2")
        return result

    async def generate(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
        response_format: dict | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        request = {
            "model": self.model_id,
            "messages": messages,
            "max_completion_tokens": self._max_tokens(max_tokens),
            "temperature": self._temperature(temperature),
            "stream": False,
        }
        if response_format is not None:
            request["response_format"] = response_format
        response = await self.client.chat.completions.create(**request)
        if not response.choices:
            raise RuntimeError("OpenRouter returned no completion choices")
        content = response.choices[0].message.content or ""
        if not content.strip():
            raise RuntimeError("OpenRouter returned an empty response")
        logger.info(
            "Generated %d chars (model=%s, tokens=%s)",
            len(content), self.model_id, getattr(response.usage, "total_tokens", "?"),
        )
        return content

    async def generate_with_web_search(
        self,
        messages: list[dict],
        max_completion_tokens: int = 2048,
    ) -> str:
        """Search externally only when explicitly enabled; search has separate pricing."""
        if not settings.OPENROUTER_ENABLE_WEB_SEARCH:
            raise RuntimeError(
                "OpenRouter web search is disabled because it is billed separately from free models."
            )
        response = await self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            temperature=0.2,
            tools=[{"type": "openrouter:web_search"}],
        )
        if not response.choices:
            raise RuntimeError("OpenRouter web search returned no completion choices")
        content = response.choices[0].message.content or ""
        if not content.strip():
            raise RuntimeError("OpenRouter web search returned an empty answer")
        return content

    async def generate_stream(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
    ) -> AsyncGenerator[str, None]:
        request = {
            "model": self.model_id,
            "messages": messages,
            "max_completion_tokens": self._max_tokens(max_tokens),
            "temperature": self._temperature(temperature),
            "stream": True,
        }
        stream = await self.client.chat.completions.create(**request)
        yielded = False
        async for chunk in stream:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content
            if content:
                yielded = True
                yield content
        if not yielded:
            logger.warning("OpenRouter stream returned no visible text; retrying as a completion")
            yield await self.generate(
                messages, max_tokens=max_tokens, temperature=temperature,
            )

