"""Shared OpenRouter client configuration for student and teacher features."""

from __future__ import annotations

from openai import AsyncOpenAI, OpenAI

from educore.config import settings


def require_openrouter_api_key() -> str:
    key = settings.OPENROUTER_API_KEY.strip()
    if not key:
        raise ValueError("OPENROUTER_API_KEY is not configured in EduCore's .env file")
    return key


def create_async_client(
    *, api_key: str | None = None, timeout: float | None = None, max_retries: int = 0
) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=api_key or require_openrouter_api_key(),
        base_url=settings.OPENROUTER_BASE_URL,
        timeout=timeout,
        max_retries=max_retries,
    )


def create_sync_client(
    *, api_key: str | None = None, timeout: float | None = None, max_retries: int = 0
) -> OpenAI:
    return OpenAI(
        api_key=api_key or require_openrouter_api_key(),
        base_url=settings.OPENROUTER_BASE_URL,
        timeout=timeout,
        max_retries=max_retries,
    )
