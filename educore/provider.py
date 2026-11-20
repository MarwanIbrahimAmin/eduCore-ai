"""Shared OpenAI-compatible client configuration for app LLM providers."""

from __future__ import annotations

from openai import AsyncOpenAI, OpenAI

from educore.config import settings


def provider_settings() -> tuple[str, str, str]:
    provider = settings.LLM_PROVIDER.strip().lower()
    if provider == "groq":
        return provider, settings.GROQ_API_KEY.strip(), settings.GROQ_BASE_URL
    if provider != "groq":
        raise ValueError("Text generation is configured for Groq only. Set LLM_PROVIDER=groq.")
    return provider, settings.GROQ_API_KEY.strip(), settings.GROQ_BASE_URL


def require_api_key() -> str:
    provider, key, _ = provider_settings()
    if not key:
        raise ValueError("GROQ_API_KEY is missing or empty in EduCore's .env file.")
    return key


def require_openrouter_api_key() -> str:
    """Backward-compatible alias for older call sites."""
    return require_api_key()


def create_async_client(
    *, api_key: str | None = None, timeout: float | None = None, max_retries: int = 0
) -> AsyncOpenAI:
    _, configured_key, base_url = provider_settings()
    return AsyncOpenAI(
        api_key=api_key or configured_key or require_api_key(),
        base_url=base_url,
        # Bound provider stalls so a RAG request fails quickly enough for the
        # caller to use its deterministic rewrite fallback or show an error.
        timeout=timeout if timeout is not None else 45.0,
        max_retries=max_retries,
    )


def create_sync_client(
    *, api_key: str | None = None, timeout: float | None = None, max_retries: int = 0
) -> OpenAI:
    _, configured_key, base_url = provider_settings()
    return OpenAI(
        api_key=api_key or configured_key or require_api_key(),
        base_url=base_url,
        timeout=timeout if timeout is not None else 45.0,
        max_retries=max_retries,
    )
