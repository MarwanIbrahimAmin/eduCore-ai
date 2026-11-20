"""Teacher AI settings wired to EduCore's existing provider configuration."""

from educore.config import settings


class ConfigurationError(ValueError):
    """A provider setting required by a teacher feature is unavailable."""


OPENROUTER_MODEL = settings.active_chat_model
OPENROUTER_REVIEW_MODEL = settings.active_chat_model
OPENROUTER_API_KEY = settings.GROQ_API_KEY if settings.LLM_PROVIDER.strip().lower() == "groq" else settings.OPENROUTER_API_KEY
OPENROUTER_MODEL_MAX_COMPLETION_TOKENS = 4096
OPENROUTER_GENERATION_MAX_TOKENS = 4096
OPENROUTER_REVIEW_MAX_TOKENS = 4096
OPENROUTER_REPAIR_MAX_TOKENS = 4096
OPENROUTER_TIMEOUT_SECONDS = 120.0
OPENROUTER_MAX_RETRIES = 2
OPENROUTER_RETRY_BASE_DELAY = 1.0
OPENROUTER_RETRY_MAX_DELAY = 10.0
OPENROUTER_MAX_RETRY_AFTER = 30.0
OPENROUTER_JSON_MODE = True
MAX_TEMPERATURE = settings.OPENROUTER_TEMPERATURE
ENABLE_LLM_REVIEW = False
REVIEW_MAX_INPUT_CHARS = 42000

# Local Ollama fallback settings
def require_openrouter_api_key() -> str:
    """Backward-compatible name for the active provider's key."""
    key = settings.GROQ_API_KEY if settings.LLM_PROVIDER.strip().lower() == "groq" else settings.OPENROUTER_API_KEY
    if not key or not key.strip():
        raise ConfigurationError(f"{settings.LLM_PROVIDER.upper()}_API_KEY is not configured in EduCore.")
    return key.strip()
