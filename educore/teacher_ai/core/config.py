"""Teacher AI settings wired to EduCore's existing provider configuration."""

from educore.config import settings


class ConfigurationError(ValueError):
    """A provider setting required by a teacher feature is unavailable."""


OPENROUTER_MODEL = settings.OPENROUTER_CHAT_MODEL
OPENROUTER_REVIEW_MODEL = settings.OPENROUTER_CHAT_MODEL
OPENROUTER_API_KEY = settings.OPENROUTER_API_KEY
OPENROUTER_MODEL_MAX_COMPLETION_TOKENS = 65536
OPENROUTER_GENERATION_MAX_TOKENS = 30000
OPENROUTER_REVIEW_MAX_TOKENS = 15000
OPENROUTER_REPAIR_MAX_TOKENS = 15000
OPENROUTER_TIMEOUT_SECONDS = 120.0
OPENROUTER_MAX_RETRIES = 1
OPENROUTER_RETRY_BASE_DELAY = 1.0
OPENROUTER_RETRY_MAX_DELAY = 20.0
# Keep 429 retries short so a daily free-tier quota is not needlessly retried.
OPENROUTER_MAX_RETRY_AFTER = 30.0
OPENROUTER_JSON_MODE = True
MAX_TEMPERATURE = settings.OPENROUTER_TEMPERATURE
ENABLE_LLM_REVIEW = False
REVIEW_MAX_INPUT_CHARS = 42000


def require_openrouter_api_key() -> str:
    """Get EduCore's configured key without displaying it in an error."""
    if not settings.OPENROUTER_API_KEY or not settings.OPENROUTER_API_KEY.strip():
        raise ConfigurationError("OPENROUTER_API_KEY is not configured in EduCore.")
    return settings.OPENROUTER_API_KEY.strip()
