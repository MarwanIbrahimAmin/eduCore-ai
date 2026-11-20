
"""
Centralised application settings.

Every setting is loaded from environment variables or .env.
Pydantic validates configuration at startup.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    DATABASE_URL: str = (
        "postgresql+asyncpg://educore:educore_pass@localhost:5432/educore"
    )
    DATABASE_URL_SYNC: str = (
        "postgresql://educore:educore_pass@localhost:5432/educore"
    )

    # OpenRouter (OpenAI-compatible API)
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_CHAT_MODEL: str = "baidu/ernie-5.5-flash-preview:free"
    OPENROUTER_MAX_TOKENS: int = Field(default=60000, gt=0)
    OPENROUTER_TEMPERATURE: float = Field(default=0.3, ge=0, le=2)
    # Legacy OpenRouter search flag; text web search is controlled by Groq's
    # explicit opt-in flag below.
    OPENROUTER_ENABLE_WEB_SEARCH: bool = False

    # Groq provider for LLM generation and built-in browser search.
    LLM_PROVIDER: str = "groq"
    GROQ_API_KEY: str = ""
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    GROQ_CHAT_MODEL: str = "openai/gpt-oss-120b"
    GROQ_FAST_MODEL: str = "openai/gpt-oss-20b"
    GROQ_WEB_SEARCH_ENABLED: bool = False

    # Gemini Vision through Google AI Studio.
    GOOGLE_API_KEY: str = ""
    GOOGLE_VISION_MODEL: str = "gemini-2.5-flash"
    VISION_MAX_UPLOAD_BYTES: int = Field(default=12 * 1024 * 1024, gt=0)
    VISION_MAX_COMPRESSED_BYTES: int = Field(default=8 * 1024 * 1024, gt=0)

    # Embeddings
    EMBEDDING_MODEL_NAME: str = "paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_DIMENSIONS: int = Field(default=384, gt=0)

    # RAG Pipeline
    CHUNK_SIZE: int = Field(default=500, gt=0)
    CHUNK_OVERLAP: int = Field(default=100, ge=0)
    MIN_CHUNK_CHARS: int = Field(default=50, ge=0)

    # OCR for scanned or image-only PDFs (Tesseract language packs are installed separately).
    OCR_LANGUAGES: str = "eng+ara"
    OCR_DPI: int = Field(default=250, gt=0)
    OCR_MAX_PAGES: int = Field(default=20, ge=0)
    TESSERACT_CMD: str = ""
    TESSDATA_DIR: str = ""

    RETRIEVAL_TOP_K: int = Field(default=5, ge=3, le=5)
    RETRIEVAL_MIN_SCORE: float = Field(default=0.3, ge=0, le=1)

    # Celery
    CELERY_BROKER_URL: str = "amqp://guest:guest@localhost:5672//"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"

    # Chat
    CHAT_HISTORY_LIMIT: int = Field(default=10, ge=0)

    # File Storage
    STORAGE_DIR: str = "./storage"

    # FastAPI
    API_HOST: str = "0.0.0.0"
    API_PORT: int = Field(default=8000, gt=0, le=65535)

    @property
    def storage_path(self) -> Path:
        path = Path(self.STORAGE_DIR)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def active_chat_model(self) -> str:
        return self.GROQ_CHAT_MODEL

    @property
    def active_web_search_enabled(self) -> bool:
        return self.GROQ_WEB_SEARCH_ENABLED


settings = Settings()
