
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
    OPENROUTER_VISION_MODEL: str = "nvidia/nemotron-3.5-super-vl:free"
    OPENROUTER_MAX_TOKENS: int = Field(default=60000, gt=0)
    OPENROUTER_TEMPERATURE: float = Field(default=0.3, ge=0, le=2)
    # OpenRouter's web search server tool is billed separately from free model
    # inference, so it is opt-in to avoid unexpected charges.
    OPENROUTER_ENABLE_WEB_SEARCH: bool = False

    # Embeddings
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSIONS: int = Field(default=384, gt=0)

    # RAG Pipeline
    CHUNK_SIZE: int = Field(default=500, gt=0)
    CHUNK_OVERLAP: int = Field(default=100, ge=0)
    MIN_CHUNK_CHARS: int = Field(default=50, ge=0)

    # OCR for scanned or image-only PDFs (Tesseract language packs are installed separately).
    OCR_LANGUAGES: str = "eng+ara"
    OCR_DPI: int = Field(default=250, gt=0)
    TESSERACT_CMD: str = ""
    TESSDATA_DIR: str = ""

    RETRIEVAL_TOP_K: int = Field(default=5, gt=0)
    RETRIEVAL_MIN_SCORE: float = Field(default=0.15, ge=0, le=1)

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


settings = Settings()
