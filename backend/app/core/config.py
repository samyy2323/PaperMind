from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- infrastructure ---
    DATABASE_URL: str = "postgresql+psycopg2://papermind:papermind@localhost:5432/papermind"
    REDIS_URL: str = "redis://localhost:6379/0"
    STORAGE_ROOT: Path = Path("./storage")
    CORS_ORIGINS: str = "http://localhost:5173"

    # --- auth ---
    SECRET_KEY: str = "change-me-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 12

    # --- pipeline execution ---
    # celery: real Celery worker via Redis (production / docker-compose)
    # thread: run the same task chain in a background thread inside the API process (local dev, no Redis)
    # inline: run synchronously (tests)
    PIPELINE_MODE: str = "celery"

    # --- OCR ---
    OCR_BACKEND: str = "trocr"  # trocr
    LINE_DETECTOR: str = "doctr"  # doctr | projection
    TROCR_MODEL: str = "microsoft/trocr-base-handwritten"
    DOCTR_DET_ARCH: str = "db_mobilenet_v3_large"
    OCR_BATCH_SIZE: int = 8

    # --- text correction ---
    # Path or HF id of a fine-tuned ByT5/BART seq2seq checkpoint. Empty => pass-through (see correction/).
    CORRECTION_MODEL_PATH: str = ""
    ENABLE_CORRECTION: bool = True

    # --- RAG ---
    EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    RAG_TOP_K: int = 3
    CHUNK_TARGET_CHARS: int = 900

    # --- LLM (open-weight, self-hosted) ---
    LLM_BACKEND: str = "ollama"  # ollama | transformers
    OLLAMA_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "llama3.1:8b"  # Ollama tag, or HF id for the transformers backend
    LLM_TIMEOUT_S: float = 300.0
    LLM_MAX_NEW_TOKENS: int = 400

    # --- flagging thresholds ---
    CONFIDENCE_THRESHOLD: float = 0.6
    HIGH_SCORE_RATIO: float = 0.8
    LOW_SIMILARITY_THRESHOLD: float = 0.30
    # Confidence assigned when the LLM does not report one.
    DEFAULT_LLM_CONFIDENCE: float = 0.75

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
