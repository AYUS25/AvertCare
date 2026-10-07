"""
AvertCare Backend · Application Configuration
Centralised settings loaded from environment variables / .env file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── App ─────────────────────────────────────────────────
    APP_NAME: str = "AvertCare API"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False

    # ── CORS ────────────────────────────────────────────────
    ALLOWED_ORIGINS: list[str] = [
        "http://localhost:3000",
        "https://avertcare.vercel.app",
    ]

    # ── Services ────────────────────────────────────────────
    QDRANT_URL: str = "http://localhost:6333"
    REDIS_URL: str = "redis://localhost:6379"

    # ── Auth ────────────────────────────────────────────────
    FIREBASE_PROJECT_ID: str = ""

    # ── ML ──────────────────────────────────────────────────
    MODEL_PATH: str = "/app/models"
    RAG_COLLECTION: str = "clinical_cases"
    RAG_TOP_K: int = 3

    # ── LLM ─────────────────────────────────────────────────
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.0-flash"

    # ── Feature flags ────────────────────────────────────────
    # Flip to True once Qdrant is seeded. Falls back to mock if False.
    LIVE_RAG_ENABLED: bool = False
    # Flip to True once GEMINI_API_KEY is set.
    LIVE_LLM_ENABLED: bool = False


settings = Settings()
