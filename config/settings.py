"""Application settings loaded from environment variables via pydantic-settings.

All runtime configuration lives here. No other module should read os.environ
directly — import get_settings() instead. This enforces the 12-factor app
principle: config comes from the environment, not from code.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for DocuVerse.

    Values are read from environment variables (case-insensitive) and .env files.
    Defaults are safe for local development; override in production via env vars.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str = Field(default="", description="OpenAI secret key.")
    openai_model: str = Field(default="gpt-4o-mini", description="Chat completion model.")

    # ── Embedding ─────────────────────────────────────────────────────────────
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model name.",
    )

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = Field(default=500, description="Target token count per chunk.", gt=0)
    chunk_overlap: int = Field(
        default=50, description="Overlap between consecutive chunks.", ge=0
    )

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k: int = Field(default=5, description="Number of chunks returned by retriever.", gt=0)

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO", description="Logging level (DEBUG/INFO/WARNING/ERROR).")

    # ── API server ────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0", description="Uvicorn bind host.")
    api_port: int = Field(default=8000, description="Uvicorn bind port.", gt=0)

    # ── Vector store ──────────────────────────────────────────────────────────
    chroma_persist_directory: str = Field(
        default="./data/chroma_db",
        description="Directory where Chroma persists its index.",
    )

    # ── Upload limits ─────────────────────────────────────────────────────────
    max_upload_size_mb: int = Field(
        default=25,
        description="Maximum PDF upload size in megabytes.",
        gt=0,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings singleton.

    Using lru_cache ensures .env is parsed exactly once per process.
    In tests, call get_settings.cache_clear() before patching env vars.
    """
    return Settings()
