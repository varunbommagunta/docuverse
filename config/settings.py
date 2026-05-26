"""Application settings loaded from environment variables via pydantic-settings.

All runtime configuration lives here. No other module should read os.environ
directly — import get_settings() instead. This enforces the 12-factor app
principle: config comes from the environment, not from code.
"""

from functools import lru_cache

from pydantic import Field, field_validator
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

    @field_validator("openai_api_key", mode="before")
    @classmethod
    def _strip_openai_key(cls, v):
        if isinstance(v, str):
            return v.strip()
        return v

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
    retrieval_strategy: str = Field(
        default="reranked_hybrid",
        description="Retrieval strategy: dense | sparse | hybrid | reranked_hybrid",
    )
    hybrid_dense_top_k: int = Field(default=20, description="Candidate count from dense in hybrid.", gt=0)
    hybrid_sparse_top_k: int = Field(default=20, description="Candidate count from BM25 in hybrid.", gt=0)
    hybrid_rrf_k: int = Field(default=60, description="RRF constant k (from RRF paper).", gt=0)
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="sentence-transformers cross-encoder model for reranking.",
    )
    reranker_fetch_k: int = Field(
        default=50, description="Candidate count to fetch before cross-encoder reranking.", gt=0
    )

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
        default=50,
        description="Maximum PDF upload size in megabytes.",
        gt=0,
    )

    # ── Query decomposition ───────────────────────────────────────────────────
    enable_query_decomposition: bool = Field(
        default=True,
        description="Split multi-aspect queries into sub-queries before retrieval. "
        "Each sub-query retrieves independently; results are merged and deduplicated.",
    )
    query_decomposer_model: str = Field(
        default="gpt-4o-mini",
        description="Model used by the query decomposer.",
    )
    decomposition_sub_top_k: int = Field(
        default=4,
        description="Chunks retrieved per sub-query when decomposition fires. "
        "Total context passed to the generator is capped at decomposition_max_chunks.",
        gt=0,
    )
    decomposition_max_chunks: int = Field(
        default=8,
        description="Maximum total chunks sent to the generator after merging sub-query results.",
        gt=0,
    )

    # ── Query rewriting ───────────────────────────────────────────────────────
    enable_query_rewriting: bool = Field(
        default=True,
        description="Enable LLM-based query rewriting for conversational follow-ups. "
        "Resolves pronouns and references against conversation history before retrieval.",
    )
    query_rewriter_model: str = Field(
        default="gpt-4o-mini",
        description="Model to use for query rewriting. Cheap models are preferred since "
        "the rewriting task is straightforward.",
    )
    query_rewriter_max_history_turns: int = Field(
        default=3,
        description="Maximum number of conversation turns to include in the rewriter prompt. "
        "Higher = more context but more tokens.",
        ge=1,
        le=10,
    )

    # ── Document classification ───────────────────────────────────────────────
    enable_llm_classifier_fallback: bool = Field(
        default=True,
        description="If True, use gpt-4o-mini for documents the rule-based classifier can't "
        "confidently identify. If False, all uncertain docs go to DefaultChunker.",
    )
    classifier_confidence_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Rule-based confidence below this triggers LLM classifier fallback.",
    )

    # ── HuggingFace Spaces deployment ─────────────────────────────────────────
    hf_spaces_mode: bool = Field(
        default=False,
        description="When true, enables stricter defaults suited for HF Spaces.",
    )
    daily_cost_cap_inr: int = Field(
        default=50,
        description="Daily OpenAI spend cap in Indian Rupees. Rejected with 429 once exceeded.",
        gt=0,
    )
    auto_ingest_on_startup: bool = Field(
        default=True,
        description="Auto-download and ingest the sample PDFs (Indian Constitution + ARC) if ChromaDB is empty at container startup.",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings singleton.

    Using lru_cache ensures .env is parsed exactly once per process.
    In tests, call get_settings.cache_clear() before patching env vars.
    """
    return Settings()
