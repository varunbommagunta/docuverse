"""Pydantic models for API request and response payloads."""

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str


# ── Ingestion ─────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    document_type: str = Field(default="default")
    classification_confidence: float = Field(default=1.0)
    classification_method: str = Field(default="none")
    chunker_used: str = Field(default="default")


# ── Query ─────────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    history: list[ChatMessage] | None = Field(default=None)
    session_id: str | None = Field(default=None)


class CitationDetail(BaseModel):
    chunk_index: int
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any]


# ── Debug models ──────────────────────────────────────────────────────────────

class ArticleFilterDebug(BaseModel):
    matched: bool
    article_id: str | None = None
    pinned_count: int = 0


class RerankerDebug(BaseModel):
    candidates_in: int
    results_out: int


class ChunkDebug(BaseModel):
    id: str
    score: float
    pinned: bool = False
    source: str = ""
    article_id: str | None = None
    section_title: str | None = None
    preview: str = ""
    text: str = ""


class LatencyDebug(BaseModel):
    rewrite_ms: float = 0.0
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0


class QueryDebug(BaseModel):
    original_query: str
    rewritten_query: str
    article_filter: ArticleFilterDebug
    retrieval_strategy: str
    chunks: list[ChunkDebug]
    reranker: RerankerDebug | None = None
    latency: LatencyDebug | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[int]
    retrieved_chunks: list[CitationDetail]
    rewritten_query: str | None = Field(default=None)
    debug: QueryDebug | None = Field(default=None)
