"""Pydantic models for API request and response payloads.

These are HTTP-layer-only models. Do NOT import domain models (Chunk, Answer,
etc.) from src/ here and re-export them — keep the HTTP schema separate so the
domain model can evolve independently of the wire format.
"""

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str


# ── Ingestion ─────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    """Response body for POST /ingest."""

    document_id: str = Field(description="UUID assigned to the ingested document.")
    filename: str = Field(description="Original filename of the uploaded PDF.")
    chunk_count: int = Field(description="Number of chunks stored in the vector index.")


# ── Query ─────────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    """A single message in a conversation history."""

    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class QueryRequest(BaseModel):
    """Request body for POST /query."""

    query: str = Field(min_length=1, description="Natural-language question.")
    history: list[ChatMessage] | None = Field(
        default=None,
        description="Conversation history for follow-up queries. Last N turns used for query rewriting.",
    )


class CitationDetail(BaseModel):
    """Serialised view of a single retrieved chunk, shown alongside the answer."""

    chunk_index: int = Field(description="0-based position in the retrieved_chunks list.")
    chunk_id: str = Field(description="Internal UUID of the chunk.")
    text: str = Field(description="Raw chunk text.")
    score: float = Field(description="Cosine similarity score (0–1).")
    metadata: dict[str, Any] = Field(description="Provenance: filename, page, etc.")


class QueryResponse(BaseModel):
    """Response body for POST /query."""

    answer: str = Field(description="Generated answer text with inline [chunk_N] citations.")
    citations: list[int] = Field(
        description="0-based indices into retrieved_chunks that were cited."
    )
    retrieved_chunks: list[CitationDetail] = Field(
        description="All chunks passed to the generator, for UI source display."
    )
    rewritten_query: str | None = Field(
        default=None,
        description="Query sent to the retriever. None when no rewriting occurred (first turn or rewriter skipped).",
    )
