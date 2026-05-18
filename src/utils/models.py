"""Core domain models shared across all layers of DocuVerse.

These Pydantic models are the lingua franca of the system. Every component —
parsers, chunkers, embedders, retrievers, generators — speaks in terms of
Chunk, RetrievedChunk, and Answer. No layer should invent its own parallel
data classes for these concepts.
"""

from typing import Any

from pydantic import BaseModel, Field


class ParsedDocument(BaseModel):
    """The output of parsing a raw document file.

    Produced by the Parser and consumed by the Chunker. Carries both the full
    concatenated text and per-page text so the Chunker can choose its strategy.
    Metadata flows downstream into Chunk.metadata and ultimately into citations.
    """

    text: str = Field(description="Full concatenated text of the document.")
    pages: list[str] = Field(description="Per-page text, one entry per logical page.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Provenance: filename, total_pages, source_path, etc.",
    )


class Chunk(BaseModel):
    """A discrete unit of text extracted from a source document.

    Produced by the Chunker and consumed by the Embedder. The metadata dict
    carries provenance information (source filename, page number, etc.) that
    will flow through to citations in the final Answer.
    """

    id: str = Field(description="Unique identifier for this chunk (UUID4 string).")
    text: str = Field(description="Raw text content of the chunk.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Provenance metadata: source, page, char offsets, document_id, etc.",
    )


class RetrievedChunk(BaseModel):
    """A Chunk returned by similarity search, decorated with its relevance score.

    The score is cosine similarity (or equivalent); higher is more relevant.
    Consumed by the Generator to build the grounded prompt.
    """

    chunk: Chunk
    score: float = Field(description="Similarity score in [0, 1]. Higher = more relevant.")


class Answer(BaseModel):
    """The final output of a RAG query.

    Contains the generated text, a list of citation indices (0-based positions into
    retrieved_chunks), and the full list of retrieved chunks so the UI can render
    source attribution alongside the answer.
    """

    text: str = Field(description="The generated answer text.")
    citations: list[int] = Field(
        default_factory=list,
        description="0-based indices into retrieved_chunks that the answer cites.",
    )
    retrieved_chunks: list[RetrievedChunk] = Field(
        default_factory=list,
        description="All chunks passed to the generator (for UI display and debugging).",
    )
