"""Core domain models shared across all layers of DocuVerse.

These Pydantic models are the lingua franca of the system. Every component —
parsers, chunkers, embedders, retrievers, generators — speaks in terms of
Chunk, RetrievedChunk, and Answer. No layer should invent its own parallel
data classes for these concepts.
"""

from typing import Any

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A discrete unit of text extracted from a source document.

    Produced by the Chunker and consumed by the Embedder. The metadata dict
    carries provenance information (source filename, page number, etc.) that
    will flow through to citations in the final Answer.
    """

    id: str = Field(description="Unique identifier for this chunk, e.g. '{doc_id}_{index}'.")
    text: str = Field(description="Raw text content of the chunk.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Provenance metadata: source, page, char offsets, etc.",
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

    Contains the generated text, a list of citation keys (chunk IDs), and the
    full list of retrieved chunks so the UI can render source attribution.
    """

    text: str = Field(description="The generated answer text.")
    citations: list[str] = Field(
        default_factory=list,
        description="Chunk IDs cited by the answer, in order of appearance.",
    )
    retrieved_chunks: list[RetrievedChunk] = Field(
        default_factory=list,
        description="All chunks passed to the generator (for UI display and debugging).",
    )
