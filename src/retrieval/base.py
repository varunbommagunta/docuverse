"""Protocol interfaces for the retrieval layer.

The retrieval layer handles three distinct concerns:

  Embedder    — converts text into a dense float vector
  VectorStore — persists and queries vectors by similarity
  Retriever   — end-to-end: embeds a query and returns the top-K chunks

Separating Embedder from VectorStore allows us to use the same embedding model
with different databases (Chroma locally, pgvector in production) and vice versa.
The high-level Retriever protocol exists for callers (Orchestrator) that don't
need to know about the two-step process.
"""

from typing import Protocol

from src.utils.models import Chunk, RetrievedChunk


class Embedder(Protocol):
    """Converts text strings into dense float vectors (embeddings).

    Implementations should be stateless — the same text must always produce
    the same vector for a given model. Batching is exposed so callers can
    amortise API round-trips.

    V1 implementation: OpenAIEmbedder — text-embedding-3-small via OpenAI API.
        Fast, high-quality, ~1536 dimensions, charged per token.
    V2 implementation: LocalEmbedder — sentence-transformers model running on CPU/GPU.
        Zero marginal cost, ~768 dimensions, requires GPU for throughput.
    V3 implementation: CohereEmbedder — Cohere Embed v3, supports multilingual docs.
    """

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into float vectors.

        Args:
            texts: Non-empty list of strings to embed.

        Returns:
            List of float vectors, same length and order as `texts`.
            All vectors have the same dimensionality (model-dependent).

        Raises:
            RetrievalError: On API failure or dimension mismatch.
        """
        ...


class VectorStore(Protocol):
    """Persists document embeddings and supports similarity search.

    Responsible for all CRUD operations on the vector index. Does NOT know
    about query formulation — the Retriever handles that.

    V1 implementation: ChromaStore — local persistent Chroma DB, zero infra.
    V2 implementation: PgVectorStore — pgvector extension on PostgreSQL;
        enables hybrid keyword + vector search via tsvector.
    V3 implementation: PineconeStore — managed serverless, scales to billions
        of vectors, built-in namespacing for multi-tenancy.
    """

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Persist chunks and their corresponding embeddings.

        Args:
            chunks: Chunk objects whose metadata will be stored alongside vectors.
            embeddings: Float vectors, one per chunk, in matching order.

        Raises:
            RetrievalError: On write failure or dimension mismatch.
        """
        ...

    def similarity_search(
        self, query_embedding: list[float], top_k: int
    ) -> list[RetrievedChunk]:
        """Return the top-K chunks most similar to query_embedding.

        Args:
            query_embedding: Float vector for the user query.
            top_k: Number of results to return.

        Returns:
            List of RetrievedChunk ordered by descending similarity score.
            May return fewer than top_k if the index has fewer entries.

        Raises:
            RetrievalError: On query failure or empty index.
        """
        ...

    def delete_document(self, doc_id: str) -> None:
        """Remove all chunks belonging to the given document.

        Args:
            doc_id: The document identifier used when chunks were added.

        Raises:
            RetrievalError: If the document is not found or deletion fails.
        """
        ...

    def get_all_chunks(self) -> list[Chunk]:
        """Return every chunk currently stored in the vector store.

        Used by BM25Retriever to build its in-memory index on startup.

        Returns:
            All Chunk objects in arbitrary order.

        Raises:
            RetrievalError: On read failure.
        """
        ...


class Retriever(Protocol):
    """High-level interface: embed a query and return top-K RetrievedChunks.

    This is the entry point the Orchestrator calls. It abstracts the
    Embedder + VectorStore two-step from callers that don't need to know
    about the internals.

    V1 implementation: DenseRetriever — wraps OpenAIEmbedder + ChromaStore.
    V2 implementation: HybridRetriever — combines dense vector search with BM25
        keyword search, fused with Reciprocal Rank Fusion.
    V3 implementation: RerankedRetriever — wraps any Retriever with a cross-encoder.
    """

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Embed query and return the most relevant chunks.

        Args:
            query: Natural language user question.
            top_k: Maximum number of chunks to return.

        Returns:
            List of RetrievedChunk ordered by descending relevance score.

        Raises:
            RetrievalError: If embedding or vector search fails.
        """
        ...


class Reranker(Protocol):
    """Re-scores and re-orders a candidate list using a cross-encoder model.

    Takes a query and a set of already-retrieved candidates, then applies a
    more expensive but more accurate relevance model. Sits between retrieval
    and generation; the Orchestrator never calls it directly.

    V1 implementation: CrossEncoderReranker — sentence-transformers cross-encoder,
        ms-marco-MiniLM-L-6-v2, runs on CPU, ~15ms per batch of 50 candidates.
    """

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        """Re-score candidates and return the top_k most relevant.

        Args:
            query: The original natural language question.
            candidates: Pre-retrieved chunks to re-score.
            top_k: Number of chunks to return after reranking.

        Returns:
            Top-k RetrievedChunks ordered by descending reranker score.
            Each chunk's score field is updated to the reranker score.
        """
        ...
