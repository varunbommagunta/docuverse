"""Chroma vector store implementation.

ChromaVectorStore is the V1 concrete implementation of the VectorStore Protocol.
It uses chromadb's PersistentClient so the index survives process restarts.
All vectors use cosine similarity space.
"""

import chromadb
import structlog
from chromadb.config import Settings as ChromaSettings

from config.settings import get_settings
from src.utils.exceptions import RetrievalError
from src.utils.models import Chunk, RetrievedChunk

logger = structlog.get_logger(__name__)

_CANNOT_ANSWER = "I cannot answer this from the provided documents."


class ChromaVectorStore:
    """Persists document embeddings in Chroma and supports cosine similarity search.

    Implements the VectorStore Protocol. The index is persisted to disk so it
    survives API restarts. Uses cosine distance; similarity = 1 - distance.
    """

    def __init__(
        self,
        persist_directory: str | None = None,
        collection_name: str = "docuverse",
    ) -> None:
        """Initialise the Chroma persistent client and collection.

        Args:
            persist_directory: Directory for on-disk persistence.
                Defaults to the value in Settings.
            collection_name: Name of the Chroma collection.
        """
        settings = get_settings()
        directory = persist_directory or settings.chroma_persist_directory
        self._client = chromadb.PersistentClient(
            path=directory,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaVectorStore initialised",
            persist_directory=directory,
            collection=collection_name,
            existing_count=self._collection.count(),
        )

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Persist chunks and their embeddings to the Chroma collection.

        Args:
            chunks: Chunk objects whose text and metadata are stored.
            embeddings: Float vectors in matching order to chunks.

        Raises:
            RetrievalError: On Chroma write failure.
        """
        if not chunks:
            return
        try:
            self._collection.add(
                ids=[c.id for c in chunks],
                embeddings=embeddings,  # type: ignore[arg-type]
                documents=[c.text for c in chunks],
                metadatas=[c.metadata for c in chunks],
            )
        except Exception as exc:
            raise RetrievalError(f"Failed to add chunks to Chroma: {exc}") from exc

        logger.info("Chunks stored", count=len(chunks))

    def similarity_search(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[RetrievedChunk]:
        """Return the top-K chunks most similar to query_embedding.

        Args:
            query_embedding: Float vector for the query.
            top_k: Maximum number of results to return.

        Returns:
            List of RetrievedChunk ordered by descending similarity score.
            May return fewer than top_k if the collection has fewer entries.

        Raises:
            RetrievalError: On Chroma query failure.
        """
        actual_k = min(top_k, self._collection.count())
        if actual_k == 0:
            return []

        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],  # type: ignore[arg-type]
                n_results=actual_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise RetrievalError(f"Chroma query failed: {exc}") from exc

        retrieved: list[RetrievedChunk] = []
        ids = results["ids"][0]
        documents = results["documents"][0]  # type: ignore[index]
        metadatas = results["metadatas"][0]  # type: ignore[index]
        distances = results["distances"][0]  # type: ignore[index]

        for chunk_id, text, meta, distance in zip(ids, documents, metadatas, distances, strict=False):
            similarity = max(0.0, 1.0 - distance)
            chunk = Chunk(id=chunk_id, text=text, metadata=meta or {})
            retrieved.append(RetrievedChunk(chunk=chunk, score=round(similarity, 4)))

        return retrieved

    def delete_document(self, document_id: str) -> int:
        """Remove all chunks belonging to document_id.

        Args:
            document_id: The document_id stored in chunk metadata.

        Returns:
            Number of chunks deleted.

        Raises:
            RetrievalError: On deletion failure.
        """
        try:
            existing = self._collection.get(where={"document_id": document_id})
            count = len(existing["ids"])
            if count > 0:
                self._collection.delete(where={"document_id": document_id})
            logger.info("Document deleted", document_id=document_id, chunks_deleted=count)
            return count
        except Exception as exc:
            raise RetrievalError(f"Failed to delete document '{document_id}': {exc}") from exc

    def get_all_chunks(self) -> list[Chunk]:
        """Return every chunk stored in the collection as Chunk objects.

        Used by BM25Retriever to build its in-memory sparse index.

        Returns:
            All Chunk objects in arbitrary order.

        Raises:
            RetrievalError: On Chroma read failure.
        """
        try:
            result = self._collection.get(include=["documents", "metadatas"])
        except Exception as exc:
            raise RetrievalError(f"Failed to fetch all chunks: {exc}") from exc

        chunks: list[Chunk] = []
        ids = result.get("ids", [])
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []

        for chunk_id, text, meta in zip(ids, documents, metadatas, strict=False):
            chunks.append(Chunk(id=chunk_id, text=text or "", metadata=meta or {}))

        logger.info("All chunks fetched", count=len(chunks))
        return chunks

    def get_by_article_id(self, article_id: str) -> list[RetrievedChunk]:
        """Return all chunks whose metadata article_id matches exactly.

        Used by ArticleFilterRetriever to inject exact-match article chunks
        ahead of semantic search results.

        Args:
            article_id: Article identifier string, e.g. "16", "312", "21A".

        Returns:
            List of RetrievedChunk with score=1.0; empty if none found.
        """
        try:
            result = self._collection.get(
                where={"article_id": article_id},
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            logger.warning("article_id_lookup_failed", article_id=article_id, error=str(exc))
            return []

        chunks: list[RetrievedChunk] = []
        for chunk_id, text, meta in zip(
            result.get("ids", []),
            result.get("documents") or [],
            result.get("metadatas") or [],
            strict=False,
        ):
            chunk = Chunk(id=chunk_id, text=text or "", metadata=meta or {})
            chunks.append(RetrievedChunk(chunk=chunk, score=1.0))

        logger.info("article_id_lookup", article_id=article_id, found=len(chunks))
        return chunks

    def count(self) -> int:
        """Return total number of chunks currently stored."""
        return self._collection.count()
