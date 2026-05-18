"""Dense retriever — embeds a query and searches the vector store.

DenseRetriever is the V1 concrete implementation of the Retriever Protocol.
It performs pure cosine-similarity retrieval with no re-ranking. The embedder
and vector store are injected so they can be swapped independently.
"""

import structlog

from src.retrieval.embedders import OpenAIEmbedder
from src.retrieval.vector_store import ChromaVectorStore
from src.utils.exceptions import RetrievalError
from src.utils.models import RetrievedChunk

logger = structlog.get_logger(__name__)


class DenseRetriever:
    """Implements the Retriever Protocol using dense vector similarity search.

    Composes an Embedder (to encode the query) with a VectorStore (to search
    the index). No re-ranking is applied in V1; Phase 2 will add hybrid BM25 +
    cross-encoder re-ranking.
    """

    def __init__(self, embedder: OpenAIEmbedder, vector_store: ChromaVectorStore) -> None:
        """Inject the embedder and vector store.

        Args:
            embedder: Used to convert the query string to a float vector.
            vector_store: Used to run the similarity search.
        """
        self._embedder = embedder
        self._vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Embed query and return the most relevant chunks.

        Args:
            query: Natural-language question from the user.
            top_k: Maximum number of chunks to return.

        Returns:
            List of RetrievedChunk ordered by descending similarity score.

        Raises:
            RetrievalError: If embedding or vector store search fails.
        """
        log = logger.bind(query_length=len(query), top_k=top_k)
        log.info("Retrieval started")

        try:
            query_embedding = self._embedder.embed_single(query)
            results = self._vector_store.similarity_search(query_embedding, top_k=top_k)
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(f"Retrieval failed: {exc}") from exc

        scores = [round(r.score, 4) for r in results]
        log.info("Retrieval complete", results_count=len(results), scores=scores)
        return results
