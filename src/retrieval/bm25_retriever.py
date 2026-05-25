"""BM25 sparse retriever using rank_bm25.

BM25Retriever implements the Retriever Protocol using keyword-based BM25Okapi
scoring. On construction it pulls all chunks from the vector store and builds
an in-memory BM25 index. This is appropriate for local deployments; for large
corpora (>1M chunks), a dedicated Elasticsearch/OpenSearch BM25 implementation
would be needed.

Scores are normalized to [0, 1] by dividing by the maximum score so they are
comparable to dense cosine similarity scores used elsewhere.
"""

import structlog
from rank_bm25 import BM25Okapi

from src.utils.exceptions import RetrievalError
from src.utils.models import Chunk, RetrievedChunk

logger = structlog.get_logger(__name__)


def _tokenize(text: str) -> list[str]:
    """Word-boundary tokenization — strips punctuation so '312.' matches '312'."""
    import re
    return re.findall(r'\b\w+\b', text.lower())


class BM25Retriever:
    """Implements the Retriever Protocol using BM25Okapi keyword matching.

    The index is built once at construction time from all chunks currently in
    the vector store. If the corpus changes, the retriever must be recreated.
    """

    def __init__(self, vector_store) -> None:
        """Build the BM25 index from all chunks in vector_store.

        Args:
            vector_store: Any object implementing the VectorStore Protocol.
                          Must have a get_all_chunks() method.
        """
        chunks: list[Chunk] = vector_store.get_all_chunks()
        self._chunks = chunks

        if not chunks:
            logger.warning("BM25Retriever built with empty corpus")
            self._index = None
        else:
            tokenized = [_tokenize(c.text) for c in chunks]
            self._index = BM25Okapi(tokenized)

        logger.info("BM25Retriever initialised", corpus_size=len(chunks))

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Score all chunks with BM25 and return the top-k.

        Args:
            query: Natural-language question.
            top_k: Maximum number of chunks to return.

        Returns:
            List of RetrievedChunk ordered by descending BM25 score, normalized
            to [0, 1] by dividing by the maximum score.

        Raises:
            RetrievalError: If the index is empty or scoring fails.
        """
        log = logger.bind(query_length=len(query), top_k=top_k)
        log.info("BM25 retrieval started")

        if self._index is None or not self._chunks:
            log.warning("BM25 index is empty — returning no results")
            return []

        try:
            query_tokens = _tokenize(query)
            scores = self._index.get_scores(query_tokens)
        except Exception as exc:
            raise RetrievalError(f"BM25 scoring failed: {exc}") from exc

        max_score = float(max(scores)) if scores.any() else 0.0

        # Pair scores with chunks, sort descending, take top_k
        scored = sorted(
            zip(scores, self._chunks, strict=False),
            key=lambda x: x[0],
            reverse=True,
        )[:top_k]

        results: list[RetrievedChunk] = []
        for raw_score, chunk in scored:
            normalized = float(raw_score) / max_score if max_score > 0 else 0.0
            results.append(RetrievedChunk(chunk=chunk, score=round(normalized, 4)))

        returned_scores = [round(r.score, 4) for r in results]
        log.info("BM25 retrieval complete", results_count=len(results), scores=returned_scores)
        return results
