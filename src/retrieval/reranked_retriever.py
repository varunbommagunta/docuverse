"""Decorator that wraps any Retriever with a Reranker (two-stage retrieval).

RerankedRetriever implements the Retriever Protocol. It first fetches a larger
candidate set from a base Retriever, then applies a Reranker to select the
final top-k. This is the classic "retrieve then rerank" pattern.

Because both Retriever and Reranker are Protocols, this class is generic — it
works with DenseRetriever, BM25Retriever, HybridRetriever, or any other
Retriever, and any Reranker implementation.
"""

import structlog

from src.utils.models import RetrievedChunk

logger = structlog.get_logger(__name__)


class RerankedRetriever:
    """Implements the Retriever Protocol via fetch-then-rerank.

    Step 1: Fetch fetch_k candidates from the base retriever.
    Step 2: Pass all candidates to the reranker, which returns top_k.
    """

    def __init__(self, base, reranker, fetch_k: int = 50) -> None:
        """Inject the base retriever and reranker.

        Args:
            base: Any Retriever Protocol implementation.
            reranker: Any Reranker Protocol implementation.
            fetch_k: Number of candidates to fetch from base before reranking.
                     Should be >> top_k so the reranker has a wide candidate pool.
        """
        self._base = base
        self._reranker = reranker
        self._fetch_k = fetch_k
        self.last_debug: dict = {}

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Fetch fetch_k candidates then rerank to top_k.

        Args:
            query: Natural-language question.
            top_k: Final number of chunks to return.

        Returns:
            Top-k RetrievedChunks after reranking, ordered by descending score.
        """
        log = logger.bind(query_length=len(query), top_k=top_k, fetch_k=self._fetch_k)
        log.info("RerankedRetriever started")

        candidates = self._base.retrieve(query, top_k=self._fetch_k)
        log.info("Candidates fetched", candidate_count=len(candidates))

        results = self._reranker.rerank(query, candidates, top_k=top_k)
        self.last_debug = {"candidates_in": len(candidates), "results_out": len(results)}
        log.info("RerankedRetriever complete", results_count=len(results))
        return results
