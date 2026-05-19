"""Hybrid retriever combining dense and sparse results with Reciprocal Rank Fusion.

HybridRetriever implements the Retriever Protocol. It retrieves from two
independent retrievers (dense vector search + BM25 keyword search), then fuses
their ranked lists using Reciprocal Rank Fusion (RRF) from Cormack et al. 2009.

RRF formula for each unique chunk:
    rrf_score = sum(1 / (k + rank_i))   for each retriever i
where k=60 (empirically optimal from the paper) and rank starts at 1.
Chunks absent from a retriever's list use a large sentinel rank (10000).
"""

import structlog

from src.utils.models import RetrievedChunk

logger = structlog.get_logger(__name__)

_ABSENT_RANK = 10_000


class HybridRetriever:
    """Implements the Retriever Protocol via RRF fusion of dense + sparse results.

    Both injected retrievers implement the Retriever Protocol — this class is
    generic and does not depend on DenseRetriever or BM25Retriever specifically.
    """

    def __init__(
        self,
        dense,
        sparse,
        rrf_k: int = 60,
        fetch_k: int = 20,
    ) -> None:
        """Inject the dense and sparse retrievers.

        Args:
            dense: Retriever Protocol implementation for dense vector search.
            sparse: Retriever Protocol implementation for BM25 keyword search.
            rrf_k: RRF constant k. Higher values reduce the influence of rank;
                   60 is the value recommended in the original RRF paper.
            fetch_k: How many candidates to fetch from each retriever before fusion.
        """
        self._dense = dense
        self._sparse = sparse
        self._rrf_k = rrf_k
        self._fetch_k = fetch_k

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Retrieve from both retrievers and return top-k after RRF fusion.

        Args:
            query: Natural-language question.
            top_k: Maximum number of chunks to return.

        Returns:
            List of RetrievedChunk ordered by descending RRF score.
        """
        log = logger.bind(query_length=len(query), top_k=top_k)

        dense_results = self._dense.retrieve(query, top_k=self._fetch_k)
        sparse_results = self._sparse.retrieve(query, top_k=self._fetch_k)

        log.info(
            "Hybrid retrieval fetched",
            dense_count=len(dense_results),
            sparse_count=len(sparse_results),
        )

        # Build rank-lookup dicts: chunk_id → 1-based rank
        dense_rank: dict[str, int] = {r.chunk.id: i + 1 for i, r in enumerate(dense_results)}
        sparse_rank: dict[str, int] = {r.chunk.id: i + 1 for i, r in enumerate(sparse_results)}

        # Collect all unique chunks (prefer the Chunk object from dense if present in both)
        chunk_map: dict[str, RetrievedChunk] = {}
        for rc in dense_results:
            chunk_map[rc.chunk.id] = rc
        for rc in sparse_results:
            if rc.chunk.id not in chunk_map:
                chunk_map[rc.chunk.id] = rc

        log.info("Unique chunks after dedup", unique_count=len(chunk_map))

        # Compute RRF scores
        k = self._rrf_k
        rrf_scores: dict[str, float] = {}
        for chunk_id in chunk_map:
            r_d = dense_rank.get(chunk_id, _ABSENT_RANK)
            r_s = sparse_rank.get(chunk_id, _ABSENT_RANK)
            rrf_scores[chunk_id] = 1.0 / (k + r_d) + 1.0 / (k + r_s)

        # Sort descending by RRF score, take top_k
        ranked_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

        results: list[RetrievedChunk] = []
        for chunk_id in ranked_ids:
            rc = chunk_map[chunk_id]
            results.append(RetrievedChunk(chunk=rc.chunk, score=round(rrf_scores[chunk_id], 6)))

        returned_scores = [round(r.score, 6) for r in results]
        log.info("Hybrid retrieval complete", results_count=len(results), scores=returned_scores)
        return results
