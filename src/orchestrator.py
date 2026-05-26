"""RAGOrchestrator — coordinates retrieval and generation for a single query."""

import structlog

from src.generation.base import Generator
from src.retrieval.base import Retriever
from src.utils.exceptions import GenerationError, RetrievalError
from src.utils.models import Answer, RetrievedChunk

logger = structlog.get_logger(__name__)

_STRATEGY_LABELS = {
    "dense":           "dense (cosine similarity)",
    "sparse":          "BM25 (keyword)",
    "hybrid":          "hybrid (BM25 + dense + RRF)",
    "reranked_hybrid": "reranked hybrid (BM25 + dense + RRF + cross-encoder)",
}


class RAGOrchestrator:
    """Coordinates the RAG pipeline: retrieve relevant chunks, then generate an answer."""

    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        query_rewriter=None,
        query_decomposer=None,
        retrieval_strategy: str = "hybrid",
        sub_top_k: int = 4,
        max_chunks: int = 8,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._query_rewriter = query_rewriter
        self._query_decomposer = query_decomposer
        self._retrieval_strategy = retrieval_strategy
        self._sub_top_k = sub_top_k
        self._max_chunks = max_chunks
        logger.info(
            "RAGOrchestrator initialised",
            retriever=type(retriever).__name__,
            generator=type(generator).__name__,
            query_rewriting_enabled=query_rewriter is not None,
            query_decomposition_enabled=query_decomposer is not None,
        )

    def answer(self, query: str, history: list[dict] | None = None) -> Answer:
        """Run the full RAG pipeline and return an Answer with debug info."""
        log = logger.bind(query_length=len(query))
        log.info("Query received")

        retrieval_query = query
        if self._query_rewriter and history:
            retrieval_query = self._query_rewriter.rewrite(query, history)

        # ── Decompose and retrieve ─────────────────────────────────────────────
        try:
            chunks = self._retrieve_with_decomposition(retrieval_query)
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(f"Unexpected retrieval error: {exc}") from exc

        log.info("Chunks retrieved", chunk_count=len(chunks))

        # ── Generate ──────────────────────────────────────────────────────────
        try:
            result = self._generator.generate(retrieval_query, chunks)
        except GenerationError:
            raise
        except Exception as exc:
            raise GenerationError(f"Unexpected generation error: {exc}") from exc

        log.info("Answer generated", answer_length=len(result.text), citation_count=len(result.citations))
        result.rewritten_query = retrieval_query if retrieval_query != query else None
        result.debug = self._build_debug(query, retrieval_query, chunks)
        return result

    def _retrieve_with_decomposition(self, query: str) -> list[RetrievedChunk]:
        """Decompose query if needed, retrieve per sub-query, merge results."""
        if self._query_decomposer is None:
            return self._retriever.retrieve(query, top_k=5)

        sub_queries = self._query_decomposer.decompose(query)

        # Single sub-query — normal path, no overhead
        if len(sub_queries) == 1:
            return self._retriever.retrieve(sub_queries[0], top_k=5)

        # Multiple sub-queries — retrieve per sub-query, merge by chunk ID
        seen: dict[str, RetrievedChunk] = {}
        for sq in sub_queries:
            for rc in self._retriever.retrieve(sq, top_k=self._sub_top_k):
                cid = rc.chunk.id
                if cid not in seen or rc.score > seen[cid].score:
                    seen[cid] = rc

        merged = sorted(seen.values(), key=lambda rc: rc.score, reverse=True)
        return merged[: self._max_chunks]

    def _build_debug(self, original_query: str, retrieval_query: str, chunks) -> dict:
        try:
            afr_debug = getattr(self._retriever, "last_debug", {}) or {}
            inner = getattr(self._retriever, "_base", None)
            rr_debug = getattr(inner, "last_debug", None)

            pinned_ids: set[str] = afr_debug.get("pinned_ids", set())

            return {
                "original_query": original_query,
                "rewritten_query": retrieval_query,
                "article_filter": {
                    "matched": afr_debug.get("matched", False),
                    "article_id": afr_debug.get("article_id"),
                    "pinned_count": afr_debug.get("pinned_count", 0),
                },
                "retrieval_strategy": _STRATEGY_LABELS.get(
                    self._retrieval_strategy, self._retrieval_strategy
                ),
                "chunks": [
                    {
                        "id": f"chunk_{i}",
                        "score": rc.score,
                        "pinned": rc.chunk.id in pinned_ids,
                        "source": rc.chunk.metadata.get("filename", ""),
                        "article_id": rc.chunk.metadata.get("article_id"),
                        "section_title": rc.chunk.metadata.get("section_title"),
                        "preview": rc.chunk.text[:80],
                    }
                    for i, rc in enumerate(chunks)
                ],
                "reranker": rr_debug if rr_debug else None,
            }
        except Exception as exc:
            logger.warning("debug_build_failed", error=str(exc))
            return {}
