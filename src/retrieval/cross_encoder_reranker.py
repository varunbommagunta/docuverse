"""Cross-encoder reranker using sentence-transformers.

CrossEncoderReranker implements the Reranker Protocol. It uses a cross-encoder
model (query + document concatenated as input) to score each candidate chunk.
Cross-encoders are more accurate than bi-encoders for relevance judgment but
require O(n) model calls per query, making them suitable only for reranking
a small candidate set (typically 20-100 chunks).

The model is loaded lazily on the first rerank() call to avoid adding startup
latency during factory wiring — the cross-encoder model takes ~1s to load.

Scores from the model are logits (unbounded). They are mapped to [0, 1] via
the sigmoid function so they are comparable to other score fields.
"""

import math
import time
from typing import Any

import structlog

from src.utils.models import RetrievedChunk

logger = structlog.get_logger(__name__)


def _sigmoid(x: float) -> float:
    """Map an unbounded logit to (0, 1)."""
    return 1.0 / (1.0 + math.exp(-x))


class CrossEncoderReranker:
    """Implements the Reranker Protocol using a sentence-transformers cross-encoder.

    The cross-encoder model is loaded lazily on first use to keep factory
    wiring fast. After loading, the model instance is cached for subsequent calls.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        """Initialise the reranker with a model name.

        The model is NOT loaded here. It is loaded on the first call to rerank().

        Args:
            model_name: sentence-transformers cross-encoder model identifier.
        """
        self._model_name = model_name
        self._model: Any = None
        logger.info("CrossEncoderReranker configured", model=model_name)

    def _load_model(self) -> None:
        """Load the cross-encoder model (called once, cached afterwards)."""
        from sentence_transformers.cross_encoder import CrossEncoder  # lazy import

        logger.info("Loading cross-encoder model", model=self._model_name)
        t0 = time.time()
        self._model = CrossEncoder(self._model_name)
        logger.info("Cross-encoder model loaded", elapsed_s=round(time.time() - t0, 2))

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int = 5
    ) -> list[RetrievedChunk]:
        """Re-score candidates using the cross-encoder and return top_k.

        Args:
            query: The original natural-language question.
            candidates: Pre-retrieved chunks to re-score. May be any length.
            top_k: How many to return after reranking.

        Returns:
            Top-k RetrievedChunks with score updated to sigmoid(cross_encoder_logit).
            Ordered by descending reranker score.
        """
        log = logger.bind(query_length=len(query), candidates=len(candidates), top_k=top_k)
        log.info("Reranking started")

        if not candidates:
            return []

        if self._model is None:
            self._load_model()

        t0 = time.time()
        pairs = [(query, rc.chunk.text) for rc in candidates]
        raw_scores = self._model.predict(pairs)
        elapsed = time.time() - t0

        reranked = sorted(
            zip(raw_scores, candidates, strict=False),
            key=lambda x: float(x[0]),
            reverse=True,
        )[:top_k]

        results: list[RetrievedChunk] = []
        for raw_score, rc in reranked:
            normalized = round(_sigmoid(float(raw_score)), 4)
            results.append(RetrievedChunk(chunk=rc.chunk, score=normalized))

        returned_scores = [r.score for r in results]
        log.info(
            "Reranking complete",
            results_count=len(results),
            scores=returned_scores,
            elapsed_s=round(elapsed, 3),
        )
        return results
