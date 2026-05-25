"""Metadata-filter decorator that pins exact article matches ahead of semantic results.

ArticleFilterRetriever wraps any Retriever. When the query names a specific
article (e.g. "what does Article 16 say?"), it fetches that article's chunks
directly from Chroma via a metadata filter and prepends them to the semantic
results, deduplicating by chunk ID. If the query contains no article reference
the base retriever is called unchanged.

This is applied to the rewritten query, not the raw user input, so pronoun
resolution ("what does it say?" → "what does Article 16 say?") triggers the
filter correctly.
"""

import re

import structlog

from src.retrieval.vector_store import ChromaVectorStore
from src.utils.models import RetrievedChunk

logger = structlog.get_logger(__name__)

_ARTICLE_RE = re.compile(r"article\s+(\d+[A-Za-z]*)", re.IGNORECASE)


class ArticleFilterRetriever:
    """Implements the Retriever Protocol with an article-ID pre-fetch layer."""

    def __init__(self, base, vector_store: ChromaVectorStore) -> None:
        """Inject base retriever and vector store.

        Args:
            base: Any Retriever Protocol implementation (e.g. RerankedRetriever).
            vector_store: ChromaVectorStore instance used for the metadata lookup.
        """
        self._base = base
        self._vector_store = vector_store
        self.last_debug: dict = {"matched": False, "article_id": None, "pinned_count": 0, "pinned_ids": set()}

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Retrieve chunks, optionally pinning exact article match at rank 0.

        Args:
            query: Rewritten natural-language question.
            top_k: Total number of chunks to return.

        Returns:
            List of RetrievedChunk ordered by relevance. If an article number
            was found, its chunk(s) appear first; semantic results fill the rest
            up to top_k, with duplicates removed.
        """
        match = _ARTICLE_RE.search(query)

        if not match:
            self.last_debug = {"matched": False, "article_id": None, "pinned_count": 0, "pinned_ids": set()}
            return self._base.retrieve(query, top_k=top_k)

        article_id = match.group(1).upper()  # normalise "21a" → "21A"
        logger.info("article_filter_triggered", article_id=article_id, query=query)

        # Fetch article chunks by exact metadata match
        pinned = self._vector_store.get_by_article_id(article_id)

        if not pinned:
            # No chunks for this article in the store — fall back silently
            logger.info("article_filter_no_match", article_id=article_id)
            self.last_debug = {"matched": True, "article_id": article_id, "pinned_count": 0, "pinned_ids": set()}
            return self._base.retrieve(query, top_k=top_k)

        pinned_ids: set[str] = {rc.chunk.id for rc in pinned}
        self.last_debug = {"matched": True, "article_id": article_id, "pinned_count": len(pinned), "pinned_ids": pinned_ids}

        # Semantic search for remaining slots; fetch extra to cover dedup losses
        semantic_top_k = max(top_k, top_k + len(pinned))
        semantic = self._base.retrieve(query, top_k=semantic_top_k)

        # Remove chunks already covered by the pinned set
        semantic_deduped = [rc for rc in semantic if rc.chunk.id not in pinned_ids]

        # Combine: pinned first, then semantic fill up to top_k
        remaining_slots = max(0, top_k - len(pinned))
        results = pinned + semantic_deduped[:remaining_slots]

        logger.info(
            "article_filter_complete",
            article_id=article_id,
            pinned_count=len(pinned),
            semantic_kept=len(semantic_deduped[:remaining_slots]),
            total=len(results),
        )
        return results
