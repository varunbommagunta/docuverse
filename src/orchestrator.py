"""RAGOrchestrator — coordinates retrieval and generation for a single query.

Phase 1 implementation: replaces the Phase 0 stub with a real two-step
pipeline. The orchestrator itself has no knowledge of HTTP, Streamlit, or
any external service — those details are hidden behind the injected components.
"""

import structlog

from src.generation.base import Generator
from src.retrieval.base import Retriever
from src.utils.exceptions import GenerationError, RetrievalError
from src.utils.models import Answer

logger = structlog.get_logger(__name__)


class RAGOrchestrator:
    """Coordinates the RAG pipeline: retrieve relevant chunks, then generate an answer."""

    def __init__(self, retriever: Retriever, generator: Generator, query_rewriter=None) -> None:
        """Inject retriever and generator dependencies.

        Args:
            retriever: Any object satisfying the Retriever Protocol.
            generator: Any object satisfying the Generator Protocol.
            query_rewriter: Optional QueryRewriter for conversational follow-up handling.
        """
        self._retriever = retriever
        self._generator = generator
        self._query_rewriter = query_rewriter
        logger.info(
            "RAGOrchestrator initialised",
            retriever=type(retriever).__name__,
            generator=type(generator).__name__,
            query_rewriting_enabled=query_rewriter is not None,
        )

    def answer(self, query: str, history: list[dict] | None = None) -> Answer:
        """Run the full RAG pipeline for a user query.

        Args:
            query: Natural-language question from the user.
            history: Optional conversation history for query rewriting.

        Returns:
            Answer with generated text, citation indices, and source chunks.

        Raises:
            RetrievalError: If the retrieval step fails.
            GenerationError: If the generation step fails.
        """
        log = logger.bind(query_length=len(query))
        log.info("Query received")

        # Rewrite query if we have a rewriter AND history
        if self._query_rewriter and history:
            retrieval_query = self._query_rewriter.rewrite(query, history)
        else:
            retrieval_query = query

        # ── Step 1: Retrieve ──────────────────────────────────────────────────
        try:
            chunks = self._retriever.retrieve(retrieval_query, top_k=5)
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(f"Unexpected retrieval error: {exc}") from exc

        log.info("Chunks retrieved", chunk_count=len(chunks))

        # ── Step 2: Generate ──────────────────────────────────────────────────
        try:
            result = self._generator.generate(retrieval_query, chunks)
        except GenerationError:
            raise
        except Exception as exc:
            raise GenerationError(f"Unexpected generation error: {exc}") from exc

        log.info("Answer generated", answer_length=len(result.text), citation_count=len(result.citations))
        return result
