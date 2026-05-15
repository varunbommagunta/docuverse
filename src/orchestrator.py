"""RAGOrchestrator — coordinates retrieval and generation for a single query.

The orchestrator is the application's core use-case class. It knows nothing
about HTTP or Streamlit; it speaks exclusively in domain objects (Answer,
RetrievedChunk). FastAPI routes and the Streamlit UI call it through the
injected instance.

Dependencies (Retriever, Generator) are injected at construction time so they
can be swapped without modifying this class — a textbook application of the
Dependency Inversion principle.
"""

import structlog

from src.generation.base import Generator
from src.retrieval.base import Retriever
from src.utils.models import Answer

logger = structlog.get_logger(__name__)


class RAGOrchestrator:
    """Coordinates the RAG pipeline: retrieve relevant chunks, then generate an answer.

    This class is intentionally thin in Phase 0 — it defines the public contract
    (method signatures and docstrings) without any implementation. Phase 3 will
    fill in the body of `answer()`.
    """

    def __init__(self, retriever: Retriever, generator: Generator) -> None:
        """Inject retriever and generator dependencies.

        Args:
            retriever: Any object satisfying the Retriever Protocol.
            generator: Any object satisfying the Generator Protocol.
        """
        self._retriever = retriever
        self._generator = generator
        logger.info("RAGOrchestrator initialised", retriever=type(retriever).__name__, generator=type(generator).__name__)

    def answer(self, query: str) -> Answer:
        """Run the full RAG pipeline for a user query.

        Steps (Phase 3 implementation):
          1. Call self._retriever.retrieve(query, top_k) → list[RetrievedChunk]
          2. Call self._generator.generate(query, chunks) → Answer
          3. Log the query, number of chunks, and answer length
          4. Return the Answer

        Args:
            query: Natural-language question from the user.

        Returns:
            Answer containing generated text, citation IDs, and source chunks.

        Raises:
            RetrievalError: If retrieval fails.
            GenerationError: If generation fails.
            NotImplementedError: Until Phase 3 implementation is added.
        """
        raise NotImplementedError("RAGOrchestrator.answer() will be implemented in Phase 3.")
