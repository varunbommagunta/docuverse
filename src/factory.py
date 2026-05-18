"""Application-level factory — wires all components from Settings.

get_rag_components() is the single entry point that reads configuration and
constructs the full dependency graph: embedder → vector store → retriever →
generator → orchestrator, plus ingestion pipeline. Called once in the FastAPI
lifespan and the results stored on app.state.
"""

import structlog

from config.settings import get_settings
from src.generation.openai_generator import OpenAIGenerator
from src.ingestion.chunkers import RecursiveChunker
from src.ingestion.parsers import PyPDFParser
from src.ingestion.pipeline import IngestionPipeline
from src.orchestrator import RAGOrchestrator
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.embedders import OpenAIEmbedder
from src.retrieval.vector_store import ChromaVectorStore

logger = structlog.get_logger(__name__)


def get_rag_components() -> tuple[RAGOrchestrator, IngestionPipeline]:
    """Construct and return the orchestrator and ingestion pipeline.

    All components are instantiated from Settings values — no hardcoded config.

    Returns:
        (RAGOrchestrator, IngestionPipeline) tuple ready for use by the API.
    """
    settings = get_settings()
    logger.info("Building RAG components")

    # ── Shared infrastructure ─────────────────────────────────────────────────
    embedder = OpenAIEmbedder(model=settings.embedding_model)
    vector_store = ChromaVectorStore(
        persist_directory=settings.chroma_persist_directory,
    )

    # ── Ingestion pipeline ────────────────────────────────────────────────────
    parser = PyPDFParser()
    chunker = RecursiveChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    pipeline = IngestionPipeline(
        parser=parser,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
    )

    # ── Query pipeline ────────────────────────────────────────────────────────
    retriever = DenseRetriever(embedder=embedder, vector_store=vector_store)
    generator = OpenAIGenerator(
        model=settings.openai_model,
        temperature=0.0,
        max_tokens=1024,
    )
    orchestrator = RAGOrchestrator(retriever=retriever, generator=generator)

    logger.info("RAG components ready")
    return orchestrator, pipeline
