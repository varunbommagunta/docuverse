"""Application-level factory — wires all components from Settings.

get_rag_components() is the single entry point that reads configuration and
constructs the full dependency graph: embedder → vector store → retriever →
generator → orchestrator, plus ingestion pipeline. Called once in the FastAPI
lifespan and the results stored on app.state.

Retrieval strategies (set RETRIEVAL_STRATEGY env var):
  dense            — DenseRetriever (default, cosine similarity)
  sparse           — BM25Retriever (keyword search only)
  hybrid           — HybridRetriever (dense + BM25, RRF fusion)
  reranked_hybrid  — RerankedRetriever(HybridRetriever + CrossEncoderReranker)
"""

import structlog
from openai import OpenAI

from config.settings import get_settings
from src.generation.openai_generator import OpenAIGenerator
from src.ingestion.chunkers import RecursiveChunker
from src.ingestion.classifier import DocumentClassifier, LLMClassifier
from src.ingestion.parsers import PyPDFParser
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.router import ChunkerRouter
from src.orchestrator import RAGOrchestrator
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.query_rewriter import OpenAIQueryRewriter
from src.retrieval.embedders import OpenAIEmbedder
from src.retrieval.vector_store import ChromaVectorStore

logger = structlog.get_logger(__name__)

_VALID_STRATEGIES = {"dense", "sparse", "hybrid", "reranked_hybrid"}


def _build_retriever(strategy: str, embedder, vector_store, settings):
    """Construct the appropriate retriever for the given strategy."""
    if strategy == "dense":
        return DenseRetriever(embedder=embedder, vector_store=vector_store)

    if strategy == "sparse":
        from src.retrieval.bm25_retriever import BM25Retriever
        return BM25Retriever(vector_store=vector_store)

    if strategy in ("hybrid", "reranked_hybrid"):
        from src.retrieval.bm25_retriever import BM25Retriever
        from src.retrieval.hybrid_retriever import HybridRetriever

        dense = DenseRetriever(embedder=embedder, vector_store=vector_store)
        sparse = BM25Retriever(vector_store=vector_store)
        hybrid = HybridRetriever(
            dense=dense,
            sparse=sparse,
            rrf_k=settings.hybrid_rrf_k,
            fetch_k=max(settings.hybrid_dense_top_k, settings.hybrid_sparse_top_k),
        )

        if strategy == "hybrid":
            return hybrid

        # reranked_hybrid
        from src.retrieval.cross_encoder_reranker import CrossEncoderReranker
        from src.retrieval.reranked_retriever import RerankedRetriever

        reranker = CrossEncoderReranker(model_name=settings.reranker_model)
        return RerankedRetriever(
            base=hybrid,
            reranker=reranker,
            fetch_k=settings.reranker_fetch_k,
        )

    raise ValueError(
        f"Unknown retrieval_strategy '{strategy}'. "
        f"Valid options: {sorted(_VALID_STRATEGIES)}"
    )


def get_rag_components() -> tuple[RAGOrchestrator, IngestionPipeline]:
    """Construct and return the orchestrator and ingestion pipeline.

    All components are instantiated from Settings values — no hardcoded config.

    Returns:
        (RAGOrchestrator, IngestionPipeline) tuple ready for use by the API.
    """
    settings = get_settings()
    strategy = settings.retrieval_strategy
    logger.info("Building RAG components", retrieval_strategy=strategy)

    if strategy not in _VALID_STRATEGIES:
        raise ValueError(
            f"Unknown retrieval_strategy '{strategy}'. "
            f"Valid options: {sorted(_VALID_STRATEGIES)}"
        )

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

    # Build the classifier + router
    openai_client = OpenAI(api_key=settings.openai_api_key)
    llm_classifier = (
        LLMClassifier(openai_client)
        if getattr(settings, "enable_llm_classifier_fallback", True)
        else None
    )
    classifier = DocumentClassifier(llm_classifier=llm_classifier)
    chunker_router = ChunkerRouter(classifier=classifier)

    pipeline = IngestionPipeline(
        parser=parser,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        chunker_router=chunker_router,
    )

    # ── Query pipeline ────────────────────────────────────────────────────────
    from src.retrieval.article_filter_retriever import ArticleFilterRetriever

    retriever = _build_retriever(strategy, embedder, vector_store, settings)
    retriever = ArticleFilterRetriever(base=retriever, vector_store=vector_store)
    generator = OpenAIGenerator(
        model=settings.openai_model,
        temperature=0.0,
        max_tokens=1024,
    )

    # Build query rewriter (or None if disabled)
    if settings.enable_query_rewriting:
        query_rewriter = OpenAIQueryRewriter(
            api_key=settings.openai_api_key,
            model=settings.query_rewriter_model,
            max_history_turns=settings.query_rewriter_max_history_turns,
        )
    else:
        query_rewriter = None

    orchestrator = RAGOrchestrator(retriever=retriever, generator=generator, query_rewriter=query_rewriter)

    logger.info("RAG components ready", retrieval_strategy=strategy)
    return orchestrator, pipeline
