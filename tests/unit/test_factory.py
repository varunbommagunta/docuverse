"""Unit tests for factory.py — verifies correct retriever is built per strategy."""

from unittest.mock import MagicMock, patch

import pytest

from src.factory import _build_retriever, get_rag_components


def _make_settings(strategy: str) -> MagicMock:
    s = MagicMock()
    s.retrieval_strategy = strategy
    s.embedding_model = "text-embedding-3-small"
    s.chroma_persist_directory = "./data/chroma_db"
    s.chunk_size = 500
    s.chunk_overlap = 50
    s.openai_model = "gpt-4o-mini"
    s.top_k = 5
    s.hybrid_dense_top_k = 20
    s.hybrid_sparse_top_k = 20
    s.hybrid_rrf_k = 60
    s.reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    s.reranker_fetch_k = 50
    return s


# ── _build_retriever ─────────────────────────────────────────────────────────

def test_build_retriever_dense_returns_dense_retriever() -> None:
    from src.retrieval.dense_retriever import DenseRetriever

    embedder = MagicMock()
    vs = MagicMock()
    settings = _make_settings("dense")
    retriever = _build_retriever("dense", embedder, vs, settings)
    assert isinstance(retriever, DenseRetriever)


def test_build_retriever_sparse_returns_bm25_retriever() -> None:
    from src.retrieval.bm25_retriever import BM25Retriever

    embedder = MagicMock()
    vs = MagicMock()
    vs.get_all_chunks.return_value = []
    settings = _make_settings("sparse")
    retriever = _build_retriever("sparse", embedder, vs, settings)
    assert isinstance(retriever, BM25Retriever)


def test_build_retriever_hybrid_returns_hybrid_retriever() -> None:
    from src.retrieval.hybrid_retriever import HybridRetriever

    embedder = MagicMock()
    vs = MagicMock()
    vs.get_all_chunks.return_value = []
    settings = _make_settings("hybrid")
    retriever = _build_retriever("hybrid", embedder, vs, settings)
    assert isinstance(retriever, HybridRetriever)


def test_build_retriever_reranked_hybrid_returns_reranked_retriever() -> None:
    from src.retrieval.reranked_retriever import RerankedRetriever

    embedder = MagicMock()
    vs = MagicMock()
    vs.get_all_chunks.return_value = []
    settings = _make_settings("reranked_hybrid")
    retriever = _build_retriever("reranked_hybrid", embedder, vs, settings)
    assert isinstance(retriever, RerankedRetriever)


def test_build_retriever_unknown_strategy_raises() -> None:
    with pytest.raises(ValueError, match="Unknown retrieval_strategy"):
        _build_retriever("nonexistent", MagicMock(), MagicMock(), _make_settings("nonexistent"))


# ── get_rag_components ────────────────────────────────────────────────────────

_PATCH_TARGETS = [
    "src.factory.get_settings",
    "src.factory.OpenAIEmbedder",
    "src.factory.ChromaVectorStore",
    "src.factory.PyPDFParser",
    "src.factory.RecursiveChunker",
    "src.factory.IngestionPipeline",
    "src.factory.OpenAIGenerator",
    "src.factory.RAGOrchestrator",
    "src.factory.DenseRetriever",
]


def _patch_factory(strategy: str):
    """Context manager that patches all factory dependencies."""
    from contextlib import ExitStack
    from unittest.mock import patch as _patch

    stack = ExitStack()
    mocks = {}
    for target in _PATCH_TARGETS:
        name = target.split(".")[-1]
        m = stack.enter_context(_patch(target))
        mocks[name] = m

    mocks["get_settings"].return_value = _make_settings(strategy)
    # ChromaVectorStore instance needs get_all_chunks for non-dense strategies
    vs_instance = MagicMock()
    vs_instance.get_all_chunks.return_value = []
    mocks["ChromaVectorStore"].return_value = vs_instance

    return stack, mocks


def test_get_rag_components_dense_strategy() -> None:
    stack, mocks = _patch_factory("dense")
    with stack:
        orchestrator, pipeline = get_rag_components()
    mocks["DenseRetriever"].assert_called_once()


def test_get_rag_components_returns_tuple() -> None:
    stack, _ = _patch_factory("dense")
    with stack:
        result = get_rag_components()
    assert len(result) == 2


def test_get_rag_components_invalid_strategy_raises() -> None:
    stack, mocks = _patch_factory("bad_strategy")
    mocks["get_settings"].return_value = _make_settings("bad_strategy")
    with stack:
        with pytest.raises(ValueError, match="Unknown retrieval_strategy"):
            get_rag_components()
