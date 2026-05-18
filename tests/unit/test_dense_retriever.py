"""Unit tests for DenseRetriever using mock embedder and vector store."""

from unittest.mock import MagicMock

import pytest

from src.retrieval.dense_retriever import DenseRetriever
from src.utils.exceptions import RetrievalError
from src.utils.models import Chunk, RetrievedChunk


def _make_retrieved(text: str = "sample text", score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(chunk=Chunk(id="chunk-1", text=text, metadata={}), score=score)


@pytest.fixture
def mock_embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.embed_single.return_value = [0.1, 0.2, 0.3]
    return embedder


@pytest.fixture
def mock_vector_store() -> MagicMock:
    vs = MagicMock()
    vs.similarity_search.return_value = [_make_retrieved(), _make_retrieved(score=0.7)]
    return vs


@pytest.fixture
def retriever(mock_embedder: MagicMock, mock_vector_store: MagicMock) -> DenseRetriever:
    return DenseRetriever(embedder=mock_embedder, vector_store=mock_vector_store)


def test_retrieve_returns_correct_count(
    retriever: DenseRetriever,
    mock_vector_store: MagicMock,
) -> None:
    results = retriever.retrieve("What is the capital?", top_k=2)
    assert len(results) == 2


def test_retrieve_calls_embed_single(
    retriever: DenseRetriever,
    mock_embedder: MagicMock,
) -> None:
    query = "What is the capital?"
    retriever.retrieve(query, top_k=3)
    mock_embedder.embed_single.assert_called_once_with(query)


def test_retrieve_calls_similarity_search(
    retriever: DenseRetriever,
    mock_embedder: MagicMock,
    mock_vector_store: MagicMock,
) -> None:
    retriever.retrieve("query", top_k=5)
    mock_vector_store.similarity_search.assert_called_once_with([0.1, 0.2, 0.3], top_k=5)


def test_retrieve_returns_retrieved_chunks(retriever: DenseRetriever) -> None:
    results = retriever.retrieve("query")
    assert all(isinstance(r, RetrievedChunk) for r in results)


def test_retrieve_propagates_retrieval_error(
    mock_embedder: MagicMock,
    mock_vector_store: MagicMock,
) -> None:
    mock_embedder.embed_single.side_effect = RetrievalError("API down")
    retriever = DenseRetriever(embedder=mock_embedder, vector_store=mock_vector_store)
    with pytest.raises(RetrievalError, match="API down"):
        retriever.retrieve("test query")


def test_retrieve_wraps_unexpected_exception(
    mock_embedder: MagicMock,
    mock_vector_store: MagicMock,
) -> None:
    mock_embedder.embed_single.side_effect = ConnectionError("network error")
    retriever = DenseRetriever(embedder=mock_embedder, vector_store=mock_vector_store)
    with pytest.raises(RetrievalError):
        retriever.retrieve("test query")
