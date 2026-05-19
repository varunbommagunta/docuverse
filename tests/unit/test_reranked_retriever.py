"""Unit tests for RerankedRetriever."""

from unittest.mock import MagicMock

import pytest

from src.retrieval.reranked_retriever import RerankedRetriever
from src.utils.models import Chunk, RetrievedChunk


def _rc(chunk_id: str, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(id=chunk_id, text=f"text {chunk_id}", metadata={}),
        score=score,
    )


def _mock_retriever(results: list[RetrievedChunk]) -> MagicMock:
    m = MagicMock()
    m.retrieve.return_value = results
    return m


def _mock_reranker(results: list[RetrievedChunk]) -> MagicMock:
    m = MagicMock()
    m.rerank.return_value = results
    return m


def test_retrieve_calls_base_with_fetch_k() -> None:
    base = _mock_retriever([_rc("c1")])
    reranker = _mock_reranker([_rc("c1")])
    rr = RerankedRetriever(base=base, reranker=reranker, fetch_k=30)

    rr.retrieve("query", top_k=5)

    base.retrieve.assert_called_once_with("query", top_k=30)


def test_retrieve_passes_candidates_to_reranker() -> None:
    candidates = [_rc(f"c{i}") for i in range(10)]
    base = _mock_retriever(candidates)
    reranker = _mock_reranker(candidates[:5])
    rr = RerankedRetriever(base=base, reranker=reranker, fetch_k=10)

    rr.retrieve("query", top_k=5)

    reranker.rerank.assert_called_once_with("query", candidates, top_k=5)


def test_retrieve_returns_reranker_output() -> None:
    expected = [_rc("best"), _rc("second")]
    base = _mock_retriever([_rc("a"), _rc("b"), _rc("c")])
    reranker = _mock_reranker(expected)
    rr = RerankedRetriever(base=base, reranker=reranker, fetch_k=10)

    results = rr.retrieve("query", top_k=2)
    assert results == expected


def test_retrieve_top_k_passed_to_reranker() -> None:
    base = _mock_retriever([_rc(f"c{i}") for i in range(20)])
    reranker = _mock_reranker([_rc("top")])
    rr = RerankedRetriever(base=base, reranker=reranker, fetch_k=20)

    rr.retrieve("query", top_k=7)

    call_args = reranker.rerank.call_args
    # top_k is passed as a keyword argument
    assert call_args.kwargs.get("top_k") == 7
