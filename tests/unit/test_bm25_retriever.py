"""Unit tests for BM25Retriever."""

from unittest.mock import MagicMock

import pytest

from src.retrieval.bm25_retriever import BM25Retriever, _tokenize
from src.utils.models import Chunk, RetrievedChunk


def _make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(id=chunk_id, text=text, metadata={"source": "test"})


def _make_vector_store(chunks: list[Chunk]) -> MagicMock:
    vs = MagicMock()
    vs.get_all_chunks.return_value = chunks
    return vs


_CORPUS = [
    _make_chunk("c1", "Jupiter is the largest planet in the solar system"),
    _make_chunk("c2", "The Constitution of India establishes fundamental rights"),
    _make_chunk("c3", "BM25 is a bag-of-words retrieval function used in information retrieval"),
    _make_chunk("c4", "Ethics in governance requires accountability and transparency"),
    _make_chunk("c5", "Saturn has a spectacular ring system made of ice and rock"),
]


@pytest.fixture
def retriever() -> BM25Retriever:
    return BM25Retriever(vector_store=_make_vector_store(_CORPUS))


def test_retrieve_returns_list(retriever: BM25Retriever) -> None:
    results = retriever.retrieve("largest planet", top_k=3)
    assert isinstance(results, list)


def test_retrieve_respects_top_k(retriever: BM25Retriever) -> None:
    results = retriever.retrieve("planet solar system", top_k=2)
    assert len(results) <= 2


def test_exact_keyword_match_scores_high(retriever: BM25Retriever) -> None:
    results = retriever.retrieve("Jupiter largest planet", top_k=5)
    top_id = results[0].chunk.id
    assert top_id == "c1", f"Expected c1 at top, got {top_id}"


def test_scores_normalized_to_0_1(retriever: BM25Retriever) -> None:
    results = retriever.retrieve("planet", top_k=5)
    for r in results:
        assert 0.0 <= r.score <= 1.0


def test_top_result_has_highest_score(retriever: BM25Retriever) -> None:
    results = retriever.retrieve("Constitution India rights", top_k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_empty_corpus_returns_empty() -> None:
    empty_vs = _make_vector_store([])
    r = BM25Retriever(vector_store=empty_vs)
    results = r.retrieve("anything", top_k=5)
    assert results == []


def test_retrieve_preserves_chunk_metadata(retriever: BM25Retriever) -> None:
    results = retriever.retrieve("BM25 information retrieval", top_k=1)
    assert results[0].chunk.metadata["source"] == "test"


def test_tokenize_lowercases() -> None:
    tokens = _tokenize("Hello WORLD foo")
    assert tokens == ["hello", "world", "foo"]


def test_tokenize_splits_on_whitespace() -> None:
    tokens = _tokenize("a b  c")
    assert "a" in tokens and "b" in tokens and "c" in tokens
