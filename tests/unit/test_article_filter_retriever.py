"""Unit tests for ArticleFilterRetriever."""

from unittest.mock import MagicMock

import pytest

from src.retrieval.article_filter_retriever import ArticleFilterRetriever
from src.utils.models import Chunk, RetrievedChunk


def _rc(chunk_id: str, text: str = "text", score: float = 0.8) -> RetrievedChunk:
    return RetrievedChunk(chunk=Chunk(id=chunk_id, text=text, metadata={}), score=score)


@pytest.fixture
def mock_base():
    m = MagicMock()
    m.retrieve.return_value = [_rc("s1"), _rc("s2"), _rc("s3"), _rc("s4"), _rc("s5")]
    return m


@pytest.fixture
def mock_store():
    m = MagicMock()
    m.get_by_article_id.return_value = []
    return m


@pytest.fixture
def retriever(mock_base, mock_store):
    return ArticleFilterRetriever(base=mock_base, vector_store=mock_store)


# ── No article in query ───────────────────────────────────────────────────────

def test_no_article_skips_filter(retriever, mock_base, mock_store):
    retriever.retrieve("what are the fundamental rights?", top_k=5)
    mock_store.get_by_article_id.assert_not_called()
    mock_base.retrieve.assert_called_once_with("what are the fundamental rights?", top_k=5)


# ── Article found in query ────────────────────────────────────────────────────

def test_article_query_triggers_lookup(retriever, mock_store):
    retriever.retrieve("what does Article 16 say?", top_k=5)
    mock_store.get_by_article_id.assert_called_once_with("16")


def test_article_number_case_normalised(retriever, mock_store):
    retriever.retrieve("explain article 21a", top_k=5)
    mock_store.get_by_article_id.assert_called_once_with("21A")


def test_pinned_chunk_is_first(mock_base, mock_store):
    pinned = _rc("pinned", score=1.0)
    mock_store.get_by_article_id.return_value = [pinned]
    mock_base.retrieve.return_value = [_rc("s1"), _rc("s2"), _rc("s3"), _rc("s4"), _rc("s5")]

    r = ArticleFilterRetriever(base=mock_base, vector_store=mock_store)
    results = r.retrieve("what does Article 16 say?", top_k=5)

    assert results[0].chunk.id == "pinned"
    assert results[0].score == 1.0


def test_total_count_respects_top_k(mock_base, mock_store):
    pinned = _rc("pinned", score=1.0)
    mock_store.get_by_article_id.return_value = [pinned]
    mock_base.retrieve.return_value = [_rc(f"s{i}") for i in range(10)]

    r = ArticleFilterRetriever(base=mock_base, vector_store=mock_store)
    results = r.retrieve("what does Article 16 say?", top_k=5)

    assert len(results) == 5


def test_pinned_chunk_deduplicated_from_semantic(mock_base, mock_store):
    shared_id = "shared"
    pinned = _rc(shared_id, score=1.0)
    mock_store.get_by_article_id.return_value = [pinned]
    # semantic results include the same chunk
    mock_base.retrieve.return_value = [_rc(shared_id, score=0.9), _rc("s2"), _rc("s3"), _rc("s4"), _rc("s5")]

    r = ArticleFilterRetriever(base=mock_base, vector_store=mock_store)
    results = r.retrieve("what does Article 312 say?", top_k=5)

    ids = [rc.chunk.id for rc in results]
    assert ids.count(shared_id) == 1  # appears exactly once
    assert ids[0] == shared_id         # pinned version (score=1.0) is first


def test_no_article_match_in_store_falls_back(mock_base, mock_store):
    mock_store.get_by_article_id.return_value = []
    semantic = [_rc(f"s{i}") for i in range(5)]
    mock_base.retrieve.return_value = semantic

    r = ArticleFilterRetriever(base=mock_base, vector_store=mock_store)
    results = r.retrieve("what does Article 999 say?", top_k=5)

    assert results == semantic
    mock_base.retrieve.assert_called_once_with("what does Article 999 say?", top_k=5)
