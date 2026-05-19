"""Unit tests for HybridRetriever (RRF fusion)."""

from unittest.mock import MagicMock

import pytest

from src.retrieval.hybrid_retriever import HybridRetriever
from src.utils.models import Chunk, RetrievedChunk


def _rc(chunk_id: str, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(id=chunk_id, text=f"text for {chunk_id}", metadata={}),
        score=score,
    )


def _mock_retriever(results: list[RetrievedChunk]) -> MagicMock:
    m = MagicMock()
    m.retrieve.return_value = results
    return m


# ── RRF math ─────────────────────────────────────────────────────────────────

def test_rrf_score_chunk_in_both_lists_higher_than_only_one():
    """A chunk ranked 1st in both lists should outscore a chunk in only one list."""
    shared = _rc("shared")
    dense_only = _rc("dense_only")

    dense = _mock_retriever([shared, dense_only])
    sparse = _mock_retriever([shared])

    hybrid = HybridRetriever(dense=dense, sparse=sparse, rrf_k=60, fetch_k=10)
    results = hybrid.retrieve("query", top_k=5)

    ids = [r.chunk.id for r in results]
    assert ids.index("shared") < ids.index("dense_only"), "shared should rank above dense_only"


def test_rrf_math_exact_values():
    """Verify RRF scores on a small hand-calculable example.

    With k=60:
      chunk A: rank 1 in dense, rank 1 in sparse  → 1/61 + 1/61 ≈ 0.032787
      chunk B: rank 1 in dense, absent in sparse   → 1/61 + 1/10060 ≈ 0.016492
    """
    k = 60
    a = _rc("A")
    b = _rc("B")

    dense = _mock_retriever([a, b])
    sparse = _mock_retriever([a])

    hybrid = HybridRetriever(dense=dense, sparse=sparse, rrf_k=k, fetch_k=10)
    results = hybrid.retrieve("q", top_k=5)

    score_map = {r.chunk.id: r.score for r in results}
    expected_a = 1 / (k + 1) + 1 / (k + 1)
    # B is rank 2 in dense (second in list), absent in sparse
    expected_b = 1 / (k + 2) + 1 / (k + 10_000)

    assert abs(score_map["A"] - expected_a) < 1e-5
    assert abs(score_map["B"] - expected_b) < 1e-5


# ── Deduplication ─────────────────────────────────────────────────────────────

def test_deduplication_same_chunk_id_appears_once():
    """The same chunk_id from both retrievers should appear exactly once."""
    shared = _rc("dup")
    dense = _mock_retriever([shared])
    sparse = _mock_retriever([shared])

    hybrid = HybridRetriever(dense=dense, sparse=sparse, rrf_k=60, fetch_k=10)
    results = hybrid.retrieve("query", top_k=5)

    ids = [r.chunk.id for r in results]
    assert ids.count("dup") == 1


# ── top_k enforcement ────────────────────────────────────────────────────────

def test_top_k_respected():
    chunks = [_rc(f"c{i}") for i in range(10)]
    dense = _mock_retriever(chunks[:5])
    sparse = _mock_retriever(chunks[5:])

    hybrid = HybridRetriever(dense=dense, sparse=sparse, rrf_k=60, fetch_k=10)
    results = hybrid.retrieve("query", top_k=3)
    assert len(results) <= 3


def test_chunk_in_only_one_list_still_appears():
    """Chunks from a single retriever must still be included in output."""
    dense = _mock_retriever([_rc("dense_only")])
    sparse = _mock_retriever([_rc("sparse_only")])

    hybrid = HybridRetriever(dense=dense, sparse=sparse, rrf_k=60, fetch_k=10)
    results = hybrid.retrieve("query", top_k=5)
    ids = {r.chunk.id for r in results}
    assert "dense_only" in ids
    assert "sparse_only" in ids


# ── Empty inputs ──────────────────────────────────────────────────────────────

def test_empty_dense_results():
    dense = _mock_retriever([])
    sparse = _mock_retriever([_rc("s1"), _rc("s2")])

    hybrid = HybridRetriever(dense=dense, sparse=sparse, rrf_k=60, fetch_k=10)
    results = hybrid.retrieve("query", top_k=5)
    assert len(results) == 2


def test_both_empty_returns_empty():
    dense = _mock_retriever([])
    sparse = _mock_retriever([])

    hybrid = HybridRetriever(dense=dense, sparse=sparse, rrf_k=60, fetch_k=10)
    results = hybrid.retrieve("query", top_k=5)
    assert results == []
