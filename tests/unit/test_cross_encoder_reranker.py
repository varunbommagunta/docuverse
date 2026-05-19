"""Unit tests for CrossEncoderReranker."""

import math
from unittest.mock import MagicMock, patch

import pytest

from src.retrieval.cross_encoder_reranker import CrossEncoderReranker, _sigmoid
from src.utils.models import Chunk, RetrievedChunk


def _rc(chunk_id: str, text: str = "some text", score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(id=chunk_id, text=text, metadata={"source": "test"}),
        score=score,
    )


def _make_reranker_with_mock_model(raw_scores: list[float]) -> CrossEncoderReranker:
    """Return a CrossEncoderReranker whose model.predict() returns raw_scores."""
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker._model_name = "mock-model"

    mock_model = MagicMock()
    mock_model.predict.return_value = raw_scores
    reranker._model = mock_model
    return reranker


# ── Sigmoid helper ────────────────────────────────────────────────────────────

def test_sigmoid_zero_returns_half() -> None:
    assert abs(_sigmoid(0.0) - 0.5) < 1e-9


def test_sigmoid_large_positive_approaches_one() -> None:
    assert _sigmoid(100.0) > 0.999


def test_sigmoid_large_negative_approaches_zero() -> None:
    assert _sigmoid(-100.0) < 0.001


# ── Reranking behaviour ──────────────────────────────────────────────────────

def test_rerank_reorders_by_score() -> None:
    candidates = [_rc("low", score=0.9), _rc("high", score=0.1)]
    # model scores: high should outscore low
    reranker = _make_reranker_with_mock_model([1.0, 10.0])  # low=1.0, high=10.0
    results = reranker.rerank("query", candidates, top_k=2)
    assert results[0].chunk.id == "high"
    assert results[1].chunk.id == "low"


def test_rerank_top_k_respected() -> None:
    candidates = [_rc(f"c{i}") for i in range(5)]
    reranker = _make_reranker_with_mock_model([float(i) for i in range(5)])
    results = reranker.rerank("query", candidates, top_k=2)
    assert len(results) == 2


def test_rerank_scores_normalized_to_0_1() -> None:
    candidates = [_rc("a"), _rc("b"), _rc("c")]
    reranker = _make_reranker_with_mock_model([100.0, -100.0, 0.0])
    results = reranker.rerank("query", candidates, top_k=3)
    for r in results:
        assert 0.0 <= r.score <= 1.0


def test_rerank_preserves_metadata() -> None:
    candidates = [_rc("meta_chunk")]
    candidates[0].chunk.metadata["key"] = "val"
    reranker = _make_reranker_with_mock_model([5.0])
    results = reranker.rerank("query", candidates, top_k=1)
    assert results[0].chunk.metadata["key"] == "val"


def test_rerank_empty_candidates_returns_empty() -> None:
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker._model_name = "mock"
    reranker._model = None
    results = reranker.rerank("query", [], top_k=5)
    assert results == []


# ── Lazy loading ─────────────────────────────────────────────────────────────

def test_model_loaded_lazily_on_first_rerank() -> None:
    """Model should not be loaded at __init__ — only on first rerank() call."""
    reranker = CrossEncoderReranker(model_name="cross-encoder/mock")
    assert reranker._model is None  # not loaded yet

    mock_model = MagicMock()
    mock_model.predict.return_value = [1.0]

    with patch(
        "src.retrieval.cross_encoder_reranker.CrossEncoderReranker._load_model",
        side_effect=lambda: setattr(reranker, "_model", mock_model),
    ):
        reranker.rerank("q", [_rc("x")], top_k=1)

    assert reranker._model is not None
