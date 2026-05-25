"""Unit tests for RAGOrchestrator using mock retriever and generator."""

from unittest.mock import MagicMock

import pytest

from src.orchestrator import RAGOrchestrator
from src.utils.exceptions import GenerationError, RetrievalError
from src.utils.models import Answer, Chunk, RetrievedChunk


def _make_chunk(text: str = "sample text") -> RetrievedChunk:
    return RetrievedChunk(chunk=Chunk(id="c1", text=text, metadata={}), score=0.9)


def _make_answer(text: str = "The answer.") -> Answer:
    return Answer(text=text, citations=[0], retrieved_chunks=[_make_chunk()])


@pytest.fixture
def mock_retriever() -> MagicMock:
    r = MagicMock()
    r.retrieve.return_value = [_make_chunk()]
    return r


@pytest.fixture
def mock_generator() -> MagicMock:
    g = MagicMock()
    g.generate.return_value = _make_answer()
    return g


@pytest.fixture
def orchestrator(mock_retriever: MagicMock, mock_generator: MagicMock) -> RAGOrchestrator:
    return RAGOrchestrator(retriever=mock_retriever, generator=mock_generator)


def test_answer_returns_answer_object(orchestrator: RAGOrchestrator) -> None:
    result = orchestrator.answer("What is the capital of France?")
    assert isinstance(result, Answer)


def test_answer_chains_retriever_then_generator(
    orchestrator: RAGOrchestrator,
    mock_retriever: MagicMock,
    mock_generator: MagicMock,
) -> None:
    orchestrator.answer("query")
    mock_retriever.retrieve.assert_called_once_with("query", top_k=5)
    mock_generator.generate.assert_called_once()


def test_answer_passes_chunks_to_generator(
    orchestrator: RAGOrchestrator,
    mock_retriever: MagicMock,
    mock_generator: MagicMock,
) -> None:
    chunks = [_make_chunk("chunk text")]
    mock_retriever.retrieve.return_value = chunks

    orchestrator.answer("query")
    call_args = mock_generator.generate.call_args
    assert call_args[0][1] == chunks or call_args[1].get("chunks") == chunks


def test_answer_raises_retrieval_error_on_retriever_failure(
    orchestrator: RAGOrchestrator,
    mock_retriever: MagicMock,
) -> None:
    mock_retriever.retrieve.side_effect = RetrievalError("vector store down")
    with pytest.raises(RetrievalError, match="vector store down"):
        orchestrator.answer("query")


def test_answer_raises_generation_error_on_generator_failure(
    orchestrator: RAGOrchestrator,
    mock_generator: MagicMock,
) -> None:
    mock_generator.generate.side_effect = GenerationError("LLM unavailable")
    with pytest.raises(GenerationError, match="LLM unavailable"):
        orchestrator.answer("query")


def test_answer_wraps_unexpected_retriever_exception(
    orchestrator: RAGOrchestrator,
    mock_retriever: MagicMock,
) -> None:
    mock_retriever.retrieve.side_effect = RuntimeError("unexpected")
    with pytest.raises(RetrievalError):
        orchestrator.answer("query")


def test_answer_wraps_unexpected_generator_exception(
    orchestrator: RAGOrchestrator,
    mock_generator: MagicMock,
) -> None:
    mock_generator.generate.side_effect = RuntimeError("unexpected")
    with pytest.raises(GenerationError):
        orchestrator.answer("query")


def test_orchestrator_calls_rewriter_when_history_provided():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []
    mock_generator = MagicMock()
    mock_generator.generate.return_value = MagicMock(text="answer", citations=[], chunks=[])
    mock_rewriter = MagicMock()
    mock_rewriter.rewrite.return_value = "rewritten query"

    orchestrator = RAGOrchestrator(
        retriever=mock_retriever,
        generator=mock_generator,
        query_rewriter=mock_rewriter,
    )

    history = [{"role": "user", "content": "prior question"}]
    orchestrator.answer("follow up", history=history)

    # Rewriter should be called
    mock_rewriter.rewrite.assert_called_once_with("follow up", history)
    # Retrieval should use rewritten query
    mock_retriever.retrieve.assert_called_once_with("rewritten query", top_k=5)
    # Generation uses rewritten query so pronouns are resolved for the generator
    mock_generator.generate.assert_called_once()
    assert mock_generator.generate.call_args.args[0] == "rewritten query"


def test_orchestrator_skips_rewriter_when_no_history():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []
    mock_generator = MagicMock()
    mock_generator.generate.return_value = MagicMock(text="answer", citations=[], chunks=[])
    mock_rewriter = MagicMock()

    orchestrator = RAGOrchestrator(
        retriever=mock_retriever,
        generator=mock_generator,
        query_rewriter=mock_rewriter,
    )

    orchestrator.answer("first question", history=None)

    # Rewriter should NOT be called
    mock_rewriter.rewrite.assert_not_called()
    # Retrieval should use original query
    mock_retriever.retrieve.assert_called_once_with("first question", top_k=5)
