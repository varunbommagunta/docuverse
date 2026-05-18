"""Unit tests for OpenAIGenerator using a mocked OpenAI client."""

from unittest.mock import MagicMock, patch

from src.generation.openai_generator import OpenAIGenerator
from src.utils.models import Answer, Chunk, RetrievedChunk


def _make_chunk(text: str = "The capital of France is Paris.", idx: int = 0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(id=f"chunk-{idx}", text=text, metadata={"filename": "test.pdf"}),
        score=0.9 - idx * 0.1,
    )


def _make_generator_with_mock(mock_response_text: str) -> tuple[OpenAIGenerator, MagicMock]:
    """Construct an OpenAIGenerator with a mocked OpenAI client."""
    with patch("src.generation.openai_generator.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.content = mock_response_text
        mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

        generator = OpenAIGenerator(model="gpt-4o-mini")
        generator._client = mock_client
        return generator, mock_client


def test_generate_returns_answer() -> None:
    generator, mock_client = _make_generator_with_mock("Paris is the capital [chunk_0].")
    mock_choice = MagicMock()
    mock_choice.message.content = "Paris is the capital [chunk_0]."
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    result = generator.generate("What is the capital of France?", [_make_chunk()])
    assert isinstance(result, Answer)


def test_generate_parses_single_citation() -> None:
    generator, mock_client = _make_generator_with_mock("")
    mock_choice = MagicMock()
    mock_choice.message.content = "Paris is the capital [chunk_0]."
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    result = generator.generate("Capital?", [_make_chunk()])
    assert 0 in result.citations


def test_generate_parses_multiple_citations() -> None:
    generator, mock_client = _make_generator_with_mock("")
    mock_choice = MagicMock()
    mock_choice.message.content = "Text [chunk_0] and more [chunk_2]."
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    chunks = [_make_chunk(idx=i) for i in range(3)]
    result = generator.generate("Question?", chunks)
    assert result.citations == [0, 2]


def test_generate_deduplicates_citations() -> None:
    generator, mock_client = _make_generator_with_mock("")
    mock_choice = MagicMock()
    mock_choice.message.content = "Text [chunk_0] and again [chunk_0]."
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    result = generator.generate("Question?", [_make_chunk()])
    assert result.citations.count(0) == 1


def test_generate_no_citations_when_cannot_answer() -> None:
    generator, mock_client = _make_generator_with_mock("")
    mock_choice = MagicMock()
    mock_choice.message.content = "I cannot answer this from the provided documents."
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    result = generator.generate("Unknown?", [_make_chunk()])
    assert result.citations == []


def test_generate_attaches_retrieved_chunks() -> None:
    generator, mock_client = _make_generator_with_mock("")
    mock_choice = MagicMock()
    mock_choice.message.content = "Answer [chunk_0]."
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    chunks = [_make_chunk(idx=0), _make_chunk(idx=1)]
    result = generator.generate("Question?", chunks)
    assert len(result.retrieved_chunks) == 2


def test_generate_retries_on_rate_limit_error() -> None:
    from openai import RateLimitError

    generator, mock_client = _make_generator_with_mock("")
    success_choice = MagicMock()
    success_choice.message.content = "Answer [chunk_0]."

    # Fail once with RateLimitError, then succeed
    rate_limit_response = MagicMock(status_code=429, headers={}, body=b"rate limited")
    mock_client.chat.completions.create.side_effect = [
        RateLimitError("rate limited", response=rate_limit_response, body=None),
        MagicMock(choices=[success_choice]),
    ]

    result = generator.generate("Question?", [_make_chunk()])
    assert mock_client.chat.completions.create.call_count == 2
    assert result.text == "Answer [chunk_0]."
