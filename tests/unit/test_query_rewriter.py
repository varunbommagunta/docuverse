"""Unit tests for OpenAIQueryRewriter."""

import pytest
from unittest.mock import MagicMock, patch

from src.retrieval.query_rewriter import OpenAIQueryRewriter


def test_rewriter_returns_query_unchanged_with_no_history():
    rewriter = OpenAIQueryRewriter(api_key="sk-test")
    result = rewriter.rewrite("What is Article 21?", history=None)
    assert result == "What is Article 21?"


def test_rewriter_returns_query_unchanged_with_empty_history():
    rewriter = OpenAIQueryRewriter(api_key="sk-test")
    result = rewriter.rewrite("What is Article 21?", history=[])
    assert result == "What is Article 21?"


@patch("src.retrieval.query_rewriter.OpenAI")
def test_rewriter_calls_openai_with_history(mock_openai_class):
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "tell me more about Article 21 of the Indian Constitution"
    mock_client.chat.completions.create.return_value = mock_response

    rewriter = OpenAIQueryRewriter(api_key="sk-test")
    history = [
        {"role": "user", "content": "What is Article 21?"},
        {"role": "assistant", "content": "Article 21 protects life and liberty."},
    ]
    result = rewriter.rewrite("tell me more about that", history=history)

    assert result == "tell me more about Article 21 of the Indian Constitution"
    mock_client.chat.completions.create.assert_called_once()


@patch("src.retrieval.query_rewriter.OpenAI")
def test_rewriter_falls_back_on_exception(mock_openai_class):
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.chat.completions.create.side_effect = Exception("API error")
    rewriter = OpenAIQueryRewriter(api_key="sk-test")
    history = [{"role": "user", "content": "test"}]

    result = rewriter.rewrite("follow-up question", history=history)
    assert result == "follow-up question"  # falls back to original


@patch("src.retrieval.query_rewriter.OpenAI")
def test_rewriter_falls_back_on_empty_response(mock_openai_class):
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_response = MagicMock()
    mock_response.choices[0].message.content = ""
    mock_client.chat.completions.create.return_value = mock_response
    rewriter = OpenAIQueryRewriter(api_key="sk-test")
    history = [{"role": "user", "content": "test"}]

    result = rewriter.rewrite("follow-up question", history=history)
    assert result == "follow-up question"


@patch("src.retrieval.query_rewriter.OpenAI")
def test_rewriter_trims_long_history(mock_openai_class):
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "rewritten query"
    mock_client.chat.completions.create.return_value = mock_response
    rewriter = OpenAIQueryRewriter(api_key="sk-test", max_history_turns=2)

    # 10 messages = 5 turns
    history = [
        {"role": "user", "content": f"q{i}"} if i % 2 == 0 else {"role": "assistant", "content": f"a{i}"}
        for i in range(10)
    ]

    rewriter.rewrite("new question", history=history)

    # Check the user message sent to OpenAI mentions only last 2 turns (last 4 messages)
    call_args = mock_client.chat.completions.create.call_args
    user_message = call_args.kwargs["messages"][1]["content"]

    # Last 4 messages should be in the prompt
    assert "q6" in user_message
    assert "a7" in user_message
    # First messages should NOT be in the prompt
    assert "q0" not in user_message
