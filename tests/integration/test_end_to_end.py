"""End-to-end integration test — requires a real OpenAI API key.

Skipped by default. Enable by setting RUN_INTEGRATION_TESTS=1 in environment.

Usage:
    RUN_INTEGRATION_TESTS=1 pytest tests/integration/ -v
"""

import os

import pytest

RUN = bool(os.getenv("RUN_INTEGRATION_TESTS"))


@pytest.mark.skipif(not RUN, reason="Set RUN_INTEGRATION_TESTS=1 to enable")
def test_ingest_and_query_solar_system(sample_pdf_path: str) -> None:
    """Ingest the sample Solar System PDF and verify an answerable question gets citations."""
    from src.factory import get_rag_components

    orchestrator, pipeline = get_rag_components()

    # Ingest the sample PDF
    result = pipeline.ingest(sample_pdf_path, filename="sample.pdf")
    assert result["chunk_count"] > 0, "Expected at least one chunk"

    # Ask a question whose answer is in the PDF
    answer = orchestrator.answer("What is the largest planet in the Solar System?")

    assert "Jupiter" in answer.text, f"Expected Jupiter in answer, got: {answer.text}"
    assert len(answer.citations) > 0, "Expected at least one citation"
    assert len(answer.retrieved_chunks) > 0, "Expected retrieved chunks in answer"
