"""Unit tests for ChunkerRouter."""

from unittest.mock import MagicMock

from src.ingestion.classifier import (
    DocumentClassifier,
    ClassificationResult,
    DocumentType,
)
from src.ingestion.router import ChunkerRouter


def test_router_dispatches_legal_to_legal_chunker():
    classifier = DocumentClassifier(llm_classifier=None)
    router = ChunkerRouter(classifier)

    # Strong legal signal: 3+ patterns
    text = "Article 5. PART I. Section 1. Constitution of India. Amendment Act, 1976."
    chunks, classification = router.route_and_chunk(text, {"source": "test.pdf"})

    assert classification.doc_type == DocumentType.LEGAL
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.metadata.get("document_type") == "legal"


def test_router_falls_back_to_default_for_unclear_docs():
    classifier = DocumentClassifier(llm_classifier=None)
    router = ChunkerRouter(classifier)

    text = "Random text with no patterns whatsoever."
    chunks, classification = router.route_and_chunk(text, {"source": "test.txt"})

    assert classification.doc_type == DocumentType.DEFAULT
    assert len(chunks) > 0


def test_router_enriches_chunks_with_classification_metadata():
    classifier = DocumentClassifier(llm_classifier=None)
    router = ChunkerRouter(classifier)

    text = "Random text with no patterns whatsoever."
    chunks, classification = router.route_and_chunk(text, {"source": "test.txt"})

    for chunk in chunks:
        assert "document_type" in chunk.metadata
        assert "classification_confidence" in chunk.metadata
        assert "classification_method" in chunk.metadata


def test_router_returns_classification_result():
    classifier = DocumentClassifier(llm_classifier=None)
    router = ChunkerRouter(classifier)

    text = "Random text."
    chunks, classification = router.route_and_chunk(text)

    assert isinstance(classification, ClassificationResult)
    assert classification.method in ("rules", "llm", "llm_failed")


def test_router_uses_llm_when_uncertain():
    mock_llm = MagicMock()
    mock_llm.classify.return_value = ClassificationResult(
        doc_type=DocumentType.LEGAL,
        confidence=0.85,
        method="llm",
    )
    classifier = DocumentClassifier(llm_classifier=mock_llm)
    router = ChunkerRouter(classifier)

    text = "Random text with no patterns whatsoever."
    chunks, classification = router.route_and_chunk(text, {"source": "mystery.pdf"})

    mock_llm.classify.assert_called_once()
    assert classification.doc_type == DocumentType.LEGAL
