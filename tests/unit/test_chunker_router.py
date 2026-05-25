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


# ── New chunker dispatch tests ────────────────────────────────────────────────

def _forced_classifier(doc_type: DocumentType) -> DocumentClassifier:
    """Classifier backed by a mock LLM that always returns the given doc_type."""
    mock_llm = MagicMock()
    mock_llm.classify.return_value = ClassificationResult(
        doc_type=doc_type, confidence=0.85, method="llm"
    )
    return DocumentClassifier(llm_classifier=mock_llm)


def test_router_dispatches_prose_to_prose_chunker():
    router = ChunkerRouter(_forced_classifier(DocumentType.PROSE))
    chunks, classification = router.route_and_chunk(
        "She walked along the shore.\n\nThe waves rolled in slowly.",
        {"source": "novel.pdf"},
    )
    assert classification.doc_type == DocumentType.PROSE
    assert all(c.metadata.get("chunker") == "prose" for c in chunks)


def test_router_dispatches_academic_to_academic_chunker():
    router = ChunkerRouter(_forced_classifier(DocumentType.ACADEMIC))
    text = "ABSTRACT\nThis paper studies retrieval.\n\n1. Introduction\nBackground text."
    chunks, classification = router.route_and_chunk(text, {"source": "paper.pdf"})
    assert classification.doc_type == DocumentType.ACADEMIC
    assert all(c.metadata.get("chunker") == "academic" for c in chunks)


def test_router_dispatches_technical_to_technical_chunker():
    router = ChunkerRouter(_forced_classifier(DocumentType.TECHNICAL))
    text = "## Installation\nRun pip install.\n\n## Usage\nImport the library."
    chunks, classification = router.route_and_chunk(text, {"source": "manual.pdf"})
    assert classification.doc_type == DocumentType.TECHNICAL
    assert all(c.metadata.get("chunker") == "technical" for c in chunks)


def test_router_rule_based_academic_detection_end_to_end():
    classifier = DocumentClassifier(llm_classifier=None)
    router = ChunkerRouter(classifier)
    text = (
        "Abstract\n"
        "This study examines hybrid retrieval.\n\n"
        "Introduction\n"
        "Prior work used BM25 (et al.). Table 1 shows results.\n\n"
        "Methodology\n"
        "We trained on MS-MARCO. Figure 2 shows architecture.\n\n"
        "Conclusion\n"
        "Hybrid systems outperform baselines.\n\n"
        "References\n"
        "1. Robertson (2009). BM25. doi: 10.1561/1500000019"
    )
    chunks, classification = router.route_and_chunk(text, {"source": "study.pdf"})
    assert classification.doc_type == DocumentType.ACADEMIC
    assert len(chunks) > 0


def test_router_rule_based_technical_detection_end_to_end():
    classifier = DocumentClassifier(llm_classifier=None)
    router = ChunkerRouter(classifier)
    text = (
        "## Installation\n"
        "Prerequisites: Python 3.10+\n\n"
        "```bash\npip install pkg\n```\n\n"
        "## Configuration\n"
        "Set the API key. Parameters: key, value. Returns a config object.\n\n"
        "## Usage\n"
        "Basic usage example shown below."
    )
    chunks, classification = router.route_and_chunk(text, {"source": "docs.md"})
    assert classification.doc_type == DocumentType.TECHNICAL
    assert len(chunks) > 0
