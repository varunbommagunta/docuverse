"""Unit tests for document classifier."""

import pytest
from unittest.mock import MagicMock
from pathlib import Path

from src.ingestion.classifier import (
    RuleBasedClassifier,
    LLMClassifier,
    DocumentClassifier,
    DocumentType,
    ClassificationResult,
)


def test_rule_classifier_detects_legal_with_strong_signal():
    classifier = RuleBasedClassifier()
    sample = """
PART II CITIZENSHIP
Article 5. Citizenship at the commencement of the Constitution.
Section 312 covers All-India services.
Schedule VII contains the Union List.
"""
    result = classifier.classify(sample)
    assert result.doc_type == DocumentType.LEGAL
    assert result.confidence >= 0.8


def test_rule_classifier_returns_uncertain_for_prose():
    classifier = RuleBasedClassifier()
    sample = """
Once upon a time in a faraway land, there lived a king who loved
his kingdom dearly. He spent his days walking in the gardens.
"""
    result = classifier.classify(sample)
    assert result.confidence < 0.8


def test_rule_classifier_moderate_legal_signal():
    classifier = RuleBasedClassifier()
    sample = "Article 5. Some content here without other markers."
    result = classifier.classify(sample)
    assert result.doc_type == DocumentType.LEGAL
    assert result.confidence == 0.6


def test_llm_classifier_with_mock(monkeypatch):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = '{"doc_type": "academic", "confidence": 0.9, "reason": "has abstract"}'
    mock_client.chat.completions.create.return_value = mock_response

    classifier = LLMClassifier(mock_client)
    result = classifier.classify("Abstract: This paper examines...")

    assert result.doc_type == DocumentType.ACADEMIC
    assert result.confidence == 0.9
    assert result.method == "llm"


def test_llm_classifier_handles_unknown_doc_type(monkeypatch):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = '{"doc_type": "unknown_type", "confidence": 0.5, "reason": "unclear"}'
    mock_client.chat.completions.create.return_value = mock_response

    classifier = LLMClassifier(mock_client)
    result = classifier.classify("Some text")

    assert result.doc_type == DocumentType.DEFAULT


def test_llm_classifier_handles_exception():
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("API error")

    classifier = LLMClassifier(mock_client)
    result = classifier.classify("Some text")

    assert result.doc_type == DocumentType.DEFAULT
    assert result.method == "llm_failed"


def test_combined_classifier_uses_rules_when_confident():
    """If rule-based is confident, don't escalate to LLM."""
    mock_llm = MagicMock()
    classifier = DocumentClassifier(llm_classifier=mock_llm)

    sample = "Article 1. Section 2. PART III. Constitution."
    result = classifier.classify(sample)

    mock_llm.classify.assert_not_called()


def test_combined_classifier_escalates_when_uncertain():
    """If rule-based is uncertain, escalate to LLM."""
    mock_llm = MagicMock()
    mock_llm.classify.return_value = ClassificationResult(
        doc_type=DocumentType.PROSE, confidence=0.8, method="llm"
    )

    classifier = DocumentClassifier(llm_classifier=mock_llm)
    result = classifier.classify("Random prose text without legal markers.")

    mock_llm.classify.assert_called_once()
    assert result.doc_type == DocumentType.PROSE


def test_combined_classifier_no_llm_returns_rule_result():
    """Without LLM fallback, return rule-based result even if uncertain."""
    classifier = DocumentClassifier(llm_classifier=None)
    result = classifier.classify("Random text with no patterns.")

    assert result.method == "rules"
    assert result.doc_type == DocumentType.DEFAULT


# ── New type detection ────────────────────────────────────────────────────────

def test_rule_classifier_detects_academic_with_strong_signal():
    classifier = RuleBasedClassifier()
    sample = """
Abstract

This paper introduces a novel approach to document retrieval.

Introduction

Information retrieval has been studied for decades. Table 1 shows baseline results.

Methodology

We trained a bi-encoder model. Results are reported in Figure 2. See also Smith et al.

References

1. Karpukhin et al. (2020). DPR. ACL.
"""
    result = classifier.classify(sample)
    assert result.doc_type == DocumentType.ACADEMIC
    assert result.confidence >= 0.8


def test_rule_classifier_detects_technical_with_strong_signal():
    classifier = RuleBasedClassifier()
    sample = """
## Installation

Prerequisites: Python 3.10+

```bash
pip install docuverse
```

## Configuration

Set the API key in your environment. Usage:

```python
from docuverse import Pipeline
```

Parameters: api_key (str), model (str). Returns a Pipeline instance.
"""
    result = classifier.classify(sample)
    assert result.doc_type == DocumentType.TECHNICAL
    assert result.confidence >= 0.8


def test_rule_classifier_legal_beats_academic_when_stronger():
    """Legal patterns take priority when they score higher."""
    classifier = RuleBasedClassifier()
    # Strong legal signal + some academic-like words
    sample = """
PART II CITIZENSHIP
Article 5. Citizenship at the commencement of the Constitution.
Section 312. All-India services.
Schedule VII. Union List.
Amendment Act, 1976.
Introduction to the amendments.
"""
    result = classifier.classify(sample)
    assert result.doc_type == DocumentType.LEGAL
