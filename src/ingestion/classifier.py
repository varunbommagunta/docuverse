"""Document type classification.

Two-stage classification:
  1. Rule-based: file extension + first 500 chars (free, instant)
  2. LLM fallback: gpt-4o-mini on first 2000 chars (~₹0.10) for ambiguous docs

Document types supported initially:
  "legal":   laws, constitutions, statutes, contracts
  "default": everything else (uses RecursiveChunker fallback)

Future doc types (placeholder for later sessions):
  "prose", "academic", "technical"
"""

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import structlog
from openai import OpenAI

logger = structlog.get_logger(__name__)


class DocumentType(str, Enum):
    LEGAL = "legal"
    PROSE = "prose"
    ACADEMIC = "academic"
    TECHNICAL = "technical"
    DEFAULT = "default"


@dataclass
class ClassificationResult:
    doc_type: DocumentType
    confidence: float
    method: str
    reason: str = ""


class RuleBasedClassifier:
    """First-pass classifier using file extension + structural heuristics."""

    LEGAL_PATTERNS = [
        r"\bArticle\s+\d{1,3}",
        r"\bSection\s+\d{1,3}",
        r"\bSchedule\s+[IVXLC]+",
        r"\bPART\s+[IVXLC]+",
        r"\bCONSTITUTION\b",
        r"\bAct,?\s+\d{4}",
        r"\bAmendment\b.*\bAct\b",
        r"\bclause\s+\(\d+\)",
        r"\bsub-?section\s+\(\d+\)",
    ]

    def classify(self, text_sample: str, file_path: Path | None = None) -> ClassificationResult:
        """Classify document from first ~500 chars sample.

        Returns ClassificationResult with confidence.
        High confidence (>=0.8) → trust the rule-based result.
        Low confidence (<0.8) → escalate to LLM classifier.
        """
        sample = text_sample[:2000]

        legal_score = sum(
            1 for pattern in self.LEGAL_PATTERNS
            if re.search(pattern, sample, re.IGNORECASE)
        )

        if legal_score >= 3:
            return ClassificationResult(
                doc_type=DocumentType.LEGAL,
                confidence=0.9,
                method="rules",
                reason=f"Matched {legal_score} legal patterns",
            )

        if legal_score >= 1:
            return ClassificationResult(
                doc_type=DocumentType.LEGAL,
                confidence=0.6,
                method="rules",
                reason=f"Matched {legal_score} legal patterns (uncertain)",
            )

        return ClassificationResult(
            doc_type=DocumentType.DEFAULT,
            confidence=0.3,
            method="rules",
            reason="No clear structural patterns; defaulting",
        )


class LLMClassifier:
    """Second-pass classifier using gpt-4o-mini on first ~2000 chars."""

    SYSTEM_PROMPT = """You are a document classifier. Given the first 2000 characters of a document, classify it into one of these types:

"legal": laws, constitutions, statutes, regulations, contracts (highly structured with articles/sections)
"prose": general articles, books, essays, blog posts, narrative text
"academic": research papers (has abstract, sections, references)
"technical": API docs, manuals, technical guides, how-tos
"default": none of the above clearly applies

Respond with ONLY a JSON object: {"doc_type": "...", "confidence": 0.0-1.0, "reason": "..."}"""

    def __init__(self, openai_client: OpenAI) -> None:
        self._client = openai_client

    def classify(self, text_sample: str) -> ClassificationResult:
        """Classify by reading the first ~2000 chars with an LLM."""
        sample = text_sample[:2000]

        try:
            response = self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": f"Document start:\n\n{sample}"},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=200,
            )

            result = json.loads(response.choices[0].message.content)

            doc_type_str = result.get("doc_type", "default")
            try:
                doc_type = DocumentType(doc_type_str)
            except ValueError:
                doc_type = DocumentType.DEFAULT

            return ClassificationResult(
                doc_type=doc_type,
                confidence=float(result.get("confidence", 0.5)),
                method="llm",
                reason=result.get("reason", ""),
            )

        except Exception as exc:
            logger.error("LLM classifier failed, falling back to DEFAULT", error=str(exc))
            return ClassificationResult(
                doc_type=DocumentType.DEFAULT,
                confidence=0.0,
                method="llm_failed",
                reason=f"LLM error: {exc}",
            )


class DocumentClassifier:
    """Combined classifier: rule-based first, LLM fallback if uncertain."""

    CONFIDENCE_THRESHOLD = 0.8

    def __init__(self, llm_classifier: LLMClassifier | None = None) -> None:
        self._rule_classifier = RuleBasedClassifier()
        self._llm_classifier = llm_classifier

    def classify(self, text_sample: str, file_path: Path | None = None) -> ClassificationResult:
        result = self._rule_classifier.classify(text_sample, file_path)

        logger.info(
            "classification_rule_pass",
            doc_type=result.doc_type,
            confidence=result.confidence,
            reason=result.reason,
        )

        if result.confidence >= self.CONFIDENCE_THRESHOLD:
            return result

        if self._llm_classifier is None:
            logger.warning("No LLM classifier available; using uncertain rule-based result")
            return result

        logger.info("classification_escalating_to_llm", rule_confidence=result.confidence)
        llm_result = self._llm_classifier.classify(text_sample)

        logger.info(
            "classification_llm_pass",
            doc_type=llm_result.doc_type,
            confidence=llm_result.confidence,
            reason=llm_result.reason,
        )

        return llm_result
