"""Document type classification.

Two-stage classification:
  1. Rule-based: structural heuristics on first 2000 chars (free, instant)
  2. LLM fallback: gpt-4o-mini on first 2000 chars for ambiguous docs

Document types:
  "legal":    laws, constitutions, statutes, contracts
  "academic": research papers, reports, ARC-style government documents
  "technical":API docs, manuals, technical specifications
  "prose":    novels, essays, articles, blog posts
  "default":  fallback (RecursiveChunker)
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
    """First-pass classifier using structural heuristics on the first 2000 chars."""

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

    ACADEMIC_PATTERNS = [
        r"\bAbstract\b",
        r"\bIntroduction\b",
        r"\bMethodolog",
        r"\bConclusion\b",
        r"\bReferences\b",
        r"\bKeywords?\b",
        r"^\d+\.\d+\s+[A-Z]",
        r"\bFigure\s+\d+\b",
        r"\bTable\s+\d+\b",
        r"\bet al\.\b",
        r"\bdoi:\s*10\.",
    ]

    TECHNICAL_PATTERNS = [
        r"^#{1,3}\s+\w",
        r"```",
        r"\bInstallation\b",
        r"\bPrerequisite",
        r"\bAPI\b",
        r"^Step\s+\d+",
        r"\bSyntax\b",
        r"\bParameter",
        r"\bReturns?\b",
        r"\bConfiguration\b",
        r"\bUsage\b",
    ]

    PROSE_PATTERNS = [
        r"\bchapter\s+\d+\b",
        r"\bonce upon\b",
        r"[“”‘’]\w",
        r"\bnarrat(?:or|ive|ed)\b",
        r"\bsaid\b.{0,20}[,.]",
        r"\bnovel\b|\bstory\b|\bessay\b",
        r"\bprotagonist\b|\bcharacter\b",
        r"\bparagraph\b",
    ]

    # (min_matches_for_high_confidence, confidence_value)
    _THRESHOLDS: dict = {
        DocumentType.LEGAL:     (3, 0.90),
        DocumentType.ACADEMIC:  (3, 0.85),
        DocumentType.TECHNICAL: (3, 0.85),
        DocumentType.PROSE:     (3, 0.80),
    }
    # Tie-breaking priority when two types score equally
    _PRIORITY = [DocumentType.LEGAL, DocumentType.ACADEMIC, DocumentType.TECHNICAL, DocumentType.PROSE]

    def classify(self, text_sample: str, file_path: Path | None = None) -> ClassificationResult:
        """Classify document from first 2000 chars using pattern matching.

        Returns ClassificationResult with confidence.
        High confidence (>=0.8) → trust the rule-based result.
        Low confidence (<0.8) → escalate to LLM classifier.
        """
        sample = text_sample[:2000]
        flags = re.IGNORECASE | re.MULTILINE

        scores = {
            DocumentType.LEGAL:     sum(1 for p in self.LEGAL_PATTERNS     if re.search(p, sample, flags)),
            DocumentType.ACADEMIC:  sum(1 for p in self.ACADEMIC_PATTERNS  if re.search(p, sample, flags)),
            DocumentType.TECHNICAL: sum(1 for p in self.TECHNICAL_PATTERNS if re.search(p, sample, flags)),
            DocumentType.PROSE:     sum(1 for p in self.PROSE_PATTERNS     if re.search(p, sample, flags)),
        }

        # Collect types that meet their threshold, rank by score then priority
        qualified = [
            (dt, scores[dt], conf)
            for dt, (threshold, conf) in self._THRESHOLDS.items()
            if scores[dt] >= threshold
        ]
        if qualified:
            qualified.sort(key=lambda x: (-x[1], self._PRIORITY.index(x[0])))
            best_type, best_score, best_conf = qualified[0]
            return ClassificationResult(
                doc_type=best_type,
                confidence=best_conf,
                method="rules",
                reason=f"Matched {best_score} {best_type.value} patterns",
            )

        # Weak legal signal (1–2 matches) — still worth escalating to LLM
        if scores[DocumentType.LEGAL] >= 1:
            return ClassificationResult(
                doc_type=DocumentType.LEGAL,
                confidence=0.6,
                method="rules",
                reason=f"Matched {scores[DocumentType.LEGAL]} legal patterns (uncertain)",
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
