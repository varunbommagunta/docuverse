"""Legal document chunker.

Splits legal documents (Constitutions, statutes, laws) at structural boundaries
(Articles, Sections) rather than character counts.

Based on analysis of the Indian Constitution PDF:
  - Em-dash separator (—) is the article title/body delimiter
  - Article pattern: r"(\\d{1,3}[A-Z]?).\\s+([A-Z][^—]{5,80})—"
  - PART roman numerals are hierarchical headers
  - Footnote blocks (after underscore dividers) should be excluded
  - TOC pages have many article titles without bodies — skip
"""

import re
import uuid
from dataclasses import dataclass, field

import structlog

from src.utils.models import Chunk

logger = structlog.get_logger(__name__)


@dataclass
class _ArticleMatch:
    """An Article match in the document text."""
    article_id: str
    title: str
    text_start: int
    part_name: str = ""
    chapter: str = ""


class LegalChunker:
    """Chunker for legal documents with structural awareness."""

    name = "legal"

    # Articles run together without newlines in many PDFs; no leading-newline anchor
    ARTICLE_PATTERN = re.compile(
        r"(\d{1,3}[A-Z]?)\.\s+([A-Z][^—\n]{5,100})—",
        re.MULTILINE,
    )

    PART_PATTERN = re.compile(
        r"PART\s+([IVXLC]+)(?:\.|\s)",
        re.IGNORECASE,
    )

    FOOTNOTE_SEP = re.compile(r"_{10,}")

    # Do NOT consume trailing digits — page numbers concatenate directly with article
    # numbers (e.g. "175312." where 175 is the page and 312 is the article).
    PAGE_HEADER = re.compile(
        r"THE CONSTITUTION OF\s*INDIA\s*\([^)]*\)",
        re.IGNORECASE,
    )

    def __init__(self, max_chunk_size: int = 5000) -> None:
        self._max_chunk_size = max_chunk_size

    def chunk(self, document_text: str, document_metadata: dict | None = None) -> list[Chunk]:
        """Split legal document into Article-level chunks."""
        if document_metadata is None:
            document_metadata = {}

        cleaned_text = self._preprocess(document_text)

        part_positions = [
            (m.start(), m.group(1))
            for m in self.PART_PATTERN.finditer(cleaned_text)
        ]

        article_matches = []
        for m in self.ARTICLE_PATTERN.finditer(cleaned_text):
            current_part = ""
            for pos, roman in reversed(part_positions):
                if pos < m.start():
                    current_part = f"PART {roman}"
                    break

            article_matches.append(_ArticleMatch(
                article_id=m.group(1),
                title=m.group(2).strip(),
                text_start=m.start(),
                part_name=current_part,
            ))

        if not article_matches:
            logger.warning("legal_chunker_no_articles_detected, returning single chunk")
            return [Chunk(
                id=str(uuid.uuid4()),
                text=cleaned_text[:self._max_chunk_size],
                metadata={**document_metadata, "chunker": self.name, "warning": "no articles detected"},
            )]

        logger.info("legal_chunker_found_articles", count=len(article_matches))

        chunks = []
        for i, art in enumerate(article_matches):
            text_end = article_matches[i + 1].text_start if i + 1 < len(article_matches) else len(cleaned_text)
            article_text = cleaned_text[art.text_start:text_end].strip()

            if len(article_text) < 100:
                continue

            if len(article_text) > self._max_chunk_size:
                sub_chunks = self._sub_chunk_long_article(article_text, art)
                for j, sub_text in enumerate(sub_chunks):
                    chunks.append(Chunk(
                        id=str(uuid.uuid4()),
                        text=sub_text,
                        metadata={
                            **document_metadata,
                            "chunker": self.name,
                            "article_id": art.article_id,
                            "article_title": art.title,
                            "part": art.part_name,
                            "sub_chunk_index": j,
                            "sub_chunk_total": len(sub_chunks),
                        },
                    ))
            else:
                chunks.append(Chunk(
                    id=str(uuid.uuid4()),
                    text=article_text,
                    metadata={
                        **document_metadata,
                        "chunker": self.name,
                        "article_id": art.article_id,
                        "article_title": art.title,
                        "part": art.part_name,
                    },
                ))

        logger.info("legal_chunker_produced_chunks", count=len(chunks))
        return chunks

    def _preprocess(self, text: str) -> str:
        """Strip running page headers. Footnotes are left in place — they contain
        no em-dashes so the article pattern won't match them as article headers."""
        return self.PAGE_HEADER.sub("", text)

    def _sub_chunk_long_article(self, article_text: str, article: _ArticleMatch) -> list[str]:
        """For very long Articles, sub-chunk at numeric clause boundaries (1)(2)(3)."""
        clause_pattern = re.compile(r"\(\d{1,2}\)")
        positions = [m.start() for m in clause_pattern.finditer(article_text)]

        if len(positions) < 2:
            return [article_text]

        chunks = []
        current_chunk_start = 0
        current_chunk_size = 0

        for pos in positions:
            if current_chunk_size > self._max_chunk_size // 2:
                chunks.append(article_text[current_chunk_start:pos].strip())
                current_chunk_start = pos
                current_chunk_size = 0
            current_chunk_size = pos - current_chunk_start

        chunks.append(article_text[current_chunk_start:].strip())
        return chunks
