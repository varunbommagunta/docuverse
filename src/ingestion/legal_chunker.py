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

    @staticmethod
    def _preprocess_legal_text(text: str) -> str:
        """Strip PDF extraction artifacts specific to Indian legal documents.

        Four pattern classes found in pypdf output of the Indian Constitution:

        1. Page headers inserted by pypdf at the top of each page.
        2. Horizontal rule separators and inline footnote content on the same line.
           NOTE: the originally specified pattern used DOTALL with a \\n\\d+\\. lookahead,
           but this PDF has articles running together without newlines so DOTALL
           consumed entire article blocks. _{20,} (not _{4,}) avoids the 10–12 char
           TOC decorative underlines; no DOTALL keeps matching within a single line.
        3. Inline footnote reference digits before [ (e.g. 1[(4) Nothing...])
           — strip only the leading digit, keep the bracket content.
        4. Orphaned page numbers — standalone digit lines between article text.
        """
        # 1. Page headers
        text = re.sub(r'^\d+\s+THE CONSTITUTION OF INDIA\s*\n', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\(Part [A-Z]+\.—[^\)]+\)\s*\n', '', text, flags=re.MULTILINE)
        # 2. Horizontal rule + same-line footnote content
        text = re.sub(r'_{20,}[^\n]*', '', text)
        # 3. Inline footnote reference digits before [ — strip digit, keep bracket content
        text = re.sub(r'(?<!\d)\d\[', '[', text)
        # 4. Orphaned page numbers — standalone digit lines
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        return text

    @staticmethod
    def _clean_text(text: str) -> str:
        """Strip constitutional amendment markers and footnote artifacts.

        The raw PDF text encodes constitutional amendments with superscript
        notation: N[inserted text] for insertions and ]N[ for transitions
        between amendments. Footnote blocks appear below a line of underscores
        and start with 'N. Subs./Ins./Rep./Omitted'.
        """
        # Transition markers ]N[ → single space
        text = re.sub(r"\]\d+\[", " ", text)
        # Opening amendment markers N[ → nothing (keep the inserted content)
        text = re.sub(r"\d+\[", "", text)
        # Closing brackets left by stripped opening markers
        text = re.sub(r"\]", "", text)
        # Orphaned opening brackets (no matching ] remains after step above)
        text = re.sub(r"\[", "", text)
        # Footnote separator lines (10+ underscores)
        text = re.sub(r"_{10,}", "", text)
        # Footnote label lines: digit + period + amendment keyword
        text = re.sub(
            r"^\d+\.\s+(?:Subs|Ins|Rep|Omitted)\..*$",
            "",
            text,
            flags=re.MULTILINE | re.IGNORECASE,
        )
        # Collapse runs of blank lines and spaces created by the removals
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"  +", " ", text)
        return text.strip()

    def _preprocess(self, text: str) -> str:
        """Strip PDF extraction artifacts, then amendment markers."""
        text = self._preprocess_legal_text(text)
        text = self.PAGE_HEADER.sub("", text)
        return self._clean_text(text)

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
