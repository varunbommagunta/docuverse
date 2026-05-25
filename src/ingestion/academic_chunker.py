"""Academic document chunker for research papers and ARC-style reports.

Splits at section headings: numbered sections (1., 1.1), Chapter/Section
keywords, and ALL CAPS headings. Each heading + its body forms one chunk.
Sections exceeding the hard cap are sub-chunked by paragraph.
"""

import re
import uuid
from dataclasses import dataclass

import structlog

from src.utils.models import Chunk

logger = structlog.get_logger(__name__)


@dataclass
class _Section:
    title: str
    number: str
    start: int


class AcademicChunker:
    """Chunker for academic documents: section-heading-based splitting."""

    name = "academic"

    # Matches (in order of group capture):
    #   Group 1+2: "1.2 Title text"  or  "1. Title text"
    #   Group 3+4: "Chapter 2" or "Chapter 2: Title"  /  "Section 3 — Title"
    #   Group 5:   "ALL CAPS HEADING" (≥5 consecutive uppercase letters/spaces)
    HEADING_PATTERN = re.compile(
        r'^(?:'
        r'(\d+(?:\.\d+)*)\.\s+([A-Z][^\n]{3,80})'
        r'|(?:Chapter|Section)\s+(\d+)(?:[:\-\s]+([^\n]{3,80}))?'
        r'|([A-Z][A-Z\s\-]{4,60}(?:[A-Z]))'
        r')$',
        re.MULTILINE,
    )

    def __init__(self, max_chunk_size: int = 1000, hard_cap: int = 1500) -> None:
        self._max = max_chunk_size
        self._hard_cap = hard_cap

    def chunk(self, document_text: str, document_metadata: dict | None = None) -> list[Chunk]:
        """Split academic document at section headings.

        Args:
            document_text: Full document text.
            document_metadata: Passed through to each chunk's metadata.

        Returns:
            List of Chunk objects with section_title and section_number metadata.
        """
        metadata = document_metadata or {}
        sections = _find_sections(self.HEADING_PATTERN, document_text)

        if not sections:
            logger.warning("academic_chunker_no_sections_detected")
            return [Chunk(
                id=str(uuid.uuid4()),
                text=document_text[:self._hard_cap],
                metadata={**metadata, "chunker": self.name, "section_title": "", "section_number": "", "document_type": "academic"},
            )]

        logger.info("academic_chunker_found_sections", count=len(sections))
        chunks: list[Chunk] = []

        for i, sec in enumerate(sections):
            end = sections[i + 1].start if i + 1 < len(sections) else len(document_text)
            section_text = document_text[sec.start:end].strip()

            if len(section_text) <= self._hard_cap:
                chunks.append(_make_chunk(section_text, sec, metadata))
            else:
                for sub_text in _sub_chunk_by_paragraph(section_text, self._max):
                    chunks.append(_make_chunk(sub_text, sec, metadata))

        logger.info("academic_chunker_produced_chunks", count=len(chunks))
        return chunks


# ── Module-level helpers ──────────────────────────────────────────────────────

def _find_sections(pattern: re.Pattern, text: str) -> list[_Section]:
    sections = []
    for m in pattern.finditer(text):
        number = (m.group(1) or m.group(3) or "").strip()
        title = (m.group(2) or m.group(4) or m.group(5) or "").strip()
        sections.append(_Section(title=title, number=number, start=m.start()))
    return sections


def _sub_chunk_by_paragraph(text: str, max_size: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    chunks: list[str] = []
    current_parts: list[str] = []
    current_size = 0

    for para in paragraphs:
        sep = 2 if current_parts else 0
        if current_size + len(para) + sep > max_size and current_parts:
            chunks.append("\n\n".join(current_parts))
            current_parts = []
            current_size = 0
        current_parts.append(para)
        current_size += len(para) + sep

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks or [text]


def _make_chunk(text: str, section: _Section, metadata: dict) -> Chunk:
    return Chunk(
        id=str(uuid.uuid4()),
        text=text,
        metadata={
            **metadata,
            "chunker": "academic",
            "section_title": section.title,
            "section_number": section.number,
            "document_type": "academic",
        },
    )
