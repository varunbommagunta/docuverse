"""Technical document chunker for manuals, API docs, specifications.

Splits at markdown headings (##/###), numbered steps (1., 2.), or ALL CAPS
titled blocks. Code blocks (```...```) are protected with placeholders during
splitting so they are never broken across chunks.
"""

import re
import uuid

import structlog

from src.utils.models import Chunk

logger = structlog.get_logger(__name__)

_CODE_BLOCK = re.compile(r'```[\s\S]*?```', re.MULTILINE)

# Matches (any of):
#   "## Heading" / "### Heading"           markdown
#   "1. Do this" / "Step 2. Do that"       numbered steps
#   "ALL CAPS TITLE" / "ALL CAPS TITLE:"   bold section labels
_HEADING = re.compile(
    r'^(?:#{1,3}\s+.+'
    r'|(?:Step\s+)?\d+\.\s+[A-Z][^\n]{0,80}'
    r'|[A-Z][A-Z\s\-]{3,60}:?)$',
    re.MULTILINE,
)


class TechnicalChunker:
    """Chunker for technical documents: heading/step-boundary splitting."""

    name = "technical"

    def __init__(self, max_chunk_size: int = 800) -> None:
        self._max = max_chunk_size

    def chunk(self, document_text: str, document_metadata: dict | None = None) -> list[Chunk]:
        """Split technical document at headings, preserving code blocks intact.

        Args:
            document_text: Full document text.
            document_metadata: Passed through to each chunk's metadata.

        Returns:
            List of Chunk objects with section_title metadata.
        """
        metadata = document_metadata or {}

        # Replace code blocks with fixed-width placeholders so heading/paragraph
        # patterns never accidentally fall inside a code block.
        code_blocks: dict[str, str] = {}

        def _save(m: re.Match) -> str:
            key = f"\x00CODE_{len(code_blocks)}\x00"
            code_blocks[key] = m.group(0)
            return key

        protected = _CODE_BLOCK.sub(_save, document_text)
        headings = list(_HEADING.finditer(protected))

        if not headings:
            logger.warning("technical_chunker_no_headings_detected")
            # Fall back: sub-chunk the whole text by paragraph
            chunks = [
                _make_chunk(_restore(t, code_blocks), "", metadata)
                for t in _sub_chunk_protected(protected, self._max)
            ]
            if not chunks:
                chunks = [_make_chunk(document_text[:self._max], "", metadata)]
            return chunks

        logger.info("technical_chunker_found_headings", count=len(headings))
        chunks: list[Chunk] = []

        for i, h in enumerate(headings):
            end = headings[i + 1].start() if i + 1 < len(headings) else len(protected)
            section_protected = protected[h.start():end]
            title = h.group(0).strip().lstrip('#').strip()

            if len(section_protected) <= self._max:
                restored = _restore(section_protected, code_blocks).strip()
                chunks.append(_make_chunk(restored, title, metadata))
            else:
                for sub in _sub_chunk_protected(section_protected, self._max):
                    restored = _restore(sub, code_blocks).strip()
                    chunks.append(_make_chunk(restored, title, metadata))

        logger.info("technical_chunker_produced_chunks", count=len(chunks))
        return chunks


# ── Module-level helpers ──────────────────────────────────────────────────────

def _make_chunk(text: str, title: str, metadata: dict) -> Chunk:
    return Chunk(
        id=str(uuid.uuid4()),
        text=text,
        metadata={
            **metadata,
            "chunker": "technical",
            "section_title": title,
            "document_type": "technical",
        },
    )


def _sub_chunk_protected(text: str, max_size: int) -> list[str]:
    """Split protected text at \\n\\n boundaries (never inside a placeholder)."""
    parts = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    result: list[str] = []
    current_parts: list[str] = []
    current_size = 0

    for part in parts:
        sep = 2 if current_parts else 0
        if current_size + len(part) + sep > max_size and current_parts:
            result.append("\n\n".join(current_parts))
            current_parts = []
            current_size = 0
        current_parts.append(part)
        current_size += len(part) + sep

    if current_parts:
        result.append("\n\n".join(current_parts))

    return result


def _restore(text: str, code_blocks: dict[str, str]) -> str:
    for key, block in code_blocks.items():
        text = text.replace(key, block)
    return text
