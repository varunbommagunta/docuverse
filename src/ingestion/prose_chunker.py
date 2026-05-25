"""Prose document chunker for novels, essays, articles, blog posts.

Splits at paragraph boundaries (\\n\\n). Oversized paragraphs are further
split at sentence boundaries. A 100-char overlap carries context from the
end of the previous chunk into the start of the next.
"""

import re
import uuid

import structlog

from src.utils.models import Chunk

logger = structlog.get_logger(__name__)

_SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])\s+')


class ProseChunker:
    """Chunker for prose documents: paragraph-boundary splitting with overlap."""

    name = "prose"

    def __init__(self, max_chunk_size: int = 800, overlap: int = 100) -> None:
        self._max = max_chunk_size
        self._overlap = overlap

    def chunk(self, document_text: str, document_metadata: dict | None = None) -> list[Chunk]:
        """Split prose document into paragraph-based chunks.

        Args:
            document_text: Full document text.
            document_metadata: Passed through to each chunk's metadata.

        Returns:
            List of Chunk objects with paragraph_index metadata.
        """
        metadata = document_metadata or {}

        raw_paragraphs = re.split(r'\n{2,}', document_text)
        paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]

        if not paragraphs:
            return []

        # Expand paragraphs that exceed max into sentence-level units.
        # Each unit is (original_paragraph_index, text).
        units: list[tuple[int, str]] = []
        for idx, para in enumerate(paragraphs):
            if len(para) > self._max:
                for sent in _split_sentences(para):
                    units.append((idx, sent))
            else:
                units.append((idx, para))

        chunks: list[Chunk] = []
        current_texts: list[str] = []
        current_size = 0
        chunk_para_start = units[0][0]

        for para_idx, text in units:
            separator_cost = 2 if current_texts else 0  # '\n\n'
            if current_size + len(text) + separator_cost > self._max and current_texts:
                chunk_text = "\n\n".join(current_texts)
                chunks.append(_make_chunk(chunk_text, chunk_para_start, metadata))

                # Carry last `overlap` chars (word-aligned) into the next chunk.
                tail = _word_aligned_tail(chunk_text, self._overlap)
                current_texts = [tail] if tail else []
                current_size = len(tail)
                chunk_para_start = para_idx

            current_texts.append(text)
            current_size += len(text) + (2 if len(current_texts) > 1 else 0)

        if current_texts:
            chunk_text = "\n\n".join(current_texts)
            if chunk_text.strip():
                chunks.append(_make_chunk(chunk_text, chunk_para_start, metadata))

        logger.info("prose_chunker_produced_chunks", count=len(chunks))
        return chunks


# ── Module-level helpers (no self needed) ────────────────────────────────────

def _make_chunk(text: str, para_idx: int, metadata: dict) -> Chunk:
    return Chunk(
        id=str(uuid.uuid4()),
        text=text,
        metadata={
            **metadata,
            "chunker": "prose",
            "paragraph_index": para_idx,
            "document_type": "prose",
        },
    )


def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_BOUNDARY.split(text)
    return [p.strip() for p in parts if p.strip()] or [text]


def _word_aligned_tail(text: str, n: int) -> str:
    """Return the last n chars of text, trimmed forward to the next word boundary."""
    if len(text) <= n:
        return ""
    tail = text[-n:]
    space = tail.find(' ')
    return tail[space + 1:] if space != -1 else tail
