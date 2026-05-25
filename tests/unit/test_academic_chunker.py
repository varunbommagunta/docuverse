"""Unit tests for AcademicChunker."""

import pytest

from src.ingestion.academic_chunker import AcademicChunker

_ABSTRACT = (
    "ABSTRACT\n"
    "This paper examines the effects of machine learning on document retrieval. "
    "We present a novel hybrid approach combining dense and sparse retrieval methods. "
    "Results show a 12% improvement over baseline BM25."
)

_INTRO = (
    "1. Introduction\n"
    "Document retrieval has long been a central problem in information retrieval. "
    "Early systems relied exclusively on keyword matching. "
    "Recent advances in neural networks have enabled semantic search capabilities."
)

_METHOD = (
    "2. Methodology\n"
    "We trained a bi-encoder model on the MS-MARCO dataset. "
    "The model uses a 768-dimensional embedding space. "
    "At query time, cosine similarity is computed against all indexed documents."
)

_REFS = (
    "REFERENCES\n"
    "1. Karpukhin et al. (2020). Dense passage retrieval. ACL 2020.\n"
    "2. Robertson and Zaragoza (2009). BM25. Now Publishers."
)

_FULL_PAPER = f"{_ABSTRACT}\n\n{_INTRO}\n\n{_METHOD}\n\n{_REFS}"


def test_academic_chunker_splits_at_numbered_sections():
    chunker = AcademicChunker()
    chunks = chunker.chunk(_FULL_PAPER)
    titles = [c.metadata.get("section_title", "") for c in chunks]
    assert any("Introduction" in t for t in titles)
    assert any("Methodology" in t for t in titles)


def test_academic_chunker_splits_at_allcaps_headings():
    chunker = AcademicChunker()
    chunks = chunker.chunk(_FULL_PAPER)
    titles = [c.metadata.get("section_title", "") for c in chunks]
    assert any("ABSTRACT" in t or "REFERENCES" in t for t in titles)


def test_academic_chunker_section_number_in_metadata():
    chunker = AcademicChunker()
    chunks = chunker.chunk(_FULL_PAPER)
    numbered = [c for c in chunks if c.metadata.get("section_number")]
    assert len(numbered) >= 2  # sections 1 and 2


def test_academic_chunker_metadata_fields_present():
    chunker = AcademicChunker()
    chunks = chunker.chunk(_FULL_PAPER)
    for chunk in chunks:
        assert "section_title" in chunk.metadata
        assert "section_number" in chunk.metadata
        assert chunk.metadata.get("document_type") == "academic"
        assert chunk.metadata.get("chunker") == "academic"


def test_academic_chunker_handles_no_sections():
    chunker = AcademicChunker()
    text = "This is just plain text without any headings or structure whatsoever."
    chunks = chunker.chunk(text)
    assert len(chunks) == 1
    assert chunks[0].metadata.get("document_type") == "academic"


def test_academic_chunker_sub_chunks_oversized_sections():
    # Create a section that greatly exceeds hard_cap
    chunker = AcademicChunker(max_chunk_size=200, hard_cap=300)
    long_body = "\n\n".join(
        [f"This is paragraph {i} of the section. " * 5 for i in range(10)]
    )
    text = f"1. Long Section\n{long_body}"
    chunks = chunker.chunk(text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 600  # reasonable slack


def test_academic_chunker_passes_through_document_metadata():
    chunker = AcademicChunker()
    doc_meta = {"filename": "paper.pdf", "document_id": "xyz-789"}
    chunks = chunker.chunk(_FULL_PAPER, doc_meta)
    for chunk in chunks:
        assert chunk.metadata.get("filename") == "paper.pdf"
        assert chunk.metadata.get("document_id") == "xyz-789"


def test_academic_chunker_produces_chunks_with_unique_ids():
    chunker = AcademicChunker()
    chunks = chunker.chunk(_FULL_PAPER)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


def test_academic_chunker_chapter_keyword_detected():
    chunker = AcademicChunker()
    text = (
        "Chapter 1\nThis is the first chapter of the report. "
        "It introduces the main themes.\n\n"
        "Chapter 2\nThe second chapter explores the findings in detail."
    )
    chunks = chunker.chunk(text)
    assert len(chunks) >= 2
