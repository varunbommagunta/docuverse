"""Unit tests for LegalChunker."""

import pytest

from src.ingestion.legal_chunker import LegalChunker


def test_legal_chunker_splits_at_articles():
    chunker = LegalChunker()
    text = """
5. Citizenship at the commencement of the Constitution.—At the commencement
of this Constitution, every person who has his domicile in India shall be
deemed to be a citizen of India.

6. Rights of citizenship.—Notwithstanding anything in article 5, a person
who has migrated to India shall be deemed a citizen.
"""
    chunks = chunker.chunk(text)
    assert len(chunks) == 2
    assert "article_id" in chunks[0].metadata
    assert chunks[0].metadata["article_id"] == "5"
    assert chunks[1].metadata["article_id"] == "6"


def test_legal_chunker_captures_article_title():
    chunker = LegalChunker()
    text = "312. All-India services.—(1) Notwithstanding anything in this Constitution, Parliament may by law provide for the creation of one or more all India services."
    chunks = chunker.chunk(text)
    assert len(chunks) >= 1
    assert "All-India services" in chunks[0].metadata.get("article_title", "")


def test_legal_chunker_captures_part_metadata():
    chunker = LegalChunker()
    text = """
PART II CITIZENSHIP

5. Citizenship.—At the commencement of this Constitution, every person who has his
domicile in the territory of India and who was born in the territory of India shall
be deemed to be a citizen of India under this article.

6. Rights of citizenship of certain persons.—Notwithstanding anything in article 5,
a person who has migrated to the territory of India from the territory now included
in Pakistan shall be deemed to be a citizen of India at the commencement.
"""
    chunks = chunker.chunk(text)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert "PART II" in chunk.metadata.get("part", "")


def test_legal_chunker_handles_no_articles_found():
    """When no articles detected, returns single chunk with warning."""
    chunker = LegalChunker()
    text = "This is just prose text without any article structure or markers."
    chunks = chunker.chunk(text)
    assert len(chunks) == 1
    assert "warning" in chunks[0].metadata


def test_legal_chunker_skips_short_false_positives():
    """Articles with very short bodies (< 100 chars) are filtered out."""
    chunker = LegalChunker()
    # Article 5 has a proper body; the very short one should be skipped
    text = """
5. Short.—x

6. Real article.—This article has a real body that is long enough to be included in the output chunks.
"""
    chunks = chunker.chunk(text)
    ids = [c.metadata.get("article_id") for c in chunks]
    assert "5" not in ids
    assert "6" in ids


def test_legal_chunker_produces_chunks_with_ids():
    chunker = LegalChunker()
    text = """
5. Citizenship.—At the commencement of this Constitution, every person who has his
domicile in the territory of India and who was born in the territory of India shall be
deemed to be a citizen of India.
"""
    chunks = chunker.chunk(text)
    for chunk in chunks:
        assert chunk.id is not None
        assert len(chunk.id) > 0


def test_legal_chunker_metadata_contains_chunker_name():
    chunker = LegalChunker()
    text = """
5. Citizenship.—At the commencement of this Constitution every person who has domicile
in India and who was born in India shall be deemed to be a citizen of India.
"""
    chunks = chunker.chunk(text)
    for chunk in chunks:
        assert chunk.metadata.get("chunker") == "legal"


def test_legal_chunker_passes_through_document_metadata():
    chunker = LegalChunker()
    text = """
5. Citizenship.—At the commencement of this Constitution every person who has domicile
in India and who was born in India shall be deemed to be a citizen of India.
"""
    doc_meta = {"source": "constitution.pdf", "document_id": "abc-123"}
    chunks = chunker.chunk(text, doc_meta)
    for chunk in chunks:
        assert chunk.metadata.get("source") == "constitution.pdf"
        assert chunk.metadata.get("document_id") == "abc-123"
