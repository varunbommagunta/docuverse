"""Unit tests for ProseChunker."""

import pytest

from src.ingestion.prose_chunker import ProseChunker


_PARA_A = "The sun rose slowly over the hills, casting long shadows across the valley below. " \
          "Birds began to sing their morning songs as the world came alive with color and sound."

_PARA_B = "She walked along the riverbank, lost in thought. " \
          "The water rushed past her feet, cold and clear, carrying leaves and small twigs downstream."

_PARA_C = "He said nothing for a long time. Then, finally, he turned and spoke her name."


def test_prose_chunker_splits_at_paragraphs():
    chunker = ProseChunker(max_chunk_size=800)
    text = f"{_PARA_A}\n\n{_PARA_B}\n\n{_PARA_C}"
    chunks = chunker.chunk(text)
    assert len(chunks) >= 1
    # All content should be present across chunks
    combined = " ".join(c.text for c in chunks)
    assert "sun rose" in combined
    assert "riverbank" in combined
    assert "said nothing" in combined


def test_prose_chunker_groups_small_paragraphs_into_one_chunk():
    chunker = ProseChunker(max_chunk_size=800)
    # Three short paragraphs totalling well under 800 chars
    text = "Para one.\n\nPara two.\n\nPara three."
    chunks = chunker.chunk(text)
    assert len(chunks) == 1
    assert "Para one" in chunks[0].text
    assert "Para three" in chunks[0].text


def test_prose_chunker_oversized_paragraph_splits_by_sentence():
    chunker = ProseChunker(max_chunk_size=100)
    # Single paragraph of ~300 chars
    long_para = (
        "The old man sat by the fire and remembered. "
        "He had lived a long life, full of hardship and joy. "
        "Now, in his final years, he found peace in the quiet of the evening. "
        "The flames danced before him like old friends."
    )
    chunks = chunker.chunk(long_para)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 200  # some slack for overlap


def test_prose_chunker_metadata_paragraph_index_present():
    chunker = ProseChunker(max_chunk_size=800)
    text = f"{_PARA_A}\n\n{_PARA_B}"
    chunks = chunker.chunk(text)
    for chunk in chunks:
        assert "paragraph_index" in chunk.metadata
        assert isinstance(chunk.metadata["paragraph_index"], int)


def test_prose_chunker_metadata_document_type_is_prose():
    chunker = ProseChunker(max_chunk_size=800)
    chunks = chunker.chunk(_PARA_A)
    for chunk in chunks:
        assert chunk.metadata.get("document_type") == "prose"
        assert chunk.metadata.get("chunker") == "prose"


def test_prose_chunker_metadata_chunker_name():
    chunker = ProseChunker(max_chunk_size=800)
    chunks = chunker.chunk(_PARA_A)
    assert all(c.metadata.get("chunker") == "prose" for c in chunks)


def test_prose_chunker_passes_through_document_metadata():
    chunker = ProseChunker(max_chunk_size=800)
    doc_meta = {"filename": "novel.pdf", "document_id": "abc-123"}
    chunks = chunker.chunk(_PARA_A, doc_meta)
    for chunk in chunks:
        assert chunk.metadata.get("filename") == "novel.pdf"
        assert chunk.metadata.get("document_id") == "abc-123"


def test_prose_chunker_produces_chunks_with_unique_ids():
    chunker = ProseChunker(max_chunk_size=100)
    text = "\n\n".join([_PARA_A, _PARA_B, _PARA_C])
    chunks = chunker.chunk(text)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


def test_prose_chunker_overlap_carries_context():
    chunker = ProseChunker(max_chunk_size=len(_PARA_A) + 5, overlap=50)
    text = f"{_PARA_A}\n\n{_PARA_B}"
    chunks = chunker.chunk(text)
    # If chunked into two, the second chunk should start with overlap text from first
    if len(chunks) == 2:
        # The overlap tail of chunk 0 should appear somewhere in chunk 1 text
        tail_words = chunks[0].text.split()[-5:]
        assert any(w in chunks[1].text for w in tail_words)


def test_prose_chunker_empty_text_returns_empty():
    chunker = ProseChunker()
    assert chunker.chunk("") == []
    assert chunker.chunk("   \n\n   ") == []
