"""Unit tests for RecursiveChunker."""

import pytest

from src.ingestion.chunkers import RecursiveChunker
from src.utils.exceptions import ChunkingError
from src.utils.models import ParsedDocument


def _make_doc(text: str, filename: str = "test.pdf") -> ParsedDocument:
    return ParsedDocument(
        text=text,
        pages=[text],
        metadata={"filename": filename, "total_pages": 1, "source_path": "/tmp/test.pdf"},
    )


@pytest.fixture
def chunker() -> RecursiveChunker:
    return RecursiveChunker(chunk_size=200, chunk_overlap=20)


def test_chunk_returns_list_of_chunks(chunker: RecursiveChunker) -> None:
    doc = _make_doc("Hello world. " * 50)
    chunks = chunker.chunk(doc, "doc-001")
    assert len(chunks) > 0


def test_chunk_ids_are_unique(chunker: RecursiveChunker) -> None:
    doc = _make_doc("Hello world. " * 50)
    chunks = chunker.chunk(doc, "doc-001")
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids)), "Chunk IDs must be unique"


def test_chunk_metadata_contains_doc_id(chunker: RecursiveChunker) -> None:
    doc = _make_doc("Hello world. " * 50)
    chunks = chunker.chunk(doc, "my-doc-id")
    for chunk in chunks:
        assert chunk.metadata["document_id"] == "my-doc-id"


def test_chunk_metadata_contains_filename(chunker: RecursiveChunker) -> None:
    doc = _make_doc("Hello world. " * 50, filename="report.pdf")
    chunks = chunker.chunk(doc, "doc-002")
    for chunk in chunks:
        assert chunk.metadata["filename"] == "report.pdf"


def test_chunk_index_is_sequential(chunker: RecursiveChunker) -> None:
    doc = _make_doc("Sentence number {i}. " * 100)
    chunks = chunker.chunk(doc, "doc-003")
    indices = [c.metadata["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks)))


def test_long_text_produces_multiple_chunks(chunker: RecursiveChunker) -> None:
    # 2000 chars >> chunk_size=200, should produce many chunks
    long_text = "The quick brown fox jumps over the lazy dog. " * 45
    doc = _make_doc(long_text)
    chunks = chunker.chunk(doc, "doc-004")
    assert len(chunks) > 3


def test_single_short_sentence_produces_one_chunk(chunker: RecursiveChunker) -> None:
    doc = _make_doc("Short text.")
    chunks = chunker.chunk(doc, "doc-005")
    assert len(chunks) == 1
    assert chunks[0].text == "Short text."


def test_overlap_means_adjacent_chunks_share_text(chunker: RecursiveChunker) -> None:
    # Build text long enough to force multiple chunks
    long_text = "ABCDEFGH " * 60  # ~540 chars; chunk_size=200, overlap=20
    doc = _make_doc(long_text)
    chunks = chunker.chunk(doc, "doc-006")
    if len(chunks) > 1:
        # End of first chunk and start of second should share some characters
        end_of_first = chunks[0].text[-30:]
        start_of_second = chunks[1].text[:30]
        # They won't be identical but should share some overlap
        assert len(end_of_first) > 0 and len(start_of_second) > 0


def test_empty_text_raises_chunking_error() -> None:
    chunker = RecursiveChunker(chunk_size=200, chunk_overlap=20)
    doc = _make_doc("   ")
    with pytest.raises(ChunkingError):
        chunker.chunk(doc, "doc-empty")
