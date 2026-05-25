"""Unit tests for TechnicalChunker."""

import pytest

from src.ingestion.technical_chunker import TechnicalChunker

_INSTALL = (
    "## Installation\n"
    "Download the package from the official repository.\n"
    "Run `pip install docuverse` to install all dependencies.\n"
    "Verify the installation with `docuverse --version`."
)

_USAGE = (
    "## Usage\n"
    "Import the library and create a pipeline instance.\n"
    "Call `pipeline.ingest(path)` to index a document.\n"
    "Use `pipeline.query(q)` to retrieve answers."
)

_API = (
    "## API Reference\n"
    "### ingest(path: str) -> dict\n"
    "Ingests a PDF file and returns metadata including chunk count.\n\n"
    "Parameters:\n"
    "- path: Absolute path to the PDF file.\n\n"
    "Returns: dict with keys document_id, filename, chunk_count."
)

_CODE_DOC = (
    "## Example\n"
    "The following snippet shows basic usage:\n\n"
    "```python\n"
    "from docuverse import Pipeline\n\n"
    "pipeline = Pipeline()\n"
    "result = pipeline.ingest('doc.pdf')\n"
    "print(result['chunk_count'])\n"
    "```\n\n"
    "The output will show the number of indexed chunks."
)

_FULL_DOC = f"{_INSTALL}\n\n{_USAGE}\n\n{_API}"


def test_technical_chunker_splits_at_markdown_headings():
    chunker = TechnicalChunker()
    chunks = chunker.chunk(_FULL_DOC)
    titles = [c.metadata.get("section_title", "") for c in chunks]
    assert any("Installation" in t for t in titles)
    assert any("Usage" in t for t in titles)
    assert any("API Reference" in t or "API" in t for t in titles)


def test_technical_chunker_splits_at_numbered_steps():
    chunker = TechnicalChunker()
    text = (
        "1. Download the installer from the website.\n"
        "Save it to a local directory.\n\n"
        "2. Run the installer as administrator.\n"
        "Accept the license agreement when prompted.\n\n"
        "3. Restart your computer to complete the installation."
    )
    chunks = chunker.chunk(text)
    assert len(chunks) >= 2


def test_technical_chunker_preserves_code_blocks():
    chunker = TechnicalChunker(max_chunk_size=100)
    chunks = chunker.chunk(_CODE_DOC)
    # The code block must appear intact in exactly one chunk
    code_chunks = [c for c in chunks if "```python" in c.text]
    assert len(code_chunks) == 1
    assert "pipeline.ingest" in code_chunks[0].text
    assert "print(result" in code_chunks[0].text


def test_technical_chunker_code_block_not_split():
    """A code block that spans more than max_chunk_size must not be broken."""
    chunker = TechnicalChunker(max_chunk_size=50)
    text = (
        "## Setup\n"
        "```python\n"
        "# This is a long code block\n"
        "import os\n"
        "import sys\n"
        "print('hello world')\n"
        "result = some_function()\n"
        "```"
    )
    chunks = chunker.chunk(text)
    code_chunks = [c for c in chunks if "```" in c.text]
    # Code block must be whole — not split into partial ``` occurrences
    for c in code_chunks:
        assert c.text.count("```") % 2 == 0 or c.text.count("```") >= 2


def test_technical_chunker_metadata_fields_present():
    chunker = TechnicalChunker()
    chunks = chunker.chunk(_FULL_DOC)
    for chunk in chunks:
        assert "section_title" in chunk.metadata
        assert chunk.metadata.get("document_type") == "technical"
        assert chunk.metadata.get("chunker") == "technical"


def test_technical_chunker_section_title_stripped_of_hashes():
    chunker = TechnicalChunker()
    chunks = chunker.chunk(_FULL_DOC)
    for chunk in chunks:
        title = chunk.metadata.get("section_title", "")
        assert not title.startswith("#"), f"title still has #: {title!r}"


def test_technical_chunker_handles_no_headings():
    chunker = TechnicalChunker()
    text = "Just some plain technical text without any headings or structure at all."
    chunks = chunker.chunk(text)
    assert len(chunks) >= 1
    assert chunks[0].metadata.get("document_type") == "technical"


def test_technical_chunker_passes_through_document_metadata():
    chunker = TechnicalChunker()
    doc_meta = {"filename": "manual.pdf", "document_id": "tech-001"}
    chunks = chunker.chunk(_FULL_DOC, doc_meta)
    for chunk in chunks:
        assert chunk.metadata.get("filename") == "manual.pdf"
        assert chunk.metadata.get("document_id") == "tech-001"


def test_technical_chunker_produces_chunks_with_unique_ids():
    chunker = TechnicalChunker()
    chunks = chunker.chunk(_FULL_DOC)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
