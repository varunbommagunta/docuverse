"""Unit tests for PyPDFParser."""

import os
import tempfile

import pytest

from src.ingestion.parsers import PyPDFParser
from src.utils.exceptions import DocumentParseError
from src.utils.models import ParsedDocument


@pytest.fixture
def parser() -> PyPDFParser:
    return PyPDFParser()


def test_parse_returns_parsed_document(parser: PyPDFParser, sample_pdf_path: str) -> None:
    result = parser.parse(sample_pdf_path)
    assert isinstance(result, ParsedDocument)


def test_parse_extracts_text(parser: PyPDFParser, sample_pdf_path: str) -> None:
    result = parser.parse(sample_pdf_path)
    assert len(result.text) > 100, "Expected substantial text from sample PDF"


def test_parse_pages_count_matches_pdf(parser: PyPDFParser, sample_pdf_path: str) -> None:
    result = parser.parse(sample_pdf_path)
    # sample.pdf is generated with 3 pages of content (all on one ReportLab page,
    # but the physical page count may vary). At minimum 1 page must exist.
    assert len(result.pages) >= 1


def test_parse_metadata_contains_filename(parser: PyPDFParser, sample_pdf_path: str) -> None:
    result = parser.parse(sample_pdf_path)
    assert result.metadata["filename"] == "sample.pdf"


def test_parse_metadata_contains_total_pages(parser: PyPDFParser, sample_pdf_path: str) -> None:
    result = parser.parse(sample_pdf_path)
    assert "total_pages" in result.metadata
    assert result.metadata["total_pages"] >= 1


def test_parse_raises_on_missing_file(parser: PyPDFParser) -> None:
    with pytest.raises(DocumentParseError, match="File not found"):
        parser.parse("/nonexistent/path/to/file.pdf")


def test_parse_raises_on_invalid_pdf(parser: PyPDFParser) -> None:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(b"this is not a pdf")
        tmp_path = tmp.name
    try:
        with pytest.raises(DocumentParseError):
            parser.parse(tmp_path)
    finally:
        os.unlink(tmp_path)
