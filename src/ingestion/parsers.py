"""PDF parser implementation using pypdf.

PyPDFParser is the V1 concrete implementation of the Parser Protocol. It
extracts text page-by-page from a PDF and returns a ParsedDocument that carries
both per-page text and document-level metadata.

Note on Protocol drift: the Parser Protocol in base.py declares
`parse(file_path) -> list[str]`. This implementation returns ParsedDocument
for richer metadata. The Protocol will be updated in a future phase once the
design stabilises. The IngestionPipeline uses PyPDFParser directly (not via the
Protocol type) so this drift has no runtime impact.
"""

import os

import structlog
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from src.utils.exceptions import DocumentParseError
from src.utils.models import ParsedDocument

logger = structlog.get_logger(__name__)


class PyPDFParser:
    """Extracts text from PDF files using the pypdf library.

    Suitable for digitally-created PDFs. Does NOT perform OCR — scanned PDFs
    will return empty or garbage text. Use AzureDocumentIntelligenceParser
    (Phase 2+) for scanned documents.
    """

    def parse(self, file_path: str, page_limit: int | None = None) -> ParsedDocument:
        """Parse a PDF file and return its content as a ParsedDocument.

        Args:
            file_path: Absolute or relative path to the PDF file.
            page_limit: If set, only the first N pages are extracted. None means all pages.

        Returns:
            ParsedDocument with per-page text, concatenated full text, and
            metadata containing filename, total_pages, and source_path.

        Raises:
            DocumentParseError: If the file does not exist, is not a valid PDF,
                or cannot be read.
        """
        log = logger.bind(file_path=file_path)
        log.info("Parsing PDF", page_limit=page_limit)

        if not os.path.exists(file_path):
            raise DocumentParseError(f"File not found: {file_path}")

        try:
            reader = PdfReader(file_path)
        except (PdfReadError, Exception) as exc:
            raise DocumentParseError(f"Cannot open PDF '{file_path}': {exc}") from exc

        all_pages = reader.pages
        pages_to_parse = all_pages[:page_limit] if page_limit is not None else all_pages

        pages: list[str] = []
        for page_num, page in enumerate(pages_to_parse):
            try:
                text = page.extract_text() or ""
            except Exception as exc:
                log.warning("Failed to extract text from page", page=page_num, error=str(exc))
                text = ""
            pages.append(text)

        full_text = "\n\n".join(p for p in pages if p)
        filename = os.path.basename(file_path)
        total_pdf_pages = len(all_pages)

        log.info(
            "PDF parsed",
            filename=filename,
            pages_parsed=len(pages),
            total_pdf_pages=total_pdf_pages,
            text_length=len(full_text),
        )

        return ParsedDocument(
            text=full_text,
            pages=pages,
            metadata={
                "filename": filename,
                "total_pages": len(pages),
                "total_pdf_pages": total_pdf_pages,
                "source_path": os.path.abspath(file_path),
            },
        )
