"""Protocol interfaces for the ingestion layer.

The ingestion layer is responsible for converting raw binary documents (PDFs)
into a sequence of Chunk objects ready for embedding. Two distinct concerns
are deliberately separated:

  Parser  — knows how to open a file format and extract raw text
  Chunker — knows how to split raw text into overlapping windows

Keeping these as separate protocols means we can swap out each independently.
For example, upgrading from PyMuPDF to pdfplumber only requires a new Parser;
the Chunker stays the same.
"""

from typing import Protocol

from src.utils.models import Chunk


class Parser(Protocol):
    """Converts a raw document file into a sequence of page-level text strings.

    Each string in the returned list corresponds to one logical page. The caller
    (Chunker or Orchestrator) is responsible for stitching pages together if
    cross-page chunks are desired.

    Why a Protocol rather than an ABC?
    Structural subtyping means any class with a matching `parse` signature
    automatically satisfies this contract — no explicit inheritance needed.
    This makes it easy to wrap third-party libraries without adapter boilerplate.

    V1 implementation: PyMuPDFParser — fast, handles most PDFs, returns plain text.
    V2 implementation: AzureDocumentIntelligenceParser — handles scanned PDFs via OCR.
    V3 implementation: UnstructuredParser — multi-format (DOCX, HTML, Markdown).
    """

    def parse(self, file_path: str) -> list[str]:
        """Parse a document and return one string per logical page.

        Args:
            file_path: Absolute or relative path to the source document.

        Returns:
            Ordered list of page text. Empty pages are represented as empty strings,
            not omitted, so callers can correlate list index to page number.

        Raises:
            DocumentParseError: If the file cannot be opened or decoded.
        """
        ...


class Chunker(Protocol):
    """Splits a sequence of page strings into overlapping Chunk objects.

    Overlapping windows ensure that sentences straddling a chunk boundary are
    captured in at least one chunk, improving retrieval recall.

    V1 implementation: RecursiveCharacterChunker — LangChain-style, splits on
        paragraph/sentence/word boundaries in order; falls back to characters.
    V2 implementation: SemanticChunker — embeds sentences, groups by cosine
        similarity threshold; produces semantically coherent chunks.
    V3 implementation: LateChunkingChunker — defers chunking until after
        embedding using long-context models; preserves global context.
    """

    def chunk(self, pages: list[str], doc_id: str) -> list[Chunk]:
        """Split page texts into Chunks with overlap.

        Args:
            pages: Ordered list of page strings, as returned by Parser.parse().
            doc_id: Stable identifier for the source document. Used to generate
                    chunk IDs of the form '{doc_id}_{index}'.

        Returns:
            Ordered list of Chunk objects. Metadata must include at minimum
            'doc_id', 'page_index', and 'chunk_index'.

        Raises:
            ChunkingError: If chunking cannot proceed (e.g., all pages are empty).
        """
        ...
