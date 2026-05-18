"""Text chunker implementation using LangChain's RecursiveCharacterTextSplitter.

RecursiveChunker is the V1 concrete implementation of the Chunker Protocol. It
splits document text into overlapping windows using a hierarchy of separators
(paragraphs → sentences → words → characters), preserving natural boundaries
wherever possible.

Note on Protocol drift: the Chunker Protocol in base.py declares
`chunk(pages: list[str], doc_id: str) -> list[Chunk]`. This implementation's
primary method signature is `chunk(parsed_doc: ParsedDocument, doc_id: str)`
to carry metadata forward from the parser. The IngestionPipeline calls this
concrete method directly.
"""

import uuid

import structlog
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.exceptions import ChunkingError
from src.utils.models import Chunk, ParsedDocument

logger = structlog.get_logger(__name__)


class RecursiveChunker:
    """Splits document text into overlapping Chunk objects.

    Uses LangChain's RecursiveCharacterTextSplitter, which tries to split on
    paragraph breaks first, then sentences, then words, then characters. This
    produces chunks that respect natural language boundaries where possible.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        """Initialise the chunker with size and overlap parameters.

        Args:
            chunk_size: Target character count per chunk.
            chunk_overlap: Number of characters of overlap between adjacent chunks.
                Overlap ensures sentences straddling boundaries appear in at least
                one chunk, improving retrieval recall.
        """
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )

    def chunk(self, parsed_doc: ParsedDocument, doc_id: str) -> list[Chunk]:
        """Split a ParsedDocument into a list of Chunk objects.

        Uses the full document text (not per-page) so the splitter can form
        clean boundaries across page breaks.

        Args:
            parsed_doc: Output of a Parser.parse() call.
            doc_id: Stable document identifier — embedded in each chunk's metadata
                and used to correlate chunks back to their source document.

        Returns:
            Ordered list of Chunk objects. Each chunk carries metadata including
            doc_id, chunk_index, source filename, and total_pages.

        Raises:
            ChunkingError: If the document has no extractable text.
        """
        log = logger.bind(doc_id=doc_id, filename=parsed_doc.metadata.get("filename", "unknown"))

        if not parsed_doc.text.strip():
            raise ChunkingError(f"Document '{doc_id}' has no extractable text to chunk.")

        raw_chunks = self._splitter.split_text(parsed_doc.text)

        if not raw_chunks:
            raise ChunkingError(f"Splitter produced no chunks for document '{doc_id}'.")

        chunks: list[Chunk] = []
        for idx, text in enumerate(raw_chunks):
            chunk_id = str(uuid.uuid4())
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=text,
                    metadata={
                        "document_id": doc_id,
                        "chunk_index": idx,
                        "filename": parsed_doc.metadata.get("filename", ""),
                        "total_pages": parsed_doc.metadata.get("total_pages", 0),
                        "source_path": parsed_doc.metadata.get("source_path", ""),
                    },
                )
            )

        log.info("Text chunked", chunk_count=len(chunks), chunk_size=self._chunk_size, chunk_overlap=self._chunk_overlap)
        return chunks
