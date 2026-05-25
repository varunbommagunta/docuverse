"""IngestionPipeline — orchestrates the parse → chunk → embed → store flow.

A single call to ingest() takes a file path, drives the full ingestion pipeline,
and returns a summary dict. The pipeline holds concrete references to its
dependencies (not Protocol types) because it calls methods that extend beyond
the Protocol contracts (e.g., PyPDFParser.parse returns ParsedDocument).
"""

import os
import time
import uuid
from typing import TYPE_CHECKING

import structlog

from src.ingestion.chunkers import RecursiveChunker
from src.ingestion.parsers import PyPDFParser
from src.retrieval.embedders import OpenAIEmbedder
from src.retrieval.vector_store import ChromaVectorStore

if TYPE_CHECKING:
    from src.ingestion.router import ChunkerRouter

logger = structlog.get_logger(__name__)


class IngestionPipeline:
    """Orchestrates the full document ingestion flow.

    Coordinates:
      1. Parse: PDF → ParsedDocument
      2. Chunk: ParsedDocument → list[Chunk]  (via ChunkerRouter if provided)
      3. Embed: list[Chunk] → list[list[float]]
      4. Store: Chunks + embeddings → VectorStore

    The pipeline is the unit that routes call through — it has no business logic
    of its own, only coordination and timing.
    """

    def __init__(
        self,
        parser: PyPDFParser,
        chunker: RecursiveChunker,
        embedder: OpenAIEmbedder,
        vector_store: ChromaVectorStore,
        chunker_router: "ChunkerRouter | None" = None,
    ) -> None:
        self._parser = parser
        self._chunker = chunker
        self._embedder = embedder
        self._vector_store = vector_store
        self._chunker_router = chunker_router

    def ingest(self, file_path: str, filename: str | None = None) -> dict[str, object]:
        """Ingest a PDF file end-to-end.

        Generates a unique document_id, parses the PDF, chunks the text,
        embeds all chunks, and stores them in the vector store.

        Args:
            file_path: Path to the PDF file on disk.
            filename: Human-readable filename to embed in metadata. If None,
                the basename of file_path is used.

        Returns:
            dict with keys: document_id (str), filename (str), chunk_count (int).

        Raises:
            DocumentParseError: If the PDF cannot be parsed.
            ChunkingError: If no chunks are produced.
        """
        doc_id = str(uuid.uuid4())
        display_name = filename or os.path.basename(file_path)
        log = logger.bind(doc_id=doc_id, filename=display_name)

        # ── 1. Parse ──────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        parsed_doc = self._parser.parse(file_path)
        # Override metadata filename with the user-facing name
        parsed_doc.metadata["filename"] = display_name
        log.info("Parse complete", elapsed_ms=round((time.perf_counter() - t0) * 1000))

        # ── 2. Chunk ──────────────────────────────────────────────────────────
        t1 = time.perf_counter()
        if self._chunker_router is not None:
            doc_meta = {
                "document_id": doc_id,
                "filename": display_name,
                "source": file_path,
                **{k: v for k, v in parsed_doc.metadata.items()
                   if k not in ("document_id", "filename", "source")},
            }
            chunks, classification = self._chunker_router.route_and_chunk(parsed_doc.text, doc_meta)
            log.info(
                "Chunk complete",
                chunk_count=len(chunks),
                doc_type=classification.doc_type.value,
                elapsed_ms=round((time.perf_counter() - t1) * 1000),
            )
        else:
            chunks = self._chunker.chunk(parsed_doc, doc_id)
            log.info("Chunk complete", chunk_count=len(chunks), elapsed_ms=round((time.perf_counter() - t1) * 1000))

        # ── 3. Embed ──────────────────────────────────────────────────────────
        t2 = time.perf_counter()
        texts = [c.text for c in chunks]
        embeddings = self._embedder.embed(texts)
        log.info("Embed complete", vector_count=len(embeddings), elapsed_ms=round((time.perf_counter() - t2) * 1000))

        # ── 4. Store ──────────────────────────────────────────────────────────
        t3 = time.perf_counter()
        self._vector_store.add_chunks(chunks, embeddings)
        log.info("Store complete", elapsed_ms=round((time.perf_counter() - t3) * 1000))

        total_ms = round((time.perf_counter() - t0) * 1000)
        log.info("Ingestion pipeline complete", total_elapsed_ms=total_ms)

        return {
            "document_id": doc_id,
            "filename": display_name,
            "chunk_count": len(chunks),
        }
