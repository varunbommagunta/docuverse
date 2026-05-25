"""Routes documents to appropriate chunkers based on classification."""

import uuid

import structlog

from src.ingestion.chunkers import RecursiveChunker
from src.ingestion.classifier import DocumentClassifier, DocumentType, ClassificationResult
from src.ingestion.legal_chunker import LegalChunker
from src.utils.models import Chunk, ParsedDocument

logger = structlog.get_logger(__name__)


class DefaultChunker:
    """Adapter wrapping RecursiveChunker for the router's document-text interface."""

    name = "default"

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        self._chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def chunk(self, document_text: str, document_metadata: dict | None = None) -> list[Chunk]:
        doc_metadata = document_metadata or {}
        doc_id = doc_metadata.get("document_id", str(uuid.uuid4()))
        parsed_doc = ParsedDocument(
            text=document_text,
            pages=[document_text],
            metadata=doc_metadata,
        )
        chunks = self._chunker.chunk(parsed_doc, doc_id)
        # RecursiveChunker only propagates specific keys; merge any extras from doc_metadata
        known_keys = {"document_id", "chunk_index", "filename", "total_pages", "source_path"}
        extra = {k: v for k, v in doc_metadata.items() if k not in known_keys}
        if extra:
            for chunk in chunks:
                chunk.metadata.update(extra)
        return chunks


class ChunkerRouter:
    """Dispatches documents to the right chunker based on classification."""

    def __init__(self, classifier: DocumentClassifier) -> None:
        self._classifier = classifier
        self._chunkers: dict[DocumentType, object] = {
            DocumentType.LEGAL: LegalChunker(),
            DocumentType.DEFAULT: DefaultChunker(),
        }

    def route_and_chunk(
        self, document_text: str, document_metadata: dict | None = None
    ) -> tuple[list[Chunk], ClassificationResult]:
        """Classify document, route to correct chunker, return chunks and classification."""
        if document_metadata is None:
            document_metadata = {}

        from pathlib import Path
        file_path = Path(document_metadata["source"]) if "source" in document_metadata else None
        classification = self._classifier.classify(document_text, file_path)

        logger.info(
            "router_classified_document",
            source=document_metadata.get("source", "unknown"),
            doc_type=classification.doc_type,
            confidence=classification.confidence,
            method=classification.method,
        )

        chunker = self._chunkers.get(classification.doc_type, self._chunkers[DocumentType.DEFAULT])

        enriched_metadata = {
            **document_metadata,
            "document_type": classification.doc_type.value,
            "classification_confidence": classification.confidence,
            "classification_method": classification.method,
        }

        chunks = chunker.chunk(document_text, enriched_metadata)

        logger.info(
            "router_produced_chunks",
            count=len(chunks),
            chunker=chunker.name,
        )

        return chunks, classification
