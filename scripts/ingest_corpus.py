#!/usr/bin/env python
"""Ingest a multi-PDF corpus into ChromaDB for Phase 3a.

Ingests each PDF in --pdf-dir using per-document page limits. With --reset,
the entire ChromaDB collection is wiped first so each run starts clean.

Page limits (to keep token costs reasonable):
  - sample.pdf               : all pages (small synthetic doc)
  - constitution_of_india.pdf: first 50 pages  (~Preamble + Parts I-V)
  - arc_ethics_governance.pdf: first 40 pages  (~Executive Summary + Ch.1-3)
  - any other .pdf           : all pages

Usage
-----
    python scripts/ingest_corpus.py --reset          # wipe + ingest all PDFs
    python scripts/ingest_corpus.py                  # ingest without wiping
    python scripts/ingest_corpus.py --pdf-dir data/sample --reset
"""

import argparse
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog

logger = structlog.get_logger(__name__)

_PAGE_LIMITS: dict[str, int | None] = {
    "sample.pdf": None,
    "constitution_of_india.pdf": None,  # All 402 pages needed for full Article-based chunking
    "arc_ethics_governance.pdf": 40,
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DocuVerse corpus ingestion script")
    p.add_argument("--pdf-dir", default="data/sample", help="Directory containing PDF files")
    p.add_argument(
        "--reset",
        action="store_true",
        help="Wipe the ChromaDB collection before ingesting",
    )
    return p.parse_args()


def _reset_collection(persist_directory: str, collection_name: str = "docuverse") -> None:
    """Delete and recreate the Chroma collection."""
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    client = chromadb.PersistentClient(
        path=persist_directory,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    try:
        client.delete_collection(collection_name)
        logger.info("Collection deleted", collection=collection_name)
    except Exception:
        logger.info("Collection did not exist, nothing to delete", collection=collection_name)
    client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("Collection recreated", collection=collection_name)


def main() -> None:
    args = _parse_args()
    pdf_dir = Path(args.pdf_dir)

    if not pdf_dir.exists():
        logger.error("PDF directory not found", pdf_dir=str(pdf_dir))
        sys.exit(1)

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        logger.error("No PDF files found", pdf_dir=str(pdf_dir))
        sys.exit(1)

    logger.info("Found PDFs", count=len(pdfs), files=[p.name for p in pdfs])

    from config.settings import get_settings

    settings = get_settings()

    if args.reset:
        logger.info("Resetting ChromaDB collection")
        _reset_collection(settings.chroma_persist_directory)

    from src.ingestion.parsers import PyPDFParser
    from src.ingestion.classifier import DocumentClassifier, LLMClassifier
    from src.ingestion.router import ChunkerRouter
    from src.retrieval.embedders import OpenAIEmbedder
    from src.retrieval.vector_store import ChromaVectorStore
    from openai import OpenAI

    parser = PyPDFParser()
    openai_client = OpenAI(api_key=settings.openai_api_key)
    llm_classifier = LLMClassifier(openai_client) if getattr(settings, "enable_llm_classifier_fallback", True) else None
    classifier = DocumentClassifier(llm_classifier=llm_classifier)
    router = ChunkerRouter(classifier)
    embedder = OpenAIEmbedder()
    vector_store = ChromaVectorStore(persist_directory=settings.chroma_persist_directory)

    print()
    print("=" * 60)
    print("  DocuVerse Corpus Ingestion")
    print(f"  PDF directory : {pdf_dir}")
    print(f"  Reset         : {args.reset}")
    print("=" * 60)
    print()

    total_start = time.time()
    for pdf_path in pdfs:
        page_limit = _PAGE_LIMITS.get(pdf_path.name)
        log = logger.bind(filename=pdf_path.name, page_limit=page_limit)
        log.info("Ingesting PDF")

        t0 = time.time()
        try:
            doc_id = str(uuid.uuid4())

            parsed = parser.parse(str(pdf_path), page_limit=page_limit)
            doc_meta = {
                "document_id": doc_id,
                "filename": pdf_path.name,
                "source": str(pdf_path),
                "total_pages": len(parsed.pages),
                "source_path": str(pdf_path),
            }
            chunks, classification = router.route_and_chunk(parsed.text, doc_meta)
            log.info("Document classified", doc_type=classification.doc_type.value,
                     confidence=round(classification.confidence, 2), method=classification.method)
            texts = [c.text for c in chunks]
            embeddings = embedder.embed(texts)
            vector_store.add_chunks(chunks, embeddings)

            elapsed = time.time() - t0
            log.info(
                "Ingestion complete",
                doc_id=doc_id,
                pages_parsed=len(parsed.pages),
                total_pdf_pages=parsed.metadata.get("total_pdf_pages", len(parsed.pages)),
                chunks=len(chunks),
                elapsed_s=round(elapsed, 1),
            )
            limit_note = f" (limit={page_limit})" if page_limit else ""
            print(
                f"  [{pdf_path.name}]{limit_note}  "
                f"{len(parsed.pages)} pages -> {len(chunks)} chunks  "
                f"({elapsed:.1f}s)"
            )
        except Exception as exc:
            log.error("Ingestion failed", error=str(exc))
            print(f"  ERROR ingesting {pdf_path.name}: {exc}")
            sys.exit(1)

    total_elapsed = time.time() - total_start
    final_count = vector_store.count()
    print()
    print(f"  Corpus ready: {final_count} total chunks in ChromaDB  ({total_elapsed:.1f}s total)")
    print()
    logger.info("Corpus ingestion complete", total_chunks=final_count, elapsed_s=round(total_elapsed, 1))


if __name__ == "__main__":
    main()
