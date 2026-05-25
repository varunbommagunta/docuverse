"""FastAPI application entry point.

Creates the FastAPI app, registers routes, and wires up the lifespan context
manager for startup/shutdown hooks. The orchestrator and ingestion pipeline are
constructed once at startup and stored on app.state for injection into routes.
"""

import subprocess
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from api.middleware.cost_cap import DailyCostCapMiddleware
from api.routes import corpus, health, ingest, query
from src.utils.logger import configure_logging

logger = structlog.get_logger(__name__)

limiter = Limiter(key_func=get_remote_address, default_limits=["20/minute"])

_AUTO_INGEST_PDFS: list[tuple[str, int | None]] = [
    ("data/sample/constitution_of_india.pdf", None),   # all 402 pages
    ("data/sample/arc_ethics_governance.pdf", 40),
]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan — startup runs before yield, shutdown after."""
    configure_logging()
    logger.info("DocuVerse API starting up")

    from src.factory import get_rag_components
    orchestrator, pipeline = get_rag_components()
    app.state.orchestrator = orchestrator
    app.state.pipeline = pipeline

    # Auto-populate ChromaDB with sample corpus if empty
    try:
        from config.settings import get_settings
        settings = get_settings()
        collection = pipeline._vector_store._collection
        chunk_count = collection.count()
        auto_ingest = settings.auto_ingest_on_startup

        if auto_ingest and chunk_count == 0:
            logger.info("auto_ingest_starting", reason="empty_chroma_db")

            pdf_paths = [
                "data/sample/constitution_of_india.pdf",
                "data/sample/arc_ethics_governance.pdf",
            ]
            missing = [p for p in pdf_paths if not Path(p).exists()]

            if missing:
                logger.info("downloading_sample_pdfs", missing_count=len(missing))
                result = subprocess.run(
                    ["python3", "scripts/download_sample_pdfs.py"],
                    capture_output=True, text=True, timeout=300,
                )
                if result.returncode != 0:
                    logger.error("download_failed", stderr=result.stderr)
                else:
                    logger.info("downloads_complete")

            logger.info("ingesting_corpus_in_process")
            ingest_errors = []
            for pdf_path_str, page_limit in _AUTO_INGEST_PDFS:
                pdf_path = Path(pdf_path_str)
                if not pdf_path.exists():
                    logger.warning("pdf_not_found_skipping", path=pdf_path_str)
                    continue
                try:
                    result_meta = pipeline.ingest(pdf_path_str, page_limit=page_limit)
                    logger.info(
                        "pdf_ingested",
                        filename=pdf_path.name,
                        chunk_count=result_meta["chunk_count"],
                        page_limit=page_limit,
                    )
                except Exception as exc:
                    logger.error("pdf_ingest_failed", filename=pdf_path.name, error=str(exc))
                    ingest_errors.append(pdf_path.name)

            new_count = collection.count()
            logger.info("auto_ingest_complete", chunks_ingested=new_count, errors=ingest_errors)
            if new_count > 0:
                logger.info("rebuilding_components_after_ingest", chunk_count=new_count)
                from src.factory import get_rag_components
                new_orchestrator, new_pipeline = get_rag_components()
                app.state.orchestrator = new_orchestrator
                app.state.pipeline = new_pipeline
                logger.info("components_rebuilt", reason="auto_ingest_populated_db")
        elif chunk_count > 0:
            logger.info("auto_ingest_skipped", reason="db_already_populated", chunk_count=chunk_count)
        else:
            logger.info("auto_ingest_skipped", reason="disabled_by_env_var")
    except Exception:
        logger.exception("auto_ingest_error", note="continuing_with_empty_db")

    logger.info("RAG components initialised, API ready")
    yield
    logger.info("DocuVerse API shutting down")


app = FastAPI(
    title="DocuVerse API",
    description="Production-grade RAG system — PDF upload, Q&A, cited answers.",
    version="0.1.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(corpus.router)

# Middleware applied in LIFO order (last added = outermost):
#   CORS (outermost — handles preflight before rate limiting) → SlowAPI → CostCap → routes
from config.settings import get_settings as _get_settings
_daily_cap = _get_settings().daily_cost_cap_inr
app.add_middleware(DailyCostCapMiddleware, daily_cap_inr=_daily_cap)
app.add_middleware(SlowAPIMiddleware)

_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "https://docuverse-o6kd.vercel.app",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
