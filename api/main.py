"""FastAPI application entry point.

Creates the FastAPI app, registers routes, and wires up the lifespan context
manager for startup/shutdown hooks. The orchestrator and ingestion pipeline are
constructed once at startup and stored on app.state for injection into routes.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from api.routes import health, ingest, query
from src.utils.logger import configure_logging

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan — startup runs before yield, shutdown after."""
    configure_logging()
    logger.info("DocuVerse API starting up")

    # Build all RAG components and store on app.state for dependency injection
    from src.factory import get_rag_components
    orchestrator, pipeline = get_rag_components()
    app.state.orchestrator = orchestrator
    app.state.pipeline = pipeline

    logger.info("RAG components initialised, API ready")
    yield
    logger.info("DocuVerse API shutting down")


app = FastAPI(
    title="DocuVerse API",
    description="Production-grade RAG system — PDF upload, Q&A, cited answers.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(query.router)
