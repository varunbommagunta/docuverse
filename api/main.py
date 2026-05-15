"""FastAPI application entry point.

Creates the FastAPI app, registers routes, and wires up the lifespan context
manager for startup/shutdown hooks. All route modules are imported here; the
app object is the single thing Uvicorn imports.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI

from api.routes import health
from src.utils.logger import configure_logging

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan — startup runs before yield, shutdown after."""
    configure_logging()
    logger.info("DocuVerse API starting up")
    yield
    logger.info("DocuVerse API shutting down")


app = FastAPI(
    title="DocuVerse API",
    description="Production-grade RAG system — PDF upload, Q&A, cited answers.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)
