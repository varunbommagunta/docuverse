"""FastAPI application entry point.

Creates the FastAPI app, registers routes, and wires up the lifespan context
manager for startup/shutdown hooks. The orchestrator and ingestion pipeline are
constructed once at startup and stored on app.state for injection into routes.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

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

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan — startup runs before yield, shutdown after."""
    configure_logging()
    logger.info("DocuVerse API starting up")

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
    "https://docuverse-eta.vercel.app",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
