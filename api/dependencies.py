"""FastAPI dependency-injection helpers.

These functions are used with FastAPI's Depends() to inject app-level
singletons (orchestrator, pipeline) into route handlers without coupling
routes to the factory module directly.
"""

from fastapi import Request

from src.ingestion.pipeline import IngestionPipeline
from src.orchestrator import RAGOrchestrator


def get_orchestrator(request: Request) -> RAGOrchestrator:
    """Return the RAGOrchestrator stored on app.state."""
    return request.app.state.orchestrator  # type: ignore[no-any-return]


def get_pipeline(request: Request) -> IngestionPipeline:
    """Return the IngestionPipeline stored on app.state."""
    return request.app.state.pipeline  # type: ignore[no-any-return]
