"""Pydantic models for API request and response payloads.

These are HTTP-layer-only models. Do NOT import domain models (Chunk, Answer,
etc.) from src/ here and re-export them — keep the HTTP schema separate so the
domain model can evolve independently of the wire format.
"""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str
