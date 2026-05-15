"""Health-check route.

A single GET /health endpoint used by Docker Compose health checks, load
balancer probes, and the CI smoke-test suite. Returns 200 {"status": "ok"}
when the process is alive. No database ping — that is a readiness check and
belongs on a separate /ready endpoint (Phase 5).
"""

from fastapi import APIRouter

from api.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Liveness probe — confirms the API process is running."""
    return HealthResponse(status="ok")
