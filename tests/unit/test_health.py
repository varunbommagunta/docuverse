"""Unit test for the GET /health endpoint.

Uses HTTPX's TestClient (synchronous) so no event loop juggling is required.
The test imports the FastAPI app directly — no network, no Docker.
"""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_returns_ok() -> None:
    """GET /health must return 200 with body {"status": "ok"}."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
