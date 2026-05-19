"""Integration tests against a deployed HuggingFace Spaces instance.

Skipped by default. Enable by setting HF_SPACES_URL, e.g.:
    HF_SPACES_URL=https://varunbommagunta-docuverse.hf.space \
        pytest tests/integration/test_deployed_endpoints.py -v
"""

import os

import pytest
import requests

HF_SPACES_URL = os.getenv("HF_SPACES_URL", "").rstrip("/")

skip_unless_deployed = pytest.mark.skipif(
    not HF_SPACES_URL,
    reason="Set HF_SPACES_URL to run against a deployed HF Spaces instance",
)


@skip_unless_deployed
def test_health_returns_200() -> None:
    """GET /api/health must return 200 {"status": "ok"}."""
    response = requests.get(f"{HF_SPACES_URL}/api/health", timeout=30)
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"


@skip_unless_deployed
def test_query_endpoint_reachable() -> None:
    """POST /api/query with a valid body must return 200 or 503, never a 5xx."""
    response = requests.post(
        f"{HF_SPACES_URL}/api/query",
        json={"query": "What is the purpose of this document?"},
        timeout=60,
    )
    assert response.status_code in (200, 503), (
        f"Expected 200 or 503, got {response.status_code}: {response.text}"
    )


@skip_unless_deployed
def test_rate_limit_triggers_after_threshold() -> None:
    """Sending 25 rapid requests to /api/health must trigger at least one 429."""
    status_codes = []
    for _ in range(25):
        r = requests.get(f"{HF_SPACES_URL}/api/health", timeout=10)
        status_codes.append(r.status_code)

    assert 429 in status_codes, (
        f"Expected a 429 rate-limit response within 25 requests; got: {status_codes}"
    )
