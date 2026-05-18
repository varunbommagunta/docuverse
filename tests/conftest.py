"""Shared pytest fixtures and test configuration.

Generates the sample PDF on first run if it does not exist, so unit tests
that exercise the real parser have a fixture document available.
"""

import os
import subprocess
import sys

import pytest

SAMPLE_PDF = os.path.join(os.path.dirname(__file__), "..", "data", "sample", "sample.pdf")


def _ensure_sample_pdf() -> None:
    """Generate sample.pdf if it doesn't exist, using the generator script."""
    if os.path.exists(SAMPLE_PDF):
        return
    script = os.path.join(os.path.dirname(__file__), "..", "scripts", "generate_sample_pdf.py")
    result = subprocess.run([sys.executable, script], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to generate sample PDF:\n{result.stderr}")


@pytest.fixture(scope="session", autouse=True)
def ensure_sample_pdf() -> None:
    """Session-scoped fixture: generate sample.pdf before any test runs."""
    _ensure_sample_pdf()


@pytest.fixture
def sample_pdf_path() -> str:
    """Return the absolute path to the sample PDF fixture."""
    return os.path.abspath(SAMPLE_PDF)
