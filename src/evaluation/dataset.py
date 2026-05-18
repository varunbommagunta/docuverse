"""Evaluation dataset loading and validation.

Provides the EvalSample domain model and utilities for loading the JSON
evaluation dataset from disk. The dataset is the authoritative record of
what the system is expected to answer correctly.
"""

import json
from typing import Literal

import structlog
from pydantic import BaseModel

logger = structlog.get_logger(__name__)

Category = Literal["simple_lookup", "multi_fact", "cross_chunk", "negative", "edge_case"]

REQUIRED_CATEGORIES: set[str] = {
    "simple_lookup",
    "multi_fact",
    "cross_chunk",
    "negative",
    "edge_case",
}


class EvalSample(BaseModel):
    """A single evaluation question with its expected answer and category."""

    question: str
    ground_truth: str
    category: Category


def load_eval_dataset(path: str) -> list[EvalSample]:
    """Load an evaluation dataset from a JSON file.

    Args:
        path: Path to a JSON file containing an array of sample objects.

    Returns:
        List of validated EvalSample objects.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If any sample is malformed.
    """
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)

    samples = [EvalSample(**item) for item in raw]
    logger.info("Dataset loaded", path=path, count=len(samples))
    return samples


def validate_dataset(samples: list[EvalSample]) -> None:
    """Validate a list of EvalSample objects for completeness and coverage.

    Checks:
    - No empty fields on any sample.
    - At least one sample per recognised category.

    Args:
        samples: Loaded evaluation samples.

    Raises:
        ValueError: On validation failure.
    """
    if not samples:
        raise ValueError("Dataset is empty.")

    for i, s in enumerate(samples):
        if not s.question.strip():
            raise ValueError(f"Sample {i} has an empty 'question'.")
        if not s.ground_truth.strip():
            raise ValueError(f"Sample {i} has an empty 'ground_truth'.")
        if not s.category:
            raise ValueError(f"Sample {i} has no 'category'.")

    found_categories = {s.category for s in samples}
    missing = REQUIRED_CATEGORIES - found_categories
    if missing:
        raise ValueError(f"Dataset is missing samples for categories: {missing}")

    category_counts = {}
    for s in samples:
        category_counts[s.category] = category_counts.get(s.category, 0) + 1

    logger.info("Dataset validated", total=len(samples), by_category=category_counts)
