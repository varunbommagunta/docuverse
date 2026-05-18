"""Unit tests for Phase 2 evaluation modules.

All tests are fully mocked — no real OpenAI or RAGAS API calls are made.
Tests cover:
- EvalSample loading and validation
- RagasEvaluator dataset assembly
- EvalReport serialisation / deserialisation
- Cost estimation
"""

import json
import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.evaluation.dataset import EvalSample, load_eval_dataset, validate_dataset
from src.evaluation.report import EvalReport, MetricStats, build_report
from src.evaluation.ragas_evaluator import RagasEvaluator
from src.utils.models import Answer, Chunk, RetrievedChunk


# ── Fixtures ───────────────────────────────────────────────────────────────────


def _make_eval_samples(n: int = 3) -> list[EvalSample]:
    categories = ["simple_lookup", "multi_fact", "cross_chunk", "negative", "edge_case"]
    return [
        EvalSample(
            question=f"Question {i}?",
            ground_truth=f"Answer {i}.",
            category=categories[i % len(categories)],
        )
        for i in range(n)
    ]


def _make_retrieved_chunk(text: str = "Jupiter is the largest planet.") -> RetrievedChunk:
    return RetrievedChunk(chunk=Chunk(id="c1", text=text, metadata={}), score=0.9)


def _make_answer(text: str = "Jupiter is the largest planet.") -> Answer:
    return Answer(
        text=text,
        citations=[0],
        retrieved_chunks=[_make_retrieved_chunk()],
    )


@pytest.fixture
def mock_orchestrator() -> MagicMock:
    orc = MagicMock()
    orc.answer.return_value = _make_answer()
    return orc


@pytest.fixture
def eval_samples() -> list[EvalSample]:
    return _make_eval_samples(5)


@pytest.fixture
def sample_dataset_file(tmp_path: Path) -> Path:
    """Write a minimal valid dataset to a temp file."""
    data = [
        {"question": f"Q{i}?", "ground_truth": f"A{i}.", "category": cat}
        for i, cat in enumerate(
            ["simple_lookup", "multi_fact", "cross_chunk", "negative", "edge_case"]
        )
    ]
    p = tmp_path / "dataset.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


# ── Dataset loading tests ──────────────────────────────────────────────────────


def test_load_eval_dataset_returns_eval_samples(sample_dataset_file: Path) -> None:
    samples = load_eval_dataset(str(sample_dataset_file))
    assert len(samples) == 5
    assert all(isinstance(s, EvalSample) for s in samples)


def test_load_eval_dataset_preserves_fields(sample_dataset_file: Path) -> None:
    samples = load_eval_dataset(str(sample_dataset_file))
    assert samples[0].question == "Q0?"
    assert samples[0].ground_truth == "A0."
    assert samples[0].category == "simple_lookup"


def test_load_eval_dataset_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        load_eval_dataset("/nonexistent/path/dataset.json")


def test_validate_dataset_passes_on_valid_samples(eval_samples: list[EvalSample]) -> None:
    validate_dataset(eval_samples)  # should not raise


def test_validate_dataset_raises_on_empty_list() -> None:
    with pytest.raises(ValueError, match="empty"):
        validate_dataset([])


def test_validate_dataset_raises_on_missing_category() -> None:
    samples = [
        EvalSample(question="Q?", ground_truth="A.", category="simple_lookup"),
    ]
    with pytest.raises(ValueError, match="missing"):
        validate_dataset(samples)


def test_validate_dataset_raises_on_blank_question() -> None:
    samples = _make_eval_samples(5)
    samples[0] = EvalSample(question="   ", ground_truth="A.", category="simple_lookup")
    with pytest.raises(ValueError, match="empty"):
        validate_dataset(samples)


# ── RagasEvaluator tests ───────────────────────────────────────────────────────

# Patch targets for all RAGAS internals that make external calls
_RAGAS_PATCHES = [
    "src.evaluation.ragas_evaluator.LangchainLLMWrapper",
    "src.evaluation.ragas_evaluator.ChatOpenAI",
    "src.evaluation.ragas_evaluator.get_settings",
]

from contextlib import ExitStack
from unittest.mock import MagicMock as _MagicMock


def _make_mock_settings() -> _MagicMock:
    m = _MagicMock()
    m.openai_api_key = "sk-test-key"
    m.embedding_model = "text-embedding-3-small"
    return m


def _ragas_patched(evaluate_mock=None):
    """Return an ExitStack that patches all RAGAS internals."""
    stack = ExitStack()
    for target in _RAGAS_PATCHES:
        if target == "src.evaluation.ragas_evaluator.get_settings":
            stack.enter_context(patch(target, return_value=_make_mock_settings()))
        else:
            stack.enter_context(patch(target))
    if evaluate_mock is not None:
        stack.enter_context(patch("src.evaluation.ragas_evaluator.evaluate", return_value=evaluate_mock))
    return stack


def test_ragas_evaluator_calls_orchestrator_for_each_sample(
    mock_orchestrator: MagicMock,
    eval_samples: list[EvalSample],
) -> None:
    """Evaluator must call orchestrator.answer() once per sample."""
    mock_ragas_result = MagicMock()
    mock_ragas_result.scores = [
        {"faithfulness": 0.8, "answer_relevancy": 0.7, "context_precision": 0.9, "context_recall": 0.6}
        for _ in eval_samples
    ]

    with _ragas_patched(evaluate_mock=mock_ragas_result):
        evaluator = RagasEvaluator(orchestrator=mock_orchestrator)
        evaluator.run_eval(eval_samples)

    assert mock_orchestrator.answer.call_count == len(eval_samples)


def test_ragas_evaluator_assembles_correct_ragas_dataset(
    mock_orchestrator: MagicMock,
    eval_samples: list[EvalSample],
) -> None:
    """Evaluator must pass the correct fields to RAGAS EvaluationDataset."""
    captured_dataset = {}
    mock_ragas_result = MagicMock()
    mock_ragas_result.scores = [
        {"faithfulness": 0.8, "answer_relevancy": 0.7, "context_precision": 0.9, "context_recall": 0.6}
        for _ in eval_samples
    ]

    def _fake_evaluate(dataset, **kwargs):
        captured_dataset["dataset"] = dataset
        return mock_ragas_result

    with ExitStack() as stack:
        for target in _RAGAS_PATCHES:
            if target == "src.evaluation.ragas_evaluator.get_settings":
                stack.enter_context(patch(target, return_value=_make_mock_settings()))
            else:
                stack.enter_context(patch(target))
        stack.enter_context(
            patch("src.evaluation.ragas_evaluator.evaluate", side_effect=_fake_evaluate)
        )
        evaluator = RagasEvaluator(orchestrator=mock_orchestrator)
        evaluator.run_eval(eval_samples)

    ds = captured_dataset["dataset"]
    assert len(ds) == len(eval_samples)
    first = ds.samples[0]
    assert first.user_input == eval_samples[0].question
    assert first.reference == eval_samples[0].ground_truth
    assert isinstance(first.retrieved_contexts, list)
    assert len(first.retrieved_contexts) > 0


def test_ragas_evaluator_handles_orchestrator_failure_gracefully(
    eval_samples: list[EvalSample],
) -> None:
    """If orchestrator raises, the sample should have empty context, not crash."""
    failing_orchestrator = MagicMock()
    failing_orchestrator.answer.side_effect = RuntimeError("retrieval error")

    mock_ragas_result = MagicMock()
    mock_ragas_result.scores = [
        {"faithfulness": float("nan"), "answer_relevancy": float("nan"),
         "context_precision": float("nan"), "context_recall": float("nan")}
        for _ in eval_samples
    ]

    with _ragas_patched(evaluate_mock=mock_ragas_result):
        evaluator = RagasEvaluator(orchestrator=failing_orchestrator)
        report = evaluator.run_eval(eval_samples)

    assert report.dataset_size == len(eval_samples)


def test_ragas_evaluator_returns_eval_report(
    mock_orchestrator: MagicMock,
    eval_samples: list[EvalSample],
) -> None:
    mock_ragas_result = MagicMock()
    mock_ragas_result.scores = [
        {"faithfulness": 0.85, "answer_relevancy": 0.75, "context_precision": 0.90, "context_recall": 0.65}
        for _ in eval_samples
    ]

    with _ragas_patched(evaluate_mock=mock_ragas_result):
        evaluator = RagasEvaluator(orchestrator=mock_orchestrator)
        report = evaluator.run_eval(eval_samples)

    assert isinstance(report, EvalReport)
    assert report.dataset_size == len(eval_samples)
    assert "faithfulness" in report.metrics


# ── Cost estimation tests ──────────────────────────────────────────────────────


def test_estimate_cost_returns_reasonable_value_for_20_samples() -> None:
    evaluator = RagasEvaluator.__new__(RagasEvaluator)
    evaluator._judge_model = "gpt-4o-mini"
    evaluator._log = MagicMock()

    samples = _make_eval_samples(20)
    cost = evaluator.estimate_cost(samples)

    assert cost["n_samples"] == 20
    assert cost["total_inr"] > 0
    assert cost["total_inr"] < 200, "Cost estimate suspiciously high"
    assert cost["total_usd"] > 0
    assert "input_tokens_est" in cost
    assert "output_tokens_est" in cost


def test_estimate_cost_scales_linearly() -> None:
    evaluator = RagasEvaluator.__new__(RagasEvaluator)
    evaluator._judge_model = "gpt-4o-mini"
    evaluator._log = MagicMock()

    cost_5 = evaluator.estimate_cost(_make_eval_samples(5))
    cost_10 = evaluator.estimate_cost(_make_eval_samples(10))
    assert abs(cost_10["total_inr"] / cost_5["total_inr"] - 2.0) < 0.01


# ── EvalReport serialisation tests ────────────────────────────────────────────


def test_eval_report_to_json_creates_file(tmp_path: Path) -> None:
    report = build_report(
        version="test-v1",
        judge_model="gpt-4o-mini",
        dataset_size=5,
        per_sample_results=[
            {
                "question": "Q?",
                "category": "simple_lookup",
                "ground_truth": "A.",
                "generated_answer": "A.",
                "retrieved_chunk_count": 1,
                "scores": {"faithfulness": 0.9, "answer_relevancy": 0.8,
                           "context_precision": 0.7, "context_recall": 0.6},
            }
        ],
        cost_estimate_inr=5.0,
        duration_seconds=10.0,
    )
    path = tmp_path / "report.json"
    report.to_json(str(path))
    assert path.exists()


def test_eval_report_json_is_valid_json(tmp_path: Path) -> None:
    report = build_report(
        version="test-v1",
        judge_model="gpt-4o-mini",
        dataset_size=2,
        per_sample_results=[
            {
                "question": "Q?",
                "category": "simple_lookup",
                "ground_truth": "A.",
                "generated_answer": "A.",
                "retrieved_chunk_count": 1,
                "scores": {"faithfulness": 0.9},
            }
        ],
        cost_estimate_inr=1.0,
        duration_seconds=5.0,
    )
    path = tmp_path / "r.json"
    report.to_json(str(path))
    parsed = json.loads(path.read_text())
    assert parsed["version"] == "test-v1"
    assert parsed["dataset_size"] == 2


def test_eval_report_nan_scores_serialise_as_null(tmp_path: Path) -> None:
    report = build_report(
        version="test-v1",
        judge_model="gpt-4o-mini",
        dataset_size=1,
        per_sample_results=[
            {
                "question": "Q?",
                "category": "negative",
                "ground_truth": "This information is not in the provided documents.",
                "generated_answer": "",
                "retrieved_chunk_count": 0,
                "scores": {"faithfulness": float("nan")},
            }
        ],
        cost_estimate_inr=1.0,
        duration_seconds=5.0,
    )
    path = tmp_path / "r.json"
    report.to_json(str(path))
    text = path.read_text()
    assert "NaN" not in text  # JSON spec disallows NaN


def test_eval_report_round_trips_correctly(tmp_path: Path) -> None:
    report = build_report(
        version="round-trip-v1",
        judge_model="gpt-4o-mini",
        dataset_size=3,
        per_sample_results=[
            {
                "question": "Q?",
                "category": "simple_lookup",
                "ground_truth": "A.",
                "generated_answer": "A.",
                "retrieved_chunk_count": 2,
                "scores": {"faithfulness": 0.75, "answer_relevancy": 0.80,
                           "context_precision": 0.85, "context_recall": 0.70},
            }
        ],
        cost_estimate_inr=3.5,
        duration_seconds=15.0,
    )
    path = tmp_path / "r.json"
    report.to_json(str(path))
    parsed = json.loads(path.read_text())

    assert parsed["version"] == "round-trip-v1"
    assert parsed["dataset_size"] == 3
    assert "faithfulness" in parsed["metrics"]
    assert parsed["metrics"]["faithfulness"]["mean"] == pytest.approx(0.75, abs=1e-3)


def test_build_report_computes_correct_means() -> None:
    samples = [
        {"question": "Q?", "category": "simple_lookup", "ground_truth": "A.",
         "generated_answer": "A.", "retrieved_chunk_count": 1,
         "scores": {"faithfulness": 0.8, "answer_relevancy": 0.6,
                    "context_precision": 0.9, "context_recall": 0.7}},
        {"question": "Q2?", "category": "multi_fact", "ground_truth": "A.",
         "generated_answer": "A.", "retrieved_chunk_count": 2,
         "scores": {"faithfulness": 0.4, "answer_relevancy": 0.4,
                    "context_precision": 0.5, "context_recall": 0.3}},
    ]
    report = build_report(
        version="v1",
        judge_model="gpt-4o-mini",
        dataset_size=2,
        per_sample_results=samples,
        cost_estimate_inr=2.0,
        duration_seconds=8.0,
    )
    assert report.metrics["faithfulness"].mean == pytest.approx(0.6, abs=1e-3)
    assert report.metrics["answer_relevancy"].mean == pytest.approx(0.5, abs=1e-3)
