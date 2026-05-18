"""Evaluation report model and rendering.

EvalReport is the structured output of a RAGAS evaluation run. It stores
aggregate metric statistics, per-sample scores, run metadata, and cost
information. Supports JSON serialisation and human-readable terminal output.
"""

import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")

INR_PER_USD = 85.0


class MetricStats(BaseModel):
    """Aggregate statistics for one RAGAS metric across all samples."""

    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    n_valid: int = 0


class EvalReport(BaseModel):
    """Full results of one RAGAS evaluation run."""

    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    version: str
    judge_model: str
    dataset_size: int
    metrics: dict[str, MetricStats]
    per_sample_results: list[dict[str, Any]]
    cost_estimate_inr: float
    duration_seconds: float

    # ── Serialisation ──────────────────────────────────────────────────────────

    def to_json(self, path: str) -> None:
        """Save this report as pretty-printed JSON.

        Args:
            path: Full file path to write. Parent directories are created if needed.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        def _clean(obj: Any) -> Any:
            if isinstance(obj, float) and math.isnan(obj):
                return None
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_clean(v) for v in obj]
            return obj

        raw = self.model_dump(mode="json")
        raw = _clean(raw)
        out.write_text(json.dumps(raw, indent=2, default=str), encoding="utf-8")
        logger.info("Report saved", path=str(out))

    def to_markdown(self) -> str:
        """Return a human-readable Markdown summary of the report."""
        ts = self.timestamp.strftime("%Y-%m-%d %H:%M UTC")
        lines = [
            f"# DocuVerse Evaluation Report — {self.version}",
            "",
            f"| Field | Value |",
            f"|-------|-------|",
            f"| Run ID | `{self.run_id}` |",
            f"| Timestamp | {ts} |",
            f"| Judge model | {self.judge_model} |",
            f"| Dataset size | {self.dataset_size} |",
            f"| Duration | {self.duration_seconds:.1f}s |",
            f"| Cost estimate | ₹{self.cost_estimate_inr:.2f} |",
            "",
            "## Metric Scores",
            "",
            "| Metric | Mean | Std | Min | Max | N Valid |",
            "|--------|------|-----|-----|-----|---------|",
        ]
        for name in METRIC_NAMES:
            s = self.metrics.get(name, MetricStats())
            mean = f"{s.mean:.3f}" if s.mean is not None else "N/A"
            std = f"{s.std:.3f}" if s.std is not None else "N/A"
            mn = f"{s.min:.3f}" if s.min is not None else "N/A"
            mx = f"{s.max:.3f}" if s.max is not None else "N/A"
            lines.append(f"| {name} | {mean} | {std} | {mn} | {mx} | {s.n_valid} |")

        lines += [
            "",
            "## Per-Sample Results",
            "",
            "| # | Category | Question | F | AR | CP | CR |",
            "|---|----------|----------|---|----|----|-----|",
        ]
        for i, r in enumerate(self.per_sample_results, 1):
            q = r["question"][:55] + "..." if len(r["question"]) > 55 else r["question"]
            sc = r.get("scores", {})

            def _fmt(v: Any) -> str:
                return f"{v:.2f}" if isinstance(v, float) and not math.isnan(v) else "—"

            lines.append(
                f"| {i} | {r['category']} | {q} "
                f"| {_fmt(sc.get('faithfulness'))} "
                f"| {_fmt(sc.get('answer_relevancy'))} "
                f"| {_fmt(sc.get('context_precision'))} "
                f"| {_fmt(sc.get('context_recall'))} |"
            )

        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print a pretty terminal summary with ASCII bar charts."""
        bar_width = 20
        ts = self.timestamp.strftime("%Y-%m-%d %H:%M UTC")

        print()
        print("=" * 60)
        print(f"  DocuVerse Evaluation: {self.version}")
        print(f"  Run ID : {self.run_id[:8]}...")
        print(f"  Date   : {ts}")
        print(f"  Samples: {self.dataset_size}   Duration: {self.duration_seconds:.1f}s")
        print(f"  Judge  : {self.judge_model}")
        print("=" * 60)
        print()
        print("  Metric Scores")
        print("  " + "-" * 56)

        for name in METRIC_NAMES:
            s = self.metrics.get(name, MetricStats())
            if s.mean is not None:
                filled = round(s.mean * bar_width)
                bar = "#" * filled + "." * (bar_width - filled)
                std_str = f"+/-{s.std:.3f}" if s.std is not None else ""
                print(f"  {name:<22} [{bar}]  {s.mean:.3f} {std_str}")
            else:
                print(f"  {name:<22} [{'?'*bar_width}]  N/A")

        print()
        print("  Category breakdown:")
        category_means: dict[str, list[float]] = {}
        for r in self.per_sample_results:
            cat = r.get("category", "unknown")
            scores = r.get("scores", {})
            vals = [v for v in scores.values() if isinstance(v, float) and not math.isnan(v)]
            if vals:
                avg = sum(vals) / len(vals)
                category_means.setdefault(cat, []).append(avg)

        for cat, avgs in sorted(category_means.items()):
            avg = sum(avgs) / len(avgs)
            print(f"    {cat:<18} {avg:.3f}  (n={len(avgs)})")

        print()
        print(f"  Cost estimate: Rs. {self.cost_estimate_inr:.2f} INR")
        print("=" * 60)
        print()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _compute_stats(values: list[float | None]) -> MetricStats:
    """Compute mean/std/min/max from a list of possibly-NaN floats."""
    clean = [v for v in values if v is not None and isinstance(v, float) and not math.isnan(v)]
    if not clean:
        return MetricStats(n_valid=0)
    n = len(clean)
    mean = sum(clean) / n
    variance = sum((x - mean) ** 2 for x in clean) / n if n > 1 else 0.0
    return MetricStats(
        mean=round(mean, 4),
        std=round(math.sqrt(variance), 4),
        min=round(min(clean), 4),
        max=round(max(clean), 4),
        n_valid=n,
    )


def build_report(
    *,
    version: str,
    judge_model: str,
    dataset_size: int,
    per_sample_results: list[dict[str, Any]],
    cost_estimate_inr: float,
    duration_seconds: float,
) -> EvalReport:
    """Construct an EvalReport from raw per-sample results.

    Args:
        version: Human-readable run version string.
        judge_model: Name of the LLM judge used.
        dataset_size: Total number of samples evaluated.
        per_sample_results: List of dicts with 'scores' dict per sample.
        cost_estimate_inr: Estimated API cost in INR.
        duration_seconds: Wall-clock seconds the evaluation took.

    Returns:
        Fully populated EvalReport.
    """
    metric_values: dict[str, list[float | None]] = {m: [] for m in METRIC_NAMES}
    for r in per_sample_results:
        sc = r.get("scores", {})
        for m in METRIC_NAMES:
            metric_values[m].append(sc.get(m))

    metrics = {m: _compute_stats(metric_values[m]) for m in METRIC_NAMES}

    return EvalReport(
        version=version,
        judge_model=judge_model,
        dataset_size=dataset_size,
        metrics=metrics,
        per_sample_results=per_sample_results,
        cost_estimate_inr=cost_estimate_inr,
        duration_seconds=duration_seconds,
    )
