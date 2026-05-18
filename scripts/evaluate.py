#!/usr/bin/env python
"""DocuVerse evaluation CLI.

Runs the RAGAS evaluation harness against a dataset of ground-truth Q&A pairs,
saves a structured JSON report, and appends a summary to the iteration log.

Usage
-----
    python scripts/evaluate.py                      # full run, prompts for confirmation
    python scripts/evaluate.py --dry-run            # show plan, no API calls
    python scripts/evaluate.py --dataset data/eval/v1_dataset.json --version v1-baseline
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure the project root is on sys.path so src/ imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog

from src.evaluation.dataset import load_eval_dataset, validate_dataset
from src.evaluation.ragas_evaluator import RagasEvaluator
from src.factory import get_rag_components

logger = structlog.get_logger(__name__)

_BANNER = """
==============================================================
  DocuVerse Evaluation Run -- {version}
==============================================================
"""

_ITERATION_LOG = "docs/ITERATION_LOG.md"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DocuVerse RAGAS evaluation harness")
    p.add_argument(
        "--dataset",
        default="data/eval/v1_dataset.json",
        help="Path to the evaluation dataset JSON.",
    )
    p.add_argument(
        "--version",
        default="v1-baseline",
        help="Human-readable version tag for this run.",
    )
    p.add_argument(
        "--output-dir",
        default="docs/eval_results",
        help="Directory to write the JSON report.",
    )
    p.add_argument(
        "--judge-model",
        default="gpt-4o-mini",
        help="OpenAI model used as the RAGAS judge.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the evaluation plan without making any API calls.",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt and proceed automatically.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N samples (for smoke tests).",
    )
    return p.parse_args()


def _ensure_chroma_populated(pipeline) -> None:
    """Re-ingest sample.pdf if the vector store is empty."""
    from src.retrieval.vector_store import ChromaVectorStore
    from config.settings import get_settings

    settings = get_settings()
    vs = ChromaVectorStore(persist_directory=settings.chroma_persist_directory)
    count = vs.count()
    if count == 0:
        logger.warning("ChromaDB is empty — re-ingesting sample.pdf")
        sample_pdf = "data/sample/sample.pdf"
        if not os.path.exists(sample_pdf):
            raise FileNotFoundError(f"Sample PDF not found at {sample_pdf}")
        pipeline.ingest(sample_pdf, filename="sample.pdf")
        logger.info("Re-ingestion complete", chunks=vs.count())
    else:
        logger.info("ChromaDB ready", chunk_count=count)


def _append_iteration_log(report, version: str) -> None:
    """Append a summary block to ITERATION_LOG.md."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    m = report.metrics

    def _fmt(stats) -> str:
        if stats and stats.mean is not None:
            return f"{stats.mean:.3f}"
        return "N/A"

    block = f"""
---

## Phase 2 Evaluation: {version}

**Date:** {ts}
**Dataset:** {report.dataset_size} questions covering 5 categories (simple_lookup, multi_fact, cross_chunk, negative, edge_case)
**Judge model:** {report.judge_model}
**Run ID:** `{report.run_id}`

### V1 Baseline Scores

| Metric | Score |
|--------|-------|
| faithfulness | {_fmt(m.get('faithfulness'))} |
| answer_relevancy | {_fmt(m.get('answer_relevancy'))} |
| context_precision | {_fmt(m.get('context_precision'))} |
| context_recall | {_fmt(m.get('context_recall'))} |

### Interpretation

- **faithfulness**: Measures whether generated answers are grounded in retrieved context. Low values indicate hallucination.
- **answer_relevancy**: Measures whether the answer actually addresses the question. Low values indicate off-topic responses.
- **context_precision**: Measures whether retrieved chunks are relevant (signal vs noise). Low values indicate poor retrieval ranking.
- **context_recall**: Measures whether retrieved chunks contain all information needed to answer. Low values indicate missing coverage.

### Next Steps

- Phase iteration targets: improve context_recall via hybrid retrieval (BM25 + dense)
- Improve faithfulness via reranking to surface the most grounded chunks
- Monitor: any metric below 0.6 is a priority fix; target >0.75 across all metrics for production

"""
    log_path = Path(_ITERATION_LOG)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(block)
    logger.info("Iteration log updated", path=str(log_path))


def main() -> None:
    args = _parse_args()

    print(_BANNER.format(version=args.version))

    # ── 1. Load and validate dataset ───────────────────────────────────────────
    print(f"  Loading dataset: {args.dataset}")
    samples = load_eval_dataset(args.dataset)
    validate_dataset(samples)

    if args.limit:
        samples = samples[: args.limit]
        print(f"  [!] Limiting to {args.limit} sample(s) for smoke test.")

    print(f"  Dataset: {len(samples)} samples")

    by_cat: dict[str, int] = {}
    for s in samples:
        by_cat[s.category] = by_cat.get(s.category, 0) + 1
    for cat, n in sorted(by_cat.items()):
        print(f"    {cat:<20} {n}")

    if args.dry_run:
        print()
        print("  [DRY RUN] Questions to be evaluated:")
        for i, s in enumerate(samples, 1):
            q = s.question[:75] + "..." if len(s.question) > 75 else s.question
            print(f"    {i:>2}. [{s.category}] {q}")
        print()
        print("  No API calls made. Remove --dry-run to execute.")
        return

    # ── 2. Cost estimate ───────────────────────────────────────────────────────
    # Build a temporary evaluator just for the cost estimate (no orchestrator needed)
    from src.evaluation.ragas_evaluator import RagasEvaluator as _RE  # local import to avoid cold load

    cost = _RE.__new__(_RE)
    cost._judge_model = args.judge_model
    cost_info = cost.estimate_cost(samples)

    print()
    print(f"  Cost estimate for {len(samples)} samples with {args.judge_model}:")
    print(f"    Input tokens (est.):  {cost_info['input_tokens_est']:,}")
    print(f"    Output tokens (est.): {cost_info['output_tokens_est']:,}")
    print(f"    Total USD:            ${cost_info['total_usd']:.4f}")
    print(f"    Total INR:            Rs. {cost_info['total_inr']:.2f}")
    print()

    if args.yes:
        print("  Auto-confirmed (--yes).")
    else:
        answer = input("  Proceed with evaluation? [y/N] ").strip().lower()
        if answer != "y":
            print("  Aborted.")
            return

    # ── 3. Build orchestrator and verify ChromaDB ──────────────────────────────
    print()
    print("  Building RAG components…")
    orchestrator, pipeline = get_rag_components()
    _ensure_chroma_populated(pipeline)

    # ── 4. Run evaluation ──────────────────────────────────────────────────────
    print("  Starting RAGAS evaluation…")
    evaluator = RagasEvaluator(orchestrator=orchestrator, judge_model=args.judge_model)
    report = evaluator.run_eval(samples)

    # Override version from CLI
    report = report.model_copy(update={"version": args.version})

    # ── 5. Save report ─────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{report.run_id}.json"
    report.to_json(str(report_path))
    print(f"\n  Report saved: {report_path}")

    # ── 6. Append to iteration log ─────────────────────────────────────────────
    _append_iteration_log(report, args.version)

    # ── 7. Print summary ───────────────────────────────────────────────────────
    report.print_summary()


if __name__ == "__main__":
    main()
