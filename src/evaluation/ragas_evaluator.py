"""RAGAS-based evaluation of the DocuVerse RAG pipeline.

RagasEvaluator takes a list of EvalSample objects, runs each question through
the RAGOrchestrator, and evaluates the results with four RAGAS metrics using
a configurable LLM judge. It returns a structured EvalReport without mutating
any RAG component.

RAGAS 0.4.x API notes:
  - evaluate() accepts only old-style Metric subclass instances.
  - LLM is configured via the llm= kwarg on evaluate(); it sets the judge
    on each MetricWithLLM that has llm=None.
  - LangchainLLMWrapper (deprecated in 0.4.x) is still the correct way to
    bridge a LangChain ChatOpenAI to the RAGAS BaseRagasLLM interface.
"""

import math
import os
import time
import warnings

import structlog
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics._answer_relevance import answer_relevancy
from ragas.metrics._context_precision import context_precision
from ragas.metrics._context_recall import context_recall
from ragas.metrics._faithfulness import faithfulness

from config.settings import get_settings
from src.evaluation.dataset import EvalSample
from src.evaluation.report import EvalReport, build_report
from src.orchestrator import RAGOrchestrator

logger = structlog.get_logger(__name__)

# Cost constants for gpt-4o-mini (USD per 1K tokens, as of 2024)
_INPUT_COST_USD_PER_1K = 0.00015
_OUTPUT_COST_USD_PER_1K = 0.0006
_INR_PER_USD = 85.0

# Estimated tokens per RAGAS LLM call (conservative upper bound)
_EST_INPUT_TOKENS_PER_CALL = 1500
_EST_OUTPUT_TOKENS_PER_CALL = 300
_EST_CALLS_PER_SAMPLE = 12  # faithfulness is multi-call


class RagasEvaluator:
    """Evaluates RAG quality using RAGAS metrics with an LLM judge."""

    def __init__(
        self,
        orchestrator: RAGOrchestrator,
        judge_model: str = "gpt-4o-mini",
    ) -> None:
        """Initialise with an orchestrator and optional judge model.

        Args:
            orchestrator: Fully wired RAGOrchestrator to evaluate.
            judge_model: OpenAI model name for RAGAS's LLM judge.
        """
        self._orchestrator = orchestrator
        self._judge_model = judge_model
        self._log = logger.bind(judge_model=judge_model)
        self._log.info("RagasEvaluator initialised")

    # ── Public API ─────────────────────────────────────────────────────────────

    def run_eval(self, samples: list[EvalSample]) -> EvalReport:
        """Run the full evaluation pipeline on a list of samples.

        For each sample:
        1. Calls orchestrator.answer() to get the system's response.
        2. Assembles a RAGAS SingleTurnSample with contexts and reference.
        3. Passes all samples to RAGAS evaluate() with four metrics.
        4. Wraps results in an EvalReport.

        Args:
            samples: Evaluation samples loaded from the dataset.

        Returns:
            EvalReport with per-sample and aggregate metric scores.
        """
        self._log.info("Evaluation started", n_samples=len(samples))
        t_start = time.perf_counter()

        ragas_samples: list[SingleTurnSample] = []
        per_sample_meta: list[dict] = []

        for i, s in enumerate(samples):
            log = self._log.bind(sample_idx=i, category=s.category)
            log.info("Running orchestrator", question=s.question[:80])

            try:
                answer = self._orchestrator.answer(s.question)
                contexts = [rc.chunk.text for rc in answer.retrieved_chunks]
                generated = answer.text
            except Exception as exc:
                log.warning("Orchestrator failed for sample", error=str(exc))
                contexts = []
                generated = ""

            ragas_samples.append(
                SingleTurnSample(
                    user_input=s.question,
                    retrieved_contexts=contexts if contexts else [""],
                    response=generated,
                    reference=s.ground_truth,
                )
            )
            per_sample_meta.append(
                {
                    "question": s.question,
                    "category": s.category,
                    "ground_truth": s.ground_truth,
                    "generated_answer": generated,
                    "retrieved_chunk_count": len(contexts),
                    "scores": {},
                }
            )

        # ── Run RAGAS ──────────────────────────────────────────────────────────
        self._log.info("Running RAGAS evaluate", n_samples=len(ragas_samples))
        settings = get_settings()

        # Ensure the API key is in the environment so RAGAS internal components
        # (e.g. embedding creation for answer_relevancy) can find it.
        os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)

        # LangchainLLMWrapper is deprecated in RAGAS 0.4.x but is still the
        # correct bridge for old-style Metric subclasses until they migrate.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            llm_wrapper = LangchainLLMWrapper(
                ChatOpenAI(
                    model=self._judge_model,
                    temperature=0,
                    api_key=settings.openai_api_key,
                )
            )

        dataset = EvaluationDataset(samples=ragas_samples)
        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

        # Reset per-sample LLM so evaluate() re-applies our custom judge each run
        for m in metrics:
            m.llm = None  # type: ignore[attr-defined]

        try:
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=llm_wrapper,
                raise_exceptions=False,
                show_progress=True,
            )
        except Exception as exc:
            self._log.error("RAGAS evaluate() raised an exception", error=str(exc))
            raise

        # ── Merge per-sample scores ────────────────────────────────────────────
        scores_list = result.scores
        for i, score_dict in enumerate(scores_list):
            per_sample_meta[i]["scores"] = {
                k: (v if not (isinstance(v, float) and math.isnan(v)) else None)
                for k, v in score_dict.items()
            }

        duration = time.perf_counter() - t_start
        cost_inr = self.estimate_cost(samples)["total_inr"]
        self._log.info("Evaluation complete", duration_s=round(duration, 1))

        return build_report(
            version="v1-baseline",
            judge_model=self._judge_model,
            dataset_size=len(samples),
            per_sample_results=per_sample_meta,
            cost_estimate_inr=cost_inr,
            duration_seconds=round(duration, 2),
        )

    def estimate_cost(self, samples: list[EvalSample]) -> dict:
        """Estimate the OpenAI API cost for evaluating the given samples.

        Uses conservative per-call token counts multiplied by the expected
        number of LLM calls RAGAS makes per sample. All figures are estimates.

        Args:
            samples: The evaluation samples to cost.

        Returns:
            Dict with input_usd, output_usd, total_usd, total_inr keys.
        """
        n = len(samples)
        input_tokens = n * _EST_CALLS_PER_SAMPLE * _EST_INPUT_TOKENS_PER_CALL
        output_tokens = n * _EST_CALLS_PER_SAMPLE * _EST_OUTPUT_TOKENS_PER_CALL

        input_usd = (input_tokens / 1000) * _INPUT_COST_USD_PER_1K
        output_usd = (output_tokens / 1000) * _OUTPUT_COST_USD_PER_1K
        total_usd = input_usd + output_usd
        total_inr = total_usd * _INR_PER_USD

        return {
            "n_samples": n,
            "input_tokens_est": input_tokens,
            "output_tokens_est": output_tokens,
            "input_usd": round(input_usd, 4),
            "output_usd": round(output_usd, 4),
            "total_usd": round(total_usd, 4),
            "total_inr": round(total_inr, 2),
        }
