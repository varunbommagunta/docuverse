# DocuVerse Evaluation System

This document explains how DocuVerse measures RAG quality, how to run evaluations,
and how to interpret the results.

---

## What is RAGAS?

[RAGAS](https://docs.ragas.io/) (Retrieval Augmented Generation Assessment) is a framework
for evaluating RAG pipelines using an LLM-as-a-judge approach. It takes as input:

- The user's **question**
- The **retrieved context** chunks returned by the retriever
- The **generated answer** produced by the system
- The **reference (ground truth)** answer

It then uses a judge LLM to score each sample on multiple dimensions.

---

## The Four Metrics

### 1. Faithfulness

**What it measures:** Whether the generated answer is factually grounded in the retrieved
context. A faithful answer contains only claims that can be inferred from the provided chunks.

**How it's computed:** RAGAS decomposes the answer into atomic statements, then checks each
statement against the retrieved context. Score = fraction of statements that are supported.

**Score range:** 0.0 – 1.0. Higher is better.

**Red flag:** A low faithfulness score means the model is hallucinating — generating plausible
but unsupported information. This is the most critical metric for trust.

**Production threshold:** ≥ 0.75

---

### 2. Answer Relevancy

**What it measures:** Whether the generated answer actually addresses the user's question.
A relevant answer stays on-topic and doesn't give irrelevant information.

**How it's computed:** RAGAS asks the judge LLM to generate several questions from the answer,
then measures the semantic similarity between those generated questions and the original
question. A high score means the answer "talks about" what was asked.

**Score range:** 0.0 – 1.0. Higher is better.

**Red flag:** A low score suggests the generator is producing generic or tangential responses
rather than directly answering the question.

**Production threshold:** ≥ 0.70

---

### 3. Context Precision

**What it measures:** Whether the retrieved chunks are relevant to the question (signal-to-noise
ratio). A system with high context precision retrieves mostly useful chunks, not junk.

**How it's computed:** For each retrieved chunk, RAGAS asks the judge whether it is actually
useful for answering the question (given the reference answer). Score = fraction of retrieved
chunks that are useful.

**Score range:** 0.0 – 1.0. Higher is better.

**Red flag:** Low precision means the retriever is returning many irrelevant chunks, which
crowds out useful information and can mislead the generator.

**Production threshold:** ≥ 0.70

---

### 4. Context Recall

**What it measures:** Whether the retrieved chunks contain all the information needed to answer
the question. A system with high context recall doesn't miss important evidence.

**How it's computed:** RAGAS decomposes the reference answer into claims, then checks whether
each claim can be attributed to the retrieved context. Score = fraction of claims supported.

**Score range:** 0.0 – 1.0. Higher is better.

**Red flag:** Low recall means the retriever is not surfacing relevant chunks. The generator
cannot answer what it doesn't see. This is the primary target of Phase 3 hybrid retrieval.

**Production threshold:** ≥ 0.65

---

## Running an Evaluation

### Prerequisites

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set `OPENAI_API_KEY` in `.env`.
3. Ensure the ChromaDB vector store has been populated:
   ```bash
   # The eval script will re-ingest sample.pdf if ChromaDB is empty
   python scripts/evaluate.py --dry-run
   ```

### Dry Run (no API calls)

```bash
python scripts/evaluate.py --dry-run
```

Prints the list of questions that would be evaluated without making any API calls.
Use this to verify the dataset and confirm the setup before spending money.

### Full Evaluation

```bash
python scripts/evaluate.py
```

You will be shown a cost estimate and asked for `y/n` confirmation before any API calls.

### Smoke Test (3 samples)

```bash
python scripts/evaluate.py --limit 3
```

Runs a quick end-to-end test on 3 samples. Costs approximately ₹1.50.
Use this to verify the pipeline works before committing to the full 20-sample run.

### Custom Options

```bash
python scripts/evaluate.py \
  --dataset data/eval/v1_dataset.json \
  --version v2-hybrid-retrieval \
  --output-dir docs/eval_results/ \
  --judge-model gpt-4o-mini \
  --limit 10
```

---

## Dataset Structure

The evaluation dataset lives at `data/eval/v1_dataset.json`. It is a JSON array where each
element has:

| Field | Type | Description |
|-------|------|-------------|
| `question` | string | The user's natural-language question |
| `ground_truth` | string | The correct answer in 1–3 sentences |
| `category` | string | One of the five categories below |

### Categories

| Category | Count | Purpose |
|----------|-------|---------|
| `simple_lookup` | 8 | Single-fact retrieval from one chunk |
| `multi_fact` | 5 | Multiple facts from one or two chunks |
| `cross_chunk` | 3 | Reasoning across two or more chunks |
| `negative` | 2 | Answer not present; system should say so |
| `edge_case` | 2 | Very short or very long queries |

### Adding New Samples

1. Append entries to `data/eval/v1_dataset.json` with the correct fields.
2. Run `python scripts/evaluate.py --dry-run` to verify the dataset loads and validates.
3. Choose a new `--version` tag (e.g. `v2-hybrid`) to distinguish this run from the baseline.

---

## Interpreting Scores

### Baseline Expectations (V1 Dense Retrieval)

| Metric | Expected Range | Meaning if Low |
|--------|---------------|----------------|
| faithfulness | 0.70 – 0.85 | Model is hallucinating |
| answer_relevancy | 0.65 – 0.80 | Answers are off-topic |
| context_precision | 0.60 – 0.80 | Retriever returns irrelevant chunks |
| context_recall | 0.55 – 0.75 | Retriever misses relevant chunks |

### Production Targets

All four metrics should exceed **0.75** before a system is considered production-ready.

### Category Patterns to Watch

- **negative**: faithfulness and context_recall are expected to be lower. If they are high,
  the model may be confabulating answers for unanswerable questions.
- **cross_chunk**: context_recall is typically the weakest metric here because the V1 dense
  retriever may not surface all required chunks.
- **edge_case**: answer_relevancy may dip on very short queries (ambiguous intent).

---

## Limitations of LLM-as-a-Judge

1. **Noise**: LLM judges are not perfectly deterministic. Scores can vary ±0.05 across runs.
2. **Cost**: Each evaluation run makes dozens of OpenAI API calls. Use `--limit` for iteration.
3. **Bias**: The judge LLM (gpt-4o-mini) tends to favour verbose, confident-sounding answers.
4. **Context length**: Very long retrieved chunks may exceed the judge's context window,
   causing it to truncate and score incorrectly.
5. **Ground truth quality**: Metrics like context_recall and context_precision depend on the
   quality of the ground_truth strings. Poor ground truths produce unreliable scores.

Always interpret scores as relative comparisons between versions, not as absolute quality measures.

---

## Output Files

| Location | Contents |
|----------|----------|
| `docs/eval_results/{run_id}.json` | Full structured report (machine-readable) |
| `docs/ITERATION_LOG.md` | Append-only human narrative log |

Reports are committed to the repository so score history is preserved across iterations.
