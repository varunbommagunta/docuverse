# DocuVerse — Iteration Log

Tracks the qualitative story of each phase: what we learned, what we changed, and why.
Quantitative evaluation metrics will appear starting Phase 4.

---

## Phase 0 — Foundation (current)

**Goal:** Runnable skeleton. `docker compose up`, `GET /health → 200`, Streamlit shell.

**Completed:**
- Directory scaffold with Ports-and-Adapters layout
- All Protocol interfaces defined (empty bodies — contracts only)
- Pydantic-Settings config with 12-factor `.env` loading
- structlog JSON logging
- FastAPI health endpoint with lifespan startup/shutdown hooks
- Streamlit "Hello DocuVerse" placeholder UI
- Unit test for `/health`
- Docker Compose wiring for api + ui services
- Makefile for common developer tasks

**Decisions made:**
- Used `typing.Protocol` (not ABC) for all interfaces — enables structural subtyping without inheritance coupling.
- Chose `structlog` over the standard `logging` module for first-class JSON and key-value context binding.
- Pinned all deps to minor versions in `requirements.txt` for reproducibility.

---

## Phase 1 — Core RAG Pipeline (current)

**Goal:** End-to-end working RAG: upload PDF → ask question → cited answer.

**Completed:**
- `PyPDFParser` — pypdf-based parser returning `ParsedDocument` with per-page text and metadata
- `RecursiveChunker` — LangChain `RecursiveCharacterTextSplitter`, overlapping chunks with UUID IDs
- `IngestionPipeline` — parse → chunk → embed → store, with per-step timing logs
- `OpenAIEmbedder` — batch embedding via OpenAI API, tenacity retry (3 attempts, exponential backoff)
- `ChromaVectorStore` — persistent on-disk Chroma index, cosine similarity, delete by doc_id
- `DenseRetriever` — composes embedder + vector store for query → top-K retrieval
- `OpenAIGenerator` — gpt-4o-mini with citation system prompt, `[chunk_N]` regex parsing
- `RAGOrchestrator` — real implementation replacing Phase 0 stub
- `src/factory.py` — wires all components from Settings, returns `(orchestrator, pipeline)`
- `POST /ingest` — multipart upload, 25 MB limit, PDF validation, temp file cleanup
- `POST /query` — JSON body, 503 guard when no docs indexed, full citation detail response
- Streamlit UI — PDF upload sidebar, chat interface, expandable source attribution
- 37 unit tests, all passing

**Key decisions:**
- `PyPDFParser.parse()` returns `ParsedDocument` (richer than Protocol's `list[str]`). Protocol drift documented; IngestionPipeline uses concrete types directly. Will reconcile in Phase 2 when Protocol is updated.
- `Answer.citations` changed from `list[str]` (chunk IDs) to `list[int]` (chunk indices) for consistency with the `[chunk_N]` citation format.
- `ChromaVectorStore` uses `hnsw:space: cosine`; similarity = `1 - distance`.
- chromadb stubs have overly strict `embeddings` type hints; suppressed with `# type: ignore[arg-type]`.
- FastAPI `Depends()` in default args flagged by ruff B008 (false positive for FastAPI pattern); B008 added to ruff ignore list.
- `openai==2.24.0`, `chromadb==1.5.2` — both significantly newer than Phase 0 spec assumed.

---

## Phase 2 — Scientific Evaluation (RAGAS Harness)

**Date:** 2026-05-18
**Goal:** Build a reproducible evaluation system measuring DocuVerse's RAG quality on 4 RAGAS metrics; establish a V1 baseline scorecard.

**Completed:**
- `data/eval/v1_dataset.json` — 20-question evaluation dataset covering 5 categories (simple_lookup ×8, multi_fact ×5, cross_chunk ×3, negative ×2, edge_case ×2), all grounded in `sample.pdf`
- `src/evaluation/dataset.py` — `EvalSample` Pydantic model, `load_eval_dataset()`, `validate_dataset()`
- `src/evaluation/ragas_evaluator.py` — `RagasEvaluator` class: orchestrates answer generation, assembles RAGAS `EvaluationDataset`, calls RAGAS `evaluate()` with configurable LLM judge, returns `EvalReport`
- `src/evaluation/report.py` — `EvalReport` with per-sample and aggregate stats, `to_json()`, `to_markdown()`, `print_summary()` with ASCII bar charts
- `scripts/evaluate.py` — CLI tool: dry-run, cost estimate, ChromaDB guard, full eval loop, iteration log append
- `docs/EVALUATION.md` — metric definitions, thresholds, usage guide, limitations
- 14 mocked unit tests; all existing 37 tests continue to pass

**Key decisions:**
- RAGAS 0.4.3 with new `EvaluationDataset` / `SingleTurnSample` API (not the legacy HuggingFace `Dataset` format)
- LLM judge: `gpt-4o-mini` by default — 10× cheaper than gpt-4, sufficient for V1 baseline
- NaN scores (edge cases where RAGAS can't compute a metric) serialise as JSON `null`, excluded from aggregate stats
- Eval script runs in-process (no Docker stack required) for simplicity at V1

**V1 Baseline Scores** *(Run `89192d11`, 2026-05-18, 20 samples, gpt-4o-mini judge)*

| Metric | Mean | Std |
|--------|------|-----|
| faithfulness | **0.892** | ±0.249 |
| answer_relevancy | **0.846** | ±0.287 |
| context_precision | **0.896** | ±0.234 |
| context_recall | **0.942** | ±0.142 |

Category breakdown: simple_lookup 0.976 · multi_fact 0.939 · cross_chunk 0.912 · edge_case 0.925 · negative 0.392

**Interpretation**

- All four metrics are **above the 0.75 production threshold** — V1 dense retrieval on this narrow domain is stronger than predicted.
- **negative (0.392)** is intentionally low: the system correctly refuses unanswerable questions, but RAGAS penalises the text mismatch. Expected and acceptable.
- **answer_relevancy (0.846)** has the highest variance (std=0.287) — negative/edge_case questions drag the mean. Simple/multi-fact score ≈0.99.
- **faithfulness (0.892)** — citation prompt is working; minimal hallucination.
- **context_recall (0.942)** — top-5 retrieval on a 9-chunk corpus surfaces nearly everything. Will likely decrease on larger document sets.

**Next Steps (Phase 3 iteration targets)**
- Improve context_recall via hybrid retrieval (BM25 + dense) — target: +0.10 lift
- Improve faithfulness via cross-encoder reranking — surface the most grounded chunks first
- Add Phase 2 eval to CI: `pytest tests/unit/test_evaluator.py` (mocked, free)
- Monitor: any metric below 0.60 is a priority fix; target >0.75 across all for production

---

## Phase 3 — Generation

**Goal:** GPT-4o-mini with cited answers; end-to-end RAG pipeline live.

---

## Phase 4 — Evaluation

**Goal:** RAGAS harness, baseline scores, regression guard in CI.

---

## Phase 5 — Production Hardening

**Goal:** CI/CD, observability, rate limiting, auth stubs.

---

## Phase 2 Evaluation: v1-baseline

**Date:** 2026-05-18
**Dataset:** 3 questions covering 5 categories (simple_lookup, multi_fact, cross_chunk, negative, edge_case)
**Judge model:** gpt-4o-mini
**Run ID:** `65c9c190-ce90-4164-ab63-e80dcd08c960`

### V1 Baseline Scores

| Metric | Score |
|--------|-------|
| faithfulness | 1.000 |
| answer_relevancy | 0.991 |
| context_precision | 0.900 |
| context_recall | 1.000 |

### Interpretation

- **faithfulness**: Measures whether generated answers are grounded in retrieved context. Low values indicate hallucination.
- **answer_relevancy**: Measures whether the answer actually addresses the question. Low values indicate off-topic responses.
- **context_precision**: Measures whether retrieved chunks are relevant (signal vs noise). Low values indicate poor retrieval ranking.
- **context_recall**: Measures whether retrieved chunks contain all information needed to answer. Low values indicate missing coverage.

### Next Steps

- Phase iteration targets: improve context_recall via hybrid retrieval (BM25 + dense)
- Improve faithfulness via reranking to surface the most grounded chunks
- Monitor: any metric below 0.6 is a priority fix; target >0.75 across all metrics for production


---

## Phase 2 Evaluation: v1-baseline

**Date:** 2026-05-18
**Dataset:** 20 questions covering 5 categories (simple_lookup, multi_fact, cross_chunk, negative, edge_case)
**Judge model:** gpt-4o-mini
**Run ID:** `89192d11-4ebd-47a4-8a60-93e77a31690e`

### V1 Baseline Scores

| Metric | Score |
|--------|-------|
| faithfulness | 0.892 |
| answer_relevancy | 0.846 |
| context_precision | 0.896 |
| context_recall | 0.942 |

### Interpretation

- **faithfulness**: Measures whether generated answers are grounded in retrieved context. Low values indicate hallucination.
- **answer_relevancy**: Measures whether the answer actually addresses the question. Low values indicate off-topic responses.
- **context_precision**: Measures whether retrieved chunks are relevant (signal vs noise). Low values indicate poor retrieval ranking.
- **context_recall**: Measures whether retrieved chunks contain all information needed to answer. Low values indicate missing coverage.

### Next Steps

- Phase iteration targets: improve context_recall via hybrid retrieval (BM25 + dense)
- Improve faithfulness via reranking to surface the most grounded chunks
- Monitor: any metric below 0.6 is a priority fix; target >0.75 across all metrics for production

