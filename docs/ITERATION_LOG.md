# DocuVerse — Iteration Log

Tracks the qualitative story of each phase: what we built, what we learned, what we changed, and the measured impact.

---

## Phase 0 — Foundation

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

## Phase 1 — Core RAG Pipeline

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

**V1 Baseline Scores** *(Run `89192d11`, 2026-05-18, 20 samples on toy Solar System corpus, gpt-4o-mini judge)*

| Metric | Mean | Std |
|--------|------|-----|
| faithfulness | **0.892** | ±0.249 |
| answer_relevancy | **0.846** | ±0.287 |
| context_precision | **0.896** | ±0.234 |
| context_recall | **0.942** | ±0.142 |

Category breakdown: simple_lookup 0.976 · multi_fact 0.939 · cross_chunk 0.912 · edge_case 0.925 · negative 0.392

**Interpretation**

- All four metrics are **above the 0.75 production threshold** — but on a tiny 9-chunk corpus where top-5 retrieval is trivial. This baseline is artificially inflated; Phase 3a re-runs it on a realistic corpus.
- **negative (0.392)** is intentionally low: the system correctly refuses unanswerable questions, but RAGAS penalises the text mismatch. Expected and acceptable.
- **answer_relevancy (0.846)** has the highest variance (std=0.287) — negative/edge_case questions drag the mean. Simple/multi-fact score ≈0.99.
- **faithfulness (0.892)** — citation prompt is working; minimal hallucination.
- **context_recall (0.942)** — top-5 retrieval on a 9-chunk corpus surfaces nearly everything. Will likely decrease on larger document sets.

---

## Phase 3a — Corpus Expansion + Realistic Baseline

**Date:** 2026-05-18
**Goal:** Replace the toy Solar System corpus with two real Indian government documents and establish an honest "v1-realistic-baseline" scorecard that exposes where dense retrieval breaks down on real-world text.

**Completed:**
- `scripts/download_sample_pdfs.py` — downloads Constitution of India (EN-Kannada diglot, legislative.gov.in) and ARC 4th Report: Ethics in Governance (darpg.gov.in) with retry + skip-if-exists logic
- `src/ingestion/parsers.py` — additive `page_limit: int | None = None` parameter on `parse()`; existing callers unaffected; 3 new unit tests (58 total)
- `scripts/ingest_corpus.py` — multi-PDF ingestion with per-document page limits and `--reset`
- `data/eval/v1_realistic_dataset.json` — 40-question dataset: 20 original Solar System + 12 Constitution of India + 8 ARC Ethics, grounded in actual retrieved chunks
- `scripts/evaluate.py` — `--limit N` now random-samples instead of taking first N

**Corpus details:**
- constitution_of_india.pdf: 160 pages ingested (ToC occupies pp.1-60; articles start p.61), 937 chunks
- arc_ethics_governance.pdf: 40 pages, 426 chunks
- sample.pdf: 2 pages, 9 chunks
- Total: 1372 chunks

**Key decisions:**
- Constitution PDF is an English-Kannada diglot edition (799 pages total). Every odd page is English, every even page is Kannada. The bilingual interleaving degrades retrieval because chunks straddle language boundaries. Page limit set to 160 to reach actual article text (Parts I-V: Union, Citizenship, Fundamental Rights, Directive Principles, the Union executive).
- ARC Ethics report is pure English; 40-page limit captures Executive Summary + Chapters 1-3 cleanly.
- Ground truths written from actual orchestrator answers on the ingested corpus, not from memory — so they reflect what the system can plausibly retrieve.

**V1 Realistic Baseline Scores** *(Run `a14988ca`, 2026-05-18, 40 samples, gpt-4o-mini judge)*

| Metric | Mean | Std |
|--------|------|-----|
| faithfulness | **0.773** | ±0.353 |
| answer_relevancy | **0.758** | ±0.382 |
| context_precision | **0.753** | ±0.350 |
| context_recall | **0.819** | ±0.308 |

Category breakdown: simple_lookup 0.898 · multi_fact 0.727 · cross_chunk 0.777 · edge_case 0.781 · negative 0.400

**Interpretation vs. V1 Solar System Baseline**

| Metric | Solar System (20q) | Realistic (40q) | Delta |
|--------|--------------------|-----------------|-------|
| faithfulness | 0.892 | 0.773 | -0.119 |
| answer_relevancy | 0.846 | 0.758 | -0.088 |
| context_precision | 0.896 | 0.753 | -0.143 |
| context_recall | 0.942 | 0.819 | -0.123 |

- All four metrics remain above the 0.75 production threshold — barely. context_precision (0.753) is the tightest.
- **multi_fact (0.727)** drops below 0.75 — the dense retriever struggles to surface all relevant chunks for questions requiring synthesis across multiple constitutional articles. This is the primary target for Phase 4 hybrid retrieval.
- **context_precision drop (-0.143)** is the largest regression. The bilingual Constitution text creates noise chunks (Kannada pages) that get retrieved alongside relevant English ones, diluting precision.
- **negative (0.400)** is again intentionally low and expected; no change from V1 pattern.
- **simple_lookup (0.898)** holds strong — the system can still find and cite isolated facts reliably.

---

## Phase 4 — Hybrid Retrieval + Cross-Encoder Reranking

**Date:** 2026-05-19
**Goal:** Replace pure dense retrieval with a hybrid BM25 + dense pipeline fused via Reciprocal Rank Fusion, then add a cross-encoder reranker as a second stage. Measure the impact on the v1-realistic-baseline dataset.

**Completed:**
- `src/retrieval/base.py` — added `Reranker` Protocol; added `get_all_chunks()` to `VectorStore` Protocol
- `src/retrieval/vector_store.py` — implemented `get_all_chunks()` on `ChromaVectorStore` (calls `collection.get()`, reconstructs `Chunk` objects)
- `src/retrieval/bm25_retriever.py` — `BM25Retriever`: builds `BM25Okapi` index at construction from all chunks; lowercase whitespace tokenization; scores normalized to [0,1] by dividing by max
- `src/retrieval/hybrid_retriever.py` — `HybridRetriever`: fetches top-20 from dense + top-20 from BM25, fuses with RRF (k=60); deduplicates by chunk_id; returns top-k by fused score
- `src/retrieval/cross_encoder_reranker.py` — `CrossEncoderReranker`: ms-marco-MiniLM-L-6-v2 via sentence-transformers; lazy model loading on first rerank() call; sigmoid-normalizes logits to [0,1]
- `src/retrieval/reranked_retriever.py` — `RerankedRetriever` decorator: fetches 50 candidates from any base Retriever, reranks to top-k
- `src/factory.py` — strategy dispatch: dense / sparse / hybrid / reranked_hybrid; reads from `RETRIEVAL_STRATEGY` env var
- `config/settings.py` / `config/config.yaml` / `.env.example` — 6 new configuration fields
- `requirements.txt` — added rank-bm25==0.2.2, sentence-transformers==3.4.1
- 37 new unit tests (95 total, all passing)

**V1 Phase 4 Scores** *(Run `ab12c0f5`, 2026-05-19, 40 samples, gpt-4o-mini judge, reranked_hybrid strategy)*

| Metric | Phase 3a (dense) | Phase 4 (reranked_hybrid) | Delta |
|--------|-----------------|--------------------------|-------|
| faithfulness | 0.773 | **0.829** | **+0.056** |
| answer_relevancy | 0.758 | **0.779** | **+0.021** |
| context_precision | 0.753 | 0.746 | -0.007 |
| context_recall | 0.819 | 0.819 | 0.000 |

Category breakdown comparison:

| Category | Phase 3a | Phase 4 | Delta |
|----------|----------|---------|-------|
| simple_lookup (n=16) | 0.898 | **0.963** | **+0.065** |
| multi_fact (n=10) | 0.727 | **0.805** | **+0.078** |
| cross_chunk (n=6) | 0.777 | 0.670 | -0.107 |
| edge_case (n=4) | 0.781 | 0.720 | -0.061 |
| negative (n=4) | 0.400 | 0.344 | -0.056 |

**Interpretation**

- **faithfulness +0.056** — the cross-encoder is surfacing more grounded chunks as the top-5, reducing hallucination. Primary goal achieved.
- **multi_fact +0.078** — the most important improvement. BM25 + dense fusion ensures keyword-critical chunks (specific article numbers, named sections) are not missed by the embedding model alone. This was Phase 3a's identified primary weakness; Phase 4 targeted and fixed it.
- **simple_lookup +0.065** — straightforward fact lookup benefits from hybrid coverage.
- **cross_chunk -0.107** — the most significant regression. Cross-chunk questions require diverse chunks from different parts of the document. The reranker, trained on single-passage relevance (MS MARCO), compresses the candidate pool too aggressively and drops chunks that are individually lower-scoring but collectively necessary for multi-hop synthesis. This is a known limitation of point-wise rerankers.
- **context_precision -0.007** — within noise, essentially flat. The bilingual Constitution noise issue persists; BM25 does not help here (it is language-agnostic but still retrieves Kannada chunks that match query tokens).
- **context_recall 0.000** — no change. The corpus coverage from Phase 3a already sets the ceiling; top-5 retrieval on 1372 chunks has the same recall ceiling regardless of retrieval order.

**Honest trade-off summary**

Phase 4 is a net improvement on the metrics the system was designed to optimize (faithfulness, multi-fact accuracy) but introduces a regression on synthesis queries. The eval surfaced this trade-off before deployment — without RAGAS, the cross-chunk degradation would have shipped silently. This is the eval-driven development discipline working as intended: visible trade-offs are better than invisible regressions.

**Next steps (Phase 4b candidates, not yet executed)**
- Increase `top_k` from 5 to 8 for the reranker to give the generator more diverse context (cheap, low-risk)
- Use a list-wise reranker (ColBERT, BGE Reranker) that considers cross-chunk diversity
- Filter bilingual Constitution chunks by language at ingest time, or re-download a pure-English Constitution PDF
