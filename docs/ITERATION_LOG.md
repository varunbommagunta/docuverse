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

## Phase 2 — Retrieval Improvements

V1 baseline coming after Phase 1 ships.

---

## Phase 2 — Retrieval

**Goal:** Chroma vector store, OpenAI embeddings, similarity search.

---

## Phase 3 — Generation

**Goal:** GPT-4o-mini with cited answers; end-to-end RAG pipeline live.

---

## Phase 4 — Evaluation

**Goal:** RAGAS harness, baseline scores, regression guard in CI.

---

## Phase 5 — Production Hardening

**Goal:** CI/CD, observability, rate limiting, auth stubs.
