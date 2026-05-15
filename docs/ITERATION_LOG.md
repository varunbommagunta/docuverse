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

## Phase 1 — Ingestion

**Goal:** PDF → Chunks stored in memory (no vector DB yet).

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
