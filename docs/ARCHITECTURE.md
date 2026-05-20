# DocuVerse — Architecture

A full architecture reference for the DocuVerse RAG system. Covers component design, request flows, retrieval strategies, deployment topology, configuration model, evaluation methodology, and known trade-offs.

---

## 1. System Overview

DocuVerse is a modular monolith structured on the Ports-and-Adapters (hexagonal architecture) pattern. All business logic lives in `src/`, with zero awareness of HTTP, Streamlit, or any I/O framework. Two thin adapters translate between the outside world and domain objects: `api/` exposes the orchestrator over FastAPI, and `ui/` wraps it in a Streamlit chat interface.

Every replaceable component — parser, embedder, vector store, retriever, reranker, generator — is defined as a `typing.Protocol` in the relevant `base.py` file. Concrete implementations satisfy these protocols structurally (no inheritance required). This lets any implementation be swapped at startup without touching business logic, and keeps unit tests simple: mock the protocol, not the class. The factory wiring lives in `src/factory.py`.

At this scale — a single-engineer project targeting a specific document corpus — a modular monolith is the right choice. A microservices split would add deployment complexity and network latency without meaningfully improving independence between components that share a single ChromaDB index and a single OpenAI API key. The monolith is kept honest by the protocol boundaries: the interface between `src/orchestrator.py` and its dependencies is as clean as a service boundary, without the operational overhead.

---

## 2. Request Flow — Upload

An upload begins when the user selects a PDF in `ui/app.py`. Streamlit issues a `multipart/form-data` POST to the `/api/ingest` path, which nginx routes to FastAPI on port 8000 (stripping the `/api` prefix). The handler at `api/routes/ingest.py` validates the file — checking MIME type, enforcing `MAX_UPLOAD_SIZE_MB`, writing to a temp file — then calls `IngestionPipeline.ingest()` from `src/ingestion/pipeline.py`.

The pipeline runs three sequential stages with per-step timing logs: `PyPDFParser.parse()` (`src/ingestion/parsers.py`) extracts per-page text and metadata into a `ParsedDocument`; `RecursiveChunker.chunk()` (`src/ingestion/chunker.py`) splits each page into overlapping `Chunk` objects with UUID IDs; `OpenAIEmbedder.embed()` (`src/retrieval/embedder.py`) batch-embeds the chunk texts via the OpenAI API with tenacity retry (3 attempts, exponential backoff).

The embedded chunks land in `ChromaVectorStore` (`src/retrieval/vector_store.py`), which persists them to disk in `data/chroma_db/` using cosine similarity (`hnsw:space: cosine`). After ingestion, the orchestrator triggers a BM25 index rebuild in `BM25Retriever` (`src/retrieval/bm25_retriever.py`) by calling `get_all_chunks()` on the vector store, keeping keyword retrieval in sync with the vector index.

---

## 3. Request Flow — Query

A query begins at `ui/app.py` when the user submits text in the chat input. Streamlit issues a POST to `/api/query`, routed through nginx to FastAPI on port 8000. Rate limiting via `slowapi` is applied at this layer before the route handler runs.

The handler at `api/routes/query.py` calls `RAGOrchestrator.query()` in `src/orchestrator.py`. The orchestrator invokes the full retriever chain: `RerankedRetriever` wraps `HybridRetriever`, which wraps both `DenseRetriever` and `BM25Retriever`. `HybridRetriever` fetches top-20 from each retriever, fuses the ranked lists with Reciprocal Rank Fusion (k=60), deduplicates by `chunk_id`, and passes the top-50 candidates to `RerankedRetriever`. The cross-encoder (`CrossEncoderReranker` in `src/retrieval/cross_encoder_reranker.py`, using ms-marco-MiniLM-L-6-v2) scores each `(query, chunk)` pair and returns the final top-k by sigmoid-normalized logit.

The orchestrator passes the top-k chunks as context to `OpenAIGenerator` (`src/generation/openai_generator.py`). The generator builds a citation-forcing system prompt that instructs gpt-4o-mini to ground every claim in `[chunk_N]` references, calls the OpenAI chat completion API, and parses the response with a regex to extract citation indices. The resulting `Answer` object — `text`, `citations` (list of chunk indices), and `retrieved_chunks` — is returned through the API and rendered in Streamlit with expandable source attribution.

---

## 4. Retrieval Strategies

Four retrieval strategies are available, selected at container startup via the `RETRIEVAL_STRATEGY` environment variable. The factory dispatch logic lives in `src/factory.py`; changing the strategy requires only a container restart.

**dense** — `DenseRetriever` embeds the query with `OpenAIEmbedder` and performs cosine similarity search in ChromaDB. Fast and zero local compute, but misses keyword-critical matches when the embedding space is sparse for technical terms or proper nouns.

**sparse** — `BM25Retriever` builds a `BM25Okapi` index at startup from all chunks in ChromaDB. Scores are normalized to [0,1] by dividing by the max score. Robust for exact keyword queries but blind to semantic synonyms and paraphrases.

**hybrid** — `HybridRetriever` fetches top-20 from dense and top-20 from BM25, then fuses the ranked lists with Reciprocal Rank Fusion. RRF (k=60, from the original paper) is scale-invariant and stable: it handles the incompatible score ranges of BM25 raw scores and cosine similarities without calibration.

**reranked_hybrid** (default in production) — `RerankedRetriever` wraps any base retriever and adds a second-stage cross-encoder pass. It fetches `RERANKER_FETCH_K=50` candidates from the hybrid retriever, rescores each pair with ms-marco-MiniLM-L-6-v2, and returns the top-k. Phase 4 evaluation measured +5.6 pp faithfulness and +7.8 pp multi-fact accuracy over pure dense retrieval on the 40-question realistic dataset.

---

## 5. Deployment Architecture

HF Spaces exposes exactly one public port, declared in the `app_port` field of the README YAML front-matter (port 7860 for DocuVerse). A single nginx process (`docker/nginx.conf`) binds on 7860 and acts as an internal reverse proxy: requests to `/api/*` forward to FastAPI on port 8000 (prefix stripped); all other requests route to Streamlit on port 8501 with WebSocket upgrade headers (`Upgrade`, `Connection`) for live UI updates.

Supervisord (`docker/supervisord.conf`) runs as PID 1 inside the container, managing both the FastAPI process (via Uvicorn) and the Streamlit process, each with `autorestart=true`. Supervisord receives the container stop signal and relays it cleanly to both children. The nginx config uses `pid /tmp/nginx.pid` and writes access/error logs to `/dev/stdout` and `/dev/stderr` for HF Spaces log streaming compatibility.

`client_max_body_size 12m` is set at the `http {}` block level in `docker/nginx.conf`, and also repeated at the `server {}` and each `location {}` block. This is load-bearing: nginx enforces the body size limit before routing decisions, so a location-only directive was found (in production) to be insufficient for the Streamlit upload path, causing HTTP 413 errors on PDFs over 1 MB.

---

## 6. Configuration Model

All configuration lives in `config/settings.py` as a Pydantic `BaseSettings` subclass. Values are read from environment variables (case-insensitive) and the `.env` file at startup via `SettingsConfigDict`. No other module reads `os.environ` directly — all config access goes through `get_settings()`, which is `lru_cache(maxsize=1)`-memoized to parse the environment exactly once per process. In tests, `get_settings.cache_clear()` is called before patching env vars.

The `RETRIEVAL_STRATEGY` env var is read at container startup by `src/factory.py` to wire the correct retriever chain. The six Phase 4 config fields (`HYBRID_DENSE_TOP_K`, `HYBRID_SPARSE_TOP_K`, `HYBRID_RRF_K`, `RERANKER_MODEL`, `RERANKER_FETCH_K`, `TOP_K`) all have safe defaults that match the evaluated production configuration. Changing any of them requires only a container restart — no code modification.

The `openai_api_key` field carries a `@field_validator(mode="before")` that calls `.strip()` on the value before Pydantic validates it. This is a defensive measure against a class of production failure observed when secrets are pasted into HF Spaces env var fields with a trailing newline: a stray `\n` causes the raw key string to be passed to the OpenAI client, which returns HTTP 401 — easy to misdiagnose as a configuration problem rather than a formatting one.

---

## 7. Evaluation Methodology

Quality is measured with RAGAS 0.4.3 using four metrics scored by a gpt-4o-mini judge: **faithfulness** (are all claims in the answer grounded in retrieved context?), **answer relevancy** (does the answer address the question?), **context precision** (what fraction of retrieved chunks are actually useful?), and **context recall** (do the retrieved chunks contain all the information needed to answer?). Each metric is scored 0–1; production thresholds are faithfulness ≥ 0.75, relevancy ≥ 0.70, precision ≥ 0.70, recall ≥ 0.65. See [docs/EVALUATION.md](EVALUATION.md) for full definitions and usage guide.

The evaluation dataset (`data/eval/v1_realistic_dataset.json`) contains 40 questions across five categories: `simple_lookup` (n=16), `multi_fact` (n=10), `cross_chunk` (n=6), `edge_case` (n=4), and `negative` (n=4), grounded in 1372 chunks from the Constitution of India and the ARC Ethics in Governance report. Ground truths were written from actual orchestrator answers on the ingested corpus — this ensures they reflect what the system can plausibly retrieve rather than an idealized human reference.

Two major runs are recorded. Phase 3a (run `a14988ca`) established the V1 realistic baseline using pure dense retrieval. Phase 4 (run `ab12c0f5`) measured the impact of `reranked_hybrid`. Raw JSON reports with per-sample scores and aggregate stats are in [docs/eval_results/](eval_results/); a human narrative interpretation is appended to [docs/ITERATION_LOG.md](ITERATION_LOG.md) after each run. Scores should be interpreted as relative comparisons between versions, not absolute quality measures — LLM judges carry ±0.05 noise across runs.

---

## 8. Known Trade-offs

**Monolith vs. microservices.** All components share one process tree and one ChromaDB index. This simplifies deployment and eliminates inter-service latency. The cost: the BM25 index is rebuilt in-process at startup (~2s on 1372 chunks) and the cross-encoder model loads lazily on first request (~3s). A microservices split would move these to dedicated service start times without improving user-facing latency at this corpus scale.

**Single container vs. multi-container.** HF Spaces free tier provides one container and one public port. nginx + supervisord inside a single Docker image is the correct architectural response to this constraint. The trade-off is that a crash in one process (FastAPI or Streamlit) is handled by supervisord autorestart rather than an independent container restart policy. In practice, both processes are stateless and restart in under 2s; the risk is low.

**OpenAI API vs. self-hosted LLM.** Using OpenAI for both embeddings and generation creates a hard dependency on an external service and incurs per-token costs. The daily cost cap (`DAILY_COST_CAP_INR=50`) enforces a hard spending ceiling, returning HTTP 429 once exceeded. Self-hosted alternatives (Ollama, vLLM) would eliminate the dependency but require GPU infrastructure unavailable on the free HF Spaces tier. The current approach is correct for the deployment environment.

**Point-wise reranker vs. list-wise.** The ms-marco cross-encoder scores each `(query, chunk)` pair independently and cannot model diversity or inter-chunk relationships. Phase 4 evaluation measured a -10.7% regression on `cross_chunk` questions: the reranker promotes the single most relevant chunk and drops others that are individually lower-scoring but collectively necessary for multi-hop synthesis. A list-wise reranker (ColBERT, BGE Reranker) or increasing `TOP_K` from 5 to 8 would address this at higher per-query latency. Documented as Phase 4b work.

**Free-tier deployment constraints.** HF Spaces free tier sleeps the container after ~15 minutes of inactivity, causing a ~60s cold start for the next visitor. The Constitution of India PDF (6.65 MB) exceeds the 6 MB upload limit enforced at both the nginx and application layers; it is accessible only via direct API calls (`POST /api/ingest`), not through the Streamlit upload UI. Both are accepted trade-offs for zero infrastructure cost.

---

## See Also

- [README.md](../README.md) — project overview, key results, live demo
- [docs/ITERATION_LOG.md](ITERATION_LOG.md) — phase-by-phase engineering log with decisions and measurement
- [docs/EVALUATION.md](EVALUATION.md) — RAGAS metric definitions, thresholds, and usage guide
- [docs/eval_results/](eval_results/) — raw JSON evaluation reports per run
