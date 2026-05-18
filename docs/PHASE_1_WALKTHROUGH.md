# DocuVerse — Phase 1 Walkthrough

> **Senior-engineer teaching guide.** Read-only. No code was changed to produce this document.  
> Covers every file introduced or modified in Phase 1, every design decision, every data flow,
> every library choice, and two complete request traces end-to-end.

---

## Table of Contents

1. [Ingestion Flow](#1-ingestion-flow)
2. [Retrieval Flow](#2-retrieval-flow)
3. [Generation Flow](#3-generation-flow)
4. [Orchestrator](#4-orchestrator)
5. [Factory Pattern](#5-factory-pattern)
6. [API Layer](#6-api-layer)
7. [UI Adapter](#7-ui-adapter)
8. [Tests](#8-tests)
9. [Big Picture — End-to-End Traces](#9-big-picture--end-to-end-traces)
10. [Protocol Proof Table](#10-protocol-proof-table)

---

## 1. Ingestion Flow

**Files:** `src/ingestion/parsers.py`, `src/ingestion/chunkers.py`, `src/ingestion/pipeline.py`

### 1.1 `parsers.py` — PyPDFParser

```
file_path (str)
    └── PdfReader(file_path)           ← pypdf opens the file
        └── page.extract_text()        ← per-page text extraction
            └── "\n\n".join(pages)     ← full_text
                └── ParsedDocument(text, pages, metadata)
```

**Why pypdf?**  
Pure-Python, zero native dependencies, ships as a single wheel. Alternatives like `pdfminer.six` give more layout fidelity but are significantly heavier. `pdfplumber` wraps pdfminer. For portfolio purposes pypdf is the correct minimal choice; you'd swap in a more capable parser for scanned PDFs (OCR via Tesseract) or complex layouts without touching anything outside `src/ingestion/`.

**ParsedDocument vs list[str]:**  
The Phase 0 `Parser` Protocol declares `parse() -> list[str]`. `PyPDFParser.parse()` returns `ParsedDocument`. This is intentional Protocol drift — `ParsedDocument` carries `pages: list[str]`, `text: str`, and `metadata: dict` that `IngestionPipeline` needs downstream. The drift is documented in both files and will be reconciled in Phase 2 when the Protocol is updated. `IngestionPipeline` imports the concrete type directly, not the Protocol.

**Error handling:**  
Two custom exceptions from `src/utils/exceptions.py`:
- `DocumentParseError` — file not found, corrupt PDF, pypdf internal failure.
- Per-page failures are **warnings**, not errors. A 100-page PDF where page 37 has no extractable text (scanned image) should not abort ingestion of the other 99 pages.

**Metadata enriched:**  
`filename`, `total_pages`, `source_path` — these flow into every `Chunk.metadata` so retrieved chunks can report provenance.

---

### 1.2 `chunkers.py` — RecursiveChunker

**Why chunk at all?**  
LLM context windows are finite and cost-per-token is real. A 200-page PDF is ~400 000 tokens. Sending the full document with every query is impractical. Chunking splits the document into semantically coherent pieces; retrieval selects only the relevant pieces.

**Why RecursiveCharacterTextSplitter?**  
LangChain's splitter tries separators in priority order: `["\n\n", "\n", ". ", "! ", "? ", " ", ""]`. It first tries to split on paragraph boundaries (`\n\n`), then sentence boundaries, then word boundaries. This preserves semantic units as much as possible. A naive fixed-size splitter (split every N characters) would shred sentences mid-thought.

**chunk_size=500, chunk_overlap=50:**  
- 500 characters ≈ 80-100 words ≈ 4-6 sentences. Enough context for a meaningful retrieval unit.
- 50-character overlap prevents a key fact from being cut across two chunks and never retrieved. The end of chunk N and the start of chunk N+1 share 50 characters.

**UUID chunk IDs:**  
`str(uuid.uuid4())` — globally unique, generated at ingestion time. ChromaDB uses these as primary keys. UUIDs mean you can re-ingest a document without worrying about ID collisions with previous runs.

**chunk_index in metadata:**  
The `[chunk_N]` citation format the LLM produces refers to the 0-based index in the *retrieved list*, not the chunk_index in the document. But chunk_index is preserved in metadata for debugging — you can look at chunk_index=42 and know it's the 42nd chunk in the original document order.

---

### 1.3 `pipeline.py` — IngestionPipeline

```
file_path, filename
    ├── doc_id = uuid4()
    ├── parser.parse(file_path)          → ParsedDocument
    │       metadata["filename"] = display_name   ← override temp filename
    ├── chunker.chunk(parsed_doc, doc_id) → list[Chunk]
    ├── embedder.embed([c.text …])        → list[list[float]]
    └── vector_store.add_chunks(chunks, embeddings)
```

**Why override `metadata["filename"]` after parsing?**  
The PDF is written to a `tempfile.NamedTemporaryFile` before parsing — the OS assigns a name like `tmpXXXXXX.pdf`. `PyPDFParser` would record that temp name as `filename`. The pipeline overrides it with the real `display_name` (original uploaded filename) so chunks carry the user-visible filename.

**Shared embedder:**  
`IngestionPipeline` and `DenseRetriever` receive the **same** `OpenAIEmbedder` instance from `src/factory.py`. This matters: if you ever add in-memory embedding caching to the embedder, both consumers benefit automatically.

**`time.perf_counter()` timing:**  
Each step (parse, chunk, embed, store) is timed and logged with structlog. This is your observability baseline — if ingestion is slow you can immediately see which step is the bottleneck without adding any instrumentation later.

**Return value:**  
`{"document_id": str, "filename": str, "chunk_count": int}` — just enough for the API to build an `IngestResponse`. The pipeline doesn't return chunks or embeddings; callers shouldn't need them.

---

## 2. Retrieval Flow

**Files:** `src/retrieval/embedders.py`, `src/retrieval/vector_store.py`, `src/retrieval/dense_retriever.py`

### 2.1 `embedders.py` — OpenAIEmbedder

**What is an embedding?**  
A text embedding is a fixed-length vector of floats where semantic similarity = geometric proximity. "The dog ran fast" and "The canine sprinted quickly" will be close in this 1536-dimensional space. "The stock market crashed" will be far away. This is what makes semantic search possible: you embed the query and find documents whose embeddings are nearby.

**`text-embedding-3-small`:**  
OpenAI's smaller embedding model. 1536 dimensions (down from 1536 for `text-embedding-ada-002` — same size but better quality at lower cost). You call `.create(model=..., input=[...])` and get back a list of embedding vectors.

**Batching:**  
`embed(texts: list[str])` sends all texts in a single API call. OpenAI accepts up to ~2048 inputs per call. This matters during ingestion: a 50-chunk document sends 1 API call, not 50.

**`embed_single` convenience method:**  
`embed([text])[0]` — used by `DenseRetriever` during query time when you have one query string.

**tenacity retry decorator:**  
```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((RateLimitError, APIError)),
    reraise=True,
)
```
- **stop_after_attempt(3):** Try 3 times total (1 initial + 2 retries).
- **wait_exponential:** Wait 1s, then 2s, then 4s (capped at 10s). Exponential backoff avoids hammering a rate-limited API.
- **retry_if_exception_type:** Only retry transient failures (`RateLimitError`, `APIError`). Don't retry `AuthenticationError` or `BadRequestError` — those won't self-heal.
- **reraise=True:** After 3 failures, re-raise the original exception (not a tenacity wrapper) so callers see a clean `RateLimitError` / `APIError`.

The public `embed()` method catches these and re-raises as `RetrievalError` — the domain's vocabulary. Callers don't need to know about OpenAI error types.

---

### 2.2 `vector_store.py` — ChromaVectorStore

**What is ChromaDB?**  
An embedded vector database — it runs in the same Python process, no separate server. `PersistentClient(path=...)` writes to disk so data survives process restarts.

**`hnsw:space: cosine`:**  
HNSW (Hierarchical Navigable Small World) is the approximate nearest-neighbor algorithm Chroma uses internally. `cosine` configures it to measure cosine distance (= 1 - cosine_similarity). This must match your embedding model's geometry — OpenAI embeddings work best with cosine similarity.

**`similarity = 1 - distance`:**  
Chroma returns distances (lower = more similar). We convert to similarity (higher = more similar) for interpretability. A similarity of 0.95 is a strong match; 0.3 is a weak one. We clamp with `max(0.0, ...)` to prevent slightly negative floats from floating-point arithmetic.

**`get_or_create_collection`:**  
Idempotent. First run: creates the collection. Subsequent runs: loads the existing one. This is why re-running the API doesn't lose your ingested documents.

**`add_chunks`:**  
Passes four parallel lists to Chroma: `ids`, `embeddings`, `documents` (raw text), `metadatas`. Chroma stores all four, so `similarity_search` can return the original text without a separate lookup.

**`# type: ignore[arg-type]` on embeddings:**  
Chroma's stub files declare `embeddings` as `ndarray | list[ndarray] | ...` — they don't include `list[list[float]]` even though the runtime accepts it. This is a stub quality issue in chromadb, not a bug. We suppress with `type: ignore` rather than converting to numpy arrays just to satisfy the type checker.

**`delete_document`:**  
Uses `where={"document_id": doc_id}` — a Chroma metadata filter. This is how you remove all chunks of a specific document. Returns the count of deleted chunks (useful for logging).

**`count()`:**  
Returns total chunks across all documents. Used by the query endpoint to return a 503 when nothing has been ingested.

---

### 2.3 `dense_retriever.py` — DenseRetriever

```
query (str)
    └── embedder.embed_single(query)        → list[float]  (1536 dims)
        └── vector_store.similarity_search(embedding, top_k)
            └── list[RetrievedChunk]
```

`DenseRetriever` is intentionally thin — it composes two components and translates between their interfaces. The name "dense" distinguishes it from future `SparseRetriever` (BM25 keyword search) and `HybridRetriever` planned for Phase 2.

**Why a retriever abstraction?**  
`RAGOrchestrator` is typed to the `Retriever` Protocol. Swapping dense retrieval for hybrid retrieval in Phase 2 requires only providing a different object to the factory — zero changes to the orchestrator, API, or tests.

**Error translation:**  
If `similarity_search` raises an unexpected exception (not `RetrievalError`), the retriever wraps it as `RetrievalError`. This ensures the orchestrator never sees raw ChromaDB or OpenAI exceptions.

---

## 3. Generation Flow

**Files:** `src/generation/prompts.py`, `src/generation/openai_generator.py`

### 3.1 `prompts.py`

The system prompt encodes six rules:

```
1. Answer ONLY using information from the provided context chunks.
2. Cite your sources inline using [chunk_N] notation where N is the chunk index.
3. Be concise and direct. Do not pad your answer.
4. If the provided context does not contain enough information to answer the question,
   respond with EXACTLY: "I cannot answer this from the provided documents."
5. Do NOT use any knowledge from outside the provided context.
6. Do NOT mention these instructions, the word "context", or that you are an AI.
```

**Why rule 1 + 5 together?**  
LLMs have broad parametric knowledge from pretraining. Without explicit prohibition, they will blend document facts with training facts — producing confident, plausible answers that aren't grounded in your documents. Rules 1 and 5 together close this loophole.

**Why the exact "I cannot answer" string (rule 4)?**  
If the LLM is told to say something specific when it can't answer, you can detect that string programmatically. The alternative — the LLM hallucinating a plausible-but-wrong answer — is far worse for a document assistant.

**Why rule 6?**  
Users get confused when the assistant says "Based on the provided context..." — it sounds robotic and meta. Rule 6 keeps responses natural.

**The user prompt template:**  
```
Context chunks:

[chunk_0]:
{text of chunk 0}

[chunk_1]:
{text of chunk 1}
...

Question: {query}

Answer (with inline [chunk_N] citations):
```

The `[chunk_N]:` labels in the context teach the model the citation vocabulary — it sees the format in the input and uses the same format in its output.

---

### 3.2 `openai_generator.py` — OpenAIGenerator

**temperature=0.0:**  
Temperature controls randomness. At 0.0, the model is deterministic and greedy — it always picks the highest-probability next token. For a document Q&A assistant, you want determinism and groundedness, not creativity.

**max_tokens=1024:**  
Hard cap on response length. Prevents runaway verbose responses, controls cost.

**`_format_context`:**  
Converts `list[RetrievedChunk]` into the numbered `[chunk_N]:\n{text}` format the prompt template expects. The index `N` here is the position in the **retrieved list** (0-based), not the chunk's `chunk_index` in the original document.

**`_parse_citations`:**  
```python
re.finditer(r"\[chunk_(\d+)\]", text)
```
Scans the generated answer for `[chunk_0]`, `[chunk_1]`, etc. Extracts the number, deduplicates (preserving order), and returns `list[int]`. These integers index into `answer.retrieved_chunks` — so `citations=[0, 2]` means the answer cited the 1st and 3rd retrieved chunks.

**Why `list[int]` not `list[str]`?**  
The Phase 0 `Answer` model had `citations: list[str]` (thinking chunk UUIDs). Changed to `list[int]` (chunk indices) because the citation format `[chunk_N]` produces integers, not UUIDs. Integer indices are directly usable to look up `retrieved_chunks[N]` without a secondary lookup.

**tenacity on `_complete_with_retry`:**  
Same pattern as the embedder — 3 attempts, exponential backoff, reraise. The public `generate()` method catches and re-raises as `GenerationError`.

---

## 4. Orchestrator

**File:** `src/orchestrator.py`

```python
class RAGOrchestrator:
    def answer(self, query: str) -> Answer:
        chunks = self._retriever.retrieve(query, top_k=5)
        result = self._generator.generate(query, chunks)
        return result
```

Two lines of business logic. This is not a mistake — it is proof that the architecture is correct.

**Why so short?**  
Every concern is separated:
- **How to retrieve** → `DenseRetriever`
- **How to generate** → `OpenAIGenerator`
- **How to wire them** → `src/factory.py`
- **How to expose them** → `api/routes/query.py`

The orchestrator's job is to sequence retrieval + generation. That's it. Separation of Concerns / Single Responsibility Principle in practice.

**Dependency injection via constructor:**  
```python
def __init__(self, retriever: Retriever, generator: Generator) -> None:
```
Both parameters are typed to **Protocols**, not concrete classes. The orchestrator has zero imports of `DenseRetriever`, `OpenAIGenerator`, or any specific implementation. This is the Dependency Inversion Principle — high-level policy depends on abstractions, not details.

**Error pass-through:**  
`RetrievalError` and `GenerationError` are re-raised unwrapped. Unexpected exceptions are wrapped and re-raised as the appropriate domain error. The API layer catches these and maps them to HTTP status codes.

---

## 5. Factory Pattern

**File:** `src/factory.py`

```
get_rag_components() → (RAGOrchestrator, IngestionPipeline)
    ├── settings = get_settings()           ← @lru_cache singleton
    ├── embedder = OpenAIEmbedder(...)      ← shared instance
    ├── vector_store = ChromaVectorStore(...)  ← shared instance
    ├── parser = PyPDFParser()
    ├── chunker = RecursiveChunker(...)
    ├── pipeline = IngestionPipeline(parser, chunker, embedder, vector_store)
    ├── retriever = DenseRetriever(embedder, vector_store)
    ├── generator = OpenAIGenerator(...)
    └── orchestrator = RAGOrchestrator(retriever, generator)
```

**What problem does the factory solve?**  
`RAGOrchestrator.__init__` takes a `Retriever` and a `Generator`. But who creates the `DenseRetriever`? And who creates the `OpenAIEmbedder` that `DenseRetriever` needs? The construction chain has to start somewhere. The factory is that starting point — the **composition root** of the application.

**Why not inside `api/main.py`?**  
Separation of concerns. The API adapter shouldn't know which concrete classes to instantiate. Keeping construction in `src/factory.py` means:
1. Tests can call `get_rag_components()` directly.
2. A future CLI entrypoint can call the same factory.
3. If you add a new dependency (e.g., a cross-encoder reranker), you change one file.

**Shared embedder/vector_store:**  
`embedder` and `vector_store` are created once and passed to *both* `IngestionPipeline` and `DenseRetriever`. One Chroma `PersistentClient` → no file locking issues. If you add embedding caching to `OpenAIEmbedder` in Phase 2, both ingestion and retrieval benefit automatically.

**`get_settings()` with `@lru_cache`:**  
Settings are parsed once per process. The factory calls `get_settings()` on every `get_rag_components()` call — but since `get_settings` is cached, it returns the same `Settings` object instantly after the first parse.

**Called from API lifespan:**  
```python
# api/main.py
orchestrator, pipeline = get_rag_components()
app.state.orchestrator = orchestrator
app.state.pipeline = pipeline
```
Components are singletons for the lifetime of the API process. Created once at startup, reused for every request.

---

## 6. API Layer

**Files:** `api/main.py`, `api/dependencies.py`, `api/routes/ingest.py`, `api/routes/query.py`, `api/schemas.py`

### 6.1 `main.py` — Lifespan and App Assembly

**`@asynccontextmanager` lifespan:**  
FastAPI's recommended pattern since v0.93. Code before `yield` runs at startup; code after `yield` runs at shutdown. This replaces deprecated `@app.on_event("startup")`.

**`app.state`:**  
FastAPI's built-in singleton store for the application. `app.state.orchestrator` and `app.state.pipeline` are set at startup and available to every request handler via `request.app.state`.

**`from collections.abc import AsyncGenerator`:**  
Not `from typing import AsyncGenerator`. The `typing` version is deprecated as of Python 3.9 (UP035 ruff rule). Always prefer `collections.abc` for runtime types.

---

### 6.2 `dependencies.py` — Dependency Injection

```python
def get_orchestrator(request: Request) -> RAGOrchestrator:
    return request.app.state.orchestrator

def get_pipeline(request: Request) -> IngestionPipeline:
    return request.app.state.pipeline
```

**FastAPI `Depends()` pattern:**  
Route functions declare their dependencies as default arguments:
```python
async def ingest_document(
    file: UploadFile,
    pipeline: IngestionPipeline = Depends(get_pipeline),
):
```
FastAPI calls `get_pipeline(request)` and injects the result. This is constructor injection for route functions — the route doesn't reach into global state or import the factory directly.

**Why `Depends()` in default args triggers B008:**  
`ruff`'s B008 rule warns about function calls in default argument positions (they're evaluated once at import time). `Depends(get_pipeline)` is a FastAPI-specific exception — FastAPI inspects the default value as metadata, not as an actual call. Fixed by adding `"B008"` to the ruff ignore list in `pyproject.toml`.

---

### 6.3 `routes/ingest.py` — POST /ingest

**Request flow:**
```
POST /ingest  (multipart/form-data, file field)
    ├── filename validation (.pdf extension check)
    ├── await file.read()                ← load bytes into memory
    ├── size check (> 25 MB → 413)
    ├── NamedTemporaryFile(suffix=".pdf")
    │       tmp.write(contents)
    │       tmp_path = tmp.name
    ├── pipeline.ingest(tmp_path, filename=filename)
    └── finally: os.unlink(tmp_path)     ← always clean up
```

**Why `UploadFile` + temp file?**  
`UploadFile` is an async file-like object representing the incoming multipart stream. `pypdf.PdfReader` needs a file path (or seekable file object). The pattern is: read all bytes → write to temp file → pass path to parser → clean up in `finally`. The `finally` block runs even if parsing raises an exception, ensuring no temp file leaks.

**`delete=False` on NamedTemporaryFile:**  
On Windows, `delete=True` (default) would try to delete the file when the `with` block exits — before we've finished using it. `delete=False` + explicit `os.unlink()` in `finally` works cross-platform.

**Status codes:**
- `400` — Not a PDF (wrong extension or content type)
- `413` — Exceeds 25 MB size limit
- `422` — Valid PDF but parse failed (corrupt, image-only, etc.) or chunking failed
- `200` — Success: `{document_id, filename, chunk_count}`

---

### 6.4 `routes/query.py` — POST /query

**503 guard:**  
```python
vector_store = orchestrator._retriever._vector_store
if vector_store.count() == 0:
    raise HTTPException(status_code=503, ...)
```
Returns 503 (Service Unavailable) when no documents are indexed. Without this guard, the LLM would receive an empty context and likely hallucinate or produce "I cannot answer." The 503 gives clients a machine-readable signal to prompt the user to upload a document first.

**Private attribute access:**  
`orchestrator._retriever._vector_store` traverses private attributes. This is acknowledged as a pragmatic workaround — the `Retriever` Protocol has no `count()` method. Phase 2 will add a `document_count()` method to the Protocol (or a separate `VectorStoreInfo` interface) to clean this up.

**`Answer → QueryResponse` mapping:**  
```python
citation_details = [
    CitationDetail(
        chunk_index=idx,
        chunk_id=rc.chunk.id,
        text=rc.chunk.text,
        score=rc.score,
        metadata=rc.chunk.metadata,
    )
    for idx, rc in enumerate(answer.retrieved_chunks)
]
return QueryResponse(
    answer=answer.text,
    citations=answer.citations,
    retrieved_chunks=citation_details,
)
```
`Answer` is a domain model. `QueryResponse` is a public API schema. The mapping adds `chunk_index` (not in `RetrievedChunk`) and exposes `chunk_id`, `text`, `score`, and `metadata` for rich client-side citation rendering.

---

### 6.5 `schemas.py` — API Contracts

All request/response types are Pydantic models in `api/schemas.py`, separate from domain models in `src/utils/models.py`. This is the Ports-and-Adapters boundary: domain models live in `src/`, API contracts live in `api/`.

`QueryRequest.query` has `min_length=1` — Pydantic validates this before the route handler runs, returning a 422 automatically for empty strings.

---

## 7. UI Adapter

**File:** `ui/app.py`

### Architecture

`ui/app.py` is a pure **adapter** — it has zero imports from `src/`. All communication with the domain goes through HTTP calls to the API. This is Ports-and-Adapters proven: the UI and the domain are fully decoupled. You could replace Streamlit with a React SPA or a CLI without touching `src/` at all.

### `API_URL` Environment Variable

```python
API_URL = os.getenv("API_URL", "http://localhost:8000")
```

- **Local development:** `http://localhost:8000` (default)
- **Docker Compose:** `API_URL=http://api:8000` (set in `docker-compose.yml`)

Docker service DNS resolution: when services run in the same Compose network, `api` resolves to the API container's IP. `localhost` from inside the UI container would refer to the UI container itself. This is 12-Factor config — one codebase, multiple environments.

### `st.session_state`

Streamlit reruns the entire script on every user interaction. `st.session_state` is a dict that persists across reruns within a session:

```python
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ingested_docs" not in st.session_state:
    st.session_state.ingested_docs = []
```

Without `session_state`, the chat history would disappear every time the user types a message.

### Upload Flow

```python
requests.post(
    f"{API_URL}/ingest",
    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
)
```

`requests.post(..., files=...)` sends a `multipart/form-data` request — the same format the browser uses for `<input type="file">`. The API's `UploadFile` parameter receives this.

### Chat Flow

```python
requests.post(f"{API_URL}/query", json={"query": prompt})
```

JSON body, matching `QueryRequest(query=...)`. The response includes `answer`, `citations` (list of int indices), and `retrieved_chunks` (full detail). The UI uses these to render expandable source attribution panels.

### Key Ruff Fixes

**F821 — `_show_sources` called before definition:**  
Streamlit scripts run top-to-bottom. If you call `_show_sources()` on line 73 and define it on line 90, Python raises `NameError`. Fixed by moving the function definition above the chat loop.

**SIM117 — nested `with` statements:**  
```python
# Before (SIM117 violation):
with st.chat_message("assistant"):
    with st.spinner("Thinking…"):

# After (fixed):
with st.chat_message("assistant"), st.spinner("Thinking…"):
```

---

## 8. Tests

**Files:** `tests/conftest.py`, `tests/unit/test_parsers.py`, `tests/unit/test_chunkers.py`, `tests/unit/test_dense_retriever.py`, `tests/unit/test_openai_generator.py`, `tests/unit/test_orchestrator.py`, `tests/integration/test_end_to_end.py`

### 8.1 `conftest.py` — Shared Fixtures

```python
@pytest.fixture(scope="session", autouse=True)
def ensure_sample_pdf():
    """Generate data/sample/sample.pdf if it doesn't exist."""
```

`scope="session"` — runs once per test session (not once per test or once per module). `autouse=True` — applies to all tests without needing `@pytest.mark.usefixtures`. The PDF is generated with `reportlab` and contains 3 pages about the Solar System.

The `sample_pdf_path` fixture returns the absolute path to this file, used by parser tests and the integration test.

---

### 8.2 `test_parsers.py` — 7 Tests, No Mocks

Parser tests use the **real pypdf library** against the real generated PDF. Why no mocks? `PyPDFParser` is a thin wrapper around pypdf. If you mock pypdf, you're just testing that you called the mock — you're not testing the parser at all. The only meaningful test is: does it extract the right text from a real PDF?

Tests cover: successful parse, correct page count, non-existent file raises `DocumentParseError`, non-PDF file raises `DocumentParseError`.

---

### 8.3 `test_chunkers.py` — 9 Tests, No Mocks

Chunker tests construct `ParsedDocument` objects in-memory with known text. No mocks needed — `RecursiveChunker` has no external dependencies. Tests cover: chunk count, chunk overlap, empty text raises `ChunkingError`, metadata propagation.

---

### 8.4 `test_dense_retriever.py` — 6 Tests with MagicMock

```python
mock_embedder = MagicMock(spec=OpenAIEmbedder)
mock_vector_store = MagicMock(spec=ChromaVectorStore)
```

Why mock here? `DenseRetriever.retrieve()` composes an embedder and a vector store. We're testing the **composition logic** (does it call `embed_single` then `similarity_search`? does it pass the embedding correctly?), not the embedder or vector store themselves — those have their own tests. Mocking at this boundary is correct.

`spec=OpenAIEmbedder` makes the mock reject calls to methods that don't exist on the real class — prevents tests from passing due to typos in method names.

---

### 8.5 `test_openai_generator.py` — 7 Tests

```python
@patch("src.generation.openai_generator.OpenAI")
def test_generate_returns_answer(mock_openai_class):
```

Patches at the module level — replaces the `OpenAI` class imported in `openai_generator.py`. The mock intercepts `client.chat.completions.create(...)` and returns a controlled response. Tests cover: citation parsing, "cannot answer" pass-through, API error → `GenerationError`, retry logic.

Why mock OpenAI here? Unlike the parser, the generator's behavior *is* OpenAI's response. We mock OpenAI to test citation parsing, error handling, and retry logic — behaviors we control — without spending money on real API calls or needing network access in CI.

---

### 8.6 `test_orchestrator.py` — 7 Tests

```python
mock_retriever = MagicMock(spec=Retriever)
mock_generator = MagicMock(spec=Generator)
orchestrator = RAGOrchestrator(retriever=mock_retriever, generator=mock_generator)
```

Tests the two-line orchestrator: does it call `retrieve`, pass results to `generate`, return the answer? Does it propagate `RetrievalError`? Does it propagate `GenerationError`? Does it wrap unexpected exceptions?

---

### 8.7 `test_end_to_end.py` — Integration Test

```python
RUN = bool(os.getenv("RUN_INTEGRATION_TESTS"))

@pytest.mark.skipif(not RUN, reason="Set RUN_INTEGRATION_TESTS=1 to enable")
def test_ingest_and_query_solar_system(sample_pdf_path):
    orchestrator, pipeline = get_rag_components()
    result = pipeline.ingest(sample_pdf_path, filename="sample.pdf")
    answer = orchestrator.answer("What is the largest planet in the Solar System?")
    assert "Jupiter" in answer.text
    assert len(answer.citations) > 0
```

**Why `skipif` with env var?**  
Integration tests require a real `OPENAI_API_KEY` and make real API calls (cost money). They should not run in CI by default or for developers who just want to run unit tests. The env var gate makes the intent explicit: opt-in, not opt-out.

**What this test proves:**  
The full pipeline — parse → chunk → embed → store → retrieve → generate — works end-to-end against a real PDF with real OpenAI calls. The Solar System PDF contains known facts; the assertion on "Jupiter" verifies the answer is grounded in the document.

---

## 9. Big Picture — End-to-End Traces

### 9.1 PDF Upload Trace

```
User clicks "Upload Document" in Streamlit
    │
    ▼
ui/app.py: requests.post(f"{API_URL}/ingest", files={"file": ...})
    │  HTTP POST /ingest  (multipart/form-data)
    ▼
api/routes/ingest.py: ingest_document(file: UploadFile, pipeline=Depends(get_pipeline))
    ├── extension check: filename.lower().endswith(".pdf")
    ├── contents = await file.read()
    ├── size check: len(contents) > max_bytes → 413
    ├── NamedTemporaryFile(suffix=".pdf", delete=False)
    │       tmp.write(contents)
    │
    ▼
src/ingestion/pipeline.py: pipeline.ingest(tmp_path, filename="report.pdf")
    ├── doc_id = uuid4()
    │
    ▼
src/ingestion/parsers.py: PyPDFParser.parse(tmp_path)
    ├── PdfReader(tmp_path)
    ├── page.extract_text() × N pages
    └── → ParsedDocument(text, pages, metadata)
    │
    ▼
src/ingestion/chunkers.py: RecursiveChunker.chunk(parsed_doc, doc_id)
    ├── RecursiveCharacterTextSplitter.split_text(text)
    └── → list[Chunk] (each with uuid id, text, metadata)
    │
    ▼
src/retrieval/embedders.py: OpenAIEmbedder.embed([chunk.text, ...])
    ├── openai client.embeddings.create(model="text-embedding-3-small", input=[...])
    └── → list[list[float]]  (one 1536-dim vector per chunk)
    │
    ▼
src/retrieval/vector_store.py: ChromaVectorStore.add_chunks(chunks, embeddings)
    ├── collection.add(ids, embeddings, documents, metadatas)
    └── → None  (persisted to disk at data/chroma_db/)
    │
    ▼
api/routes/ingest.py: finally: os.unlink(tmp_path)
    └── → IngestResponse(document_id, filename, chunk_count=42)
    │  HTTP 200  {"document_id": "...", "filename": "report.pdf", "chunk_count": 42}
    ▼
ui/app.py: st.success("Ingested report.pdf — 42 chunks")
    ├── st.session_state.ingested_docs.append("report.pdf")
```

---

### 9.2 Question-Answer Trace

```
User types "What is the capital of France?" and presses Enter
    │
    ▼
ui/app.py: requests.post(f"{API_URL}/query", json={"query": "What is the capital..."})
    │  HTTP POST /query  (application/json)
    ▼
api/routes/query.py: query_documents(body: QueryRequest, orchestrator=Depends(get_orchestrator))
    ├── vector_store.count() == 0 → 503 guard (passes if docs ingested)
    │
    ▼
src/orchestrator.py: RAGOrchestrator.answer("What is the capital of France?")
    │
    ▼
src/retrieval/dense_retriever.py: DenseRetriever.retrieve(query, top_k=5)
    │
    ▼
src/retrieval/embedders.py: OpenAIEmbedder.embed_single("What is the capital of France?")
    ├── openai client.embeddings.create(model="text-embedding-3-small", input=["What..."])
    └── → [0.023, -0.104, 0.891, ...]  (1536 floats)
    │
    ▼
src/retrieval/vector_store.py: ChromaVectorStore.similarity_search(query_embedding, top_k=5)
    ├── collection.query(query_embeddings=[...], n_results=5, include=[...])
    └── → list[RetrievedChunk]  (5 chunks, each with score = 1 - cosine_distance)
    │
    ▼  back to orchestrator
src/generation/openai_generator.py: OpenAIGenerator.generate(query, chunks)
    ├── _format_context(chunks) → "[chunk_0]:\n{text}\n\n[chunk_1]:\n{text}..."
    ├── CITATION_USER_PROMPT_TEMPLATE.format(context=..., query=...)
    ├── client.chat.completions.create(
    │       model="gpt-4o-mini",
    │       temperature=0.0,
    │       messages=[system_prompt, user_message])
    └── → "Paris is the capital of France. [chunk_0] It has been..."
    │
    ├── _parse_citations("...Paris...[chunk_0]...") → [0]
    └── → Answer(text="Paris is the capital...", citations=[0], retrieved_chunks=[...])
    │
    ▼  back to api/routes/query.py
api/routes/query.py: Answer → QueryResponse
    ├── enumerate(answer.retrieved_chunks) → CitationDetail × 5
    └── → QueryResponse(answer="Paris is...", citations=[0], retrieved_chunks=[...])
    │  HTTP 200  {"answer": "Paris is...", "citations": [0], "retrieved_chunks": [...]}
    ▼
ui/app.py: display answer in chat bubble
    ├── st.session_state.messages.append({"role": "assistant", "content": answer})
    └── _show_sources(retrieved_chunks)  ← expandable citation panels
```

---

## 10. Protocol Proof Table

This table maps every Protocol defined in `src/interfaces/base.py` to its Phase 1 concrete implementation, what a Phase 2 swap would look like, and which files change when you swap.

| Protocol | Phase 1 Implementation | Phase 2 Swap Example | Files Changed on Swap |
|---|---|---|---|
| `Parser` | `PyPDFParser` | `PDFPlumberParser` (better layout), `TesseractOCRParser` (scanned PDFs) | `src/ingestion/parsers.py`, `src/factory.py` |
| `Chunker` | `RecursiveChunker` | `SemanticChunker` (embedding-based sentence grouping) | `src/ingestion/chunkers.py`, `src/factory.py` |
| `Embedder` | `OpenAIEmbedder` | `CohereEmbedder`, `SentenceTransformerEmbedder` (local) | `src/retrieval/embedders.py`, `src/factory.py` |
| `VectorStore` | `ChromaVectorStore` | `PGVectorStore` (pgvector + PostgreSQL) | `src/retrieval/vector_store.py`, `src/factory.py` |
| `Retriever` | `DenseRetriever` | `HybridRetriever` (BM25 + dense + cross-encoder rerank) | `src/retrieval/`, `src/factory.py` |
| `Generator` | `OpenAIGenerator` | `AnthropicGenerator`, `OllamaGenerator` (local LLM) | `src/generation/`, `src/factory.py` |

**The invariant:** In every swap, only `src/factory.py` and the specific implementation file change. `src/orchestrator.py`, `api/`, `ui/`, and all tests that mock at Protocol boundaries continue working without modification.

**Protocol drift (Parser):**  
`base.py` declares `parse() -> list[str]`. `PyPDFParser.parse()` returns `ParsedDocument`. This drift exists because `ParsedDocument` carries richer data (`pages`, `metadata`) that `IngestionPipeline` needs. `IngestionPipeline` uses concrete types. Phase 2 will update the `Parser` Protocol to return `ParsedDocument` and reconcile the drift.

---

## Appendix — Key Design Principles Demonstrated

| Principle | Where It Appears |
|---|---|
| **Ports and Adapters** | `src/` = domain, `api/` and `ui/` = adapters. UI has zero `src/` imports. |
| **Dependency Inversion** | `RAGOrchestrator` typed to `Retriever` and `Generator` Protocols |
| **Single Responsibility** | Each class does one thing: parse, chunk, embed, store, retrieve, generate, orchestrate |
| **12-Factor Config** | All settings from env via `pydantic-settings`, no hardcoded values |
| **Composition Root** | `src/factory.py` is the only place that instantiates concrete classes |
| **Fail Fast** | Custom exceptions (`DocumentParseError`, `RetrievalError`, etc.) at domain boundaries |
| **Observability First** | `time.perf_counter()` per pipeline step, structlog JSON on every operation |
| **Test at the Right Boundary** | Mock OpenAI (external), use real pypdf (library wrapper), mock at Protocol boundaries |

---

*DocuVerse Phase 1 — walkthrough generated 2026-05-15*
