# DocuVerse — Architecture

> This document is a living skeleton. Sections marked **[TBD — Phase N]** will be filled in as each phase ships.

---

## Guiding Principles

1. **Ports and Adapters (Hexagonal Architecture)** — business logic in `src/` has zero knowledge of HTTP, Streamlit, or any I/O framework. Adapters in `api/` and `ui/` translate between the outside world and domain objects.

2. **Protocol-based interfaces** — every replaceable component (parser, embedder, vector store, generator) is defined as a `typing.Protocol` in `base.py`. Concrete implementations are swapped via `config/config.yaml` without touching business logic.

3. **12-Factor config** — all environment-specific values come from environment variables, loaded via Pydantic-Settings. No hardcoded credentials or paths.

4. **Async-first API** — FastAPI endpoints use `async def`. Blocking I/O (file reads, vector DB calls) will be wrapped with `asyncio.to_thread` in later phases.

---

## Module Map

| Module | Responsibility |
|---|---|
| `config/` | Load and validate env vars; select component implementations |
| `src/ingestion/` | Parse raw PDFs → pages → chunks |
| `src/retrieval/` | Embed text; store and query vectors |
| `src/generation/` | Call LLM with context; format cited answer |
| `src/orchestrator.py` | Coordinate ingestion → retrieval → generation pipeline |
| `api/` | HTTP adapter — expose orchestrator over FastAPI |
| `ui/` | Streamlit adapter — user-facing chat and upload interface |

---

## Data Flow

```
PDF Upload
   │
   ▼
[Parser]          src/ingestion/  — extracts raw text per page
   │
   ▼
[Chunker]         src/ingestion/  — splits text into overlapping Chunk objects
   │
   ▼
[Embedder]        src/retrieval/  — converts Chunk.text → float vectors
   │
   ▼
[VectorStore]     src/retrieval/  — persists vectors; supports similarity search
   │
   ╔══════════════════════════════╗
   ║  Query Time                  ║
   ╚══════════════════════════════╝
   │
[Retriever]       src/retrieval/  — embeds query, returns top-K RetrievedChunks
   │
   ▼
[Generator]       src/generation/ — builds prompt with chunks, calls LLM, returns Answer
   │
   ▼
Answer(text, citations, retrieved_chunks)
```

---

## Component Selection (config.yaml)

[TBD — Phase 2]

The `config/config.yaml` file will declare which concrete class satisfies each Protocol.
Example:

```yaml
ingestion:
  parser: src.ingestion.pdf_parser.PyMuPDFParser
  chunker: src.ingestion.chunker.RecursiveCharacterChunker

retrieval:
  embedder: src.retrieval.openai_embedder.OpenAIEmbedder
  vector_store: src.retrieval.chroma_store.ChromaStore
  retriever: src.retrieval.similarity_retriever.SimilarityRetriever

generation:
  generator: src.generation.openai_generator.OpenAIGenerator
```

---

## API Surface

[TBD — Phase 1]

Planned endpoints:

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe ✅ |
| `POST` | `/documents` | Upload and ingest a PDF |
| `POST` | `/ask` | Submit a query; receive cited answer |
| `GET` | `/documents` | List ingested documents |

---

## Evaluation Strategy

[TBD — Phase 4]

RAGAS metrics: faithfulness, answer relevancy, context precision, context recall.

---

## Observability

[TBD — Phase 5]

- Structured JSON logs via `structlog` (Phase 0 ✅)
- Prometheus metrics endpoint
- OpenTelemetry traces
