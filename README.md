# DocuVerse

![Python](https://img.shields.io/badge/python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45-red)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
![Phase](https://img.shields.io/badge/phase-0%20%E2%80%94%20Foundation-orange)

**DocuVerse** is a production-grade Retrieval-Augmented Generation (RAG) system. Upload PDFs, ask questions in natural language, and receive cited answers grounded in your documents.

---

## What It Does

| Feature | Status |
|---|---|
| PDF upload & parsing | Phase 1 |
| Semantic chunking | Phase 1 |
| Vector embedding & storage | Phase 2 |
| Similarity retrieval | Phase 2 |
| LLM-generated cited answers | Phase 3 |
| Evaluation harness | Phase 4 |
| CI/CD + observability | Phase 5 |

---

## Current Phase Status

**Phase 0 — Foundation** is complete. The scaffold is wired end-to-end:

- `GET /health` returns `{"status": "ok"}`
- Streamlit UI renders "Hello DocuVerse"
- All Protocol interfaces are defined; no implementations yet
- `docker compose up` starts both services

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        DocuVerse                        │
│                                                         │
│  ┌──────────┐   HTTP    ┌──────────────────────────┐   │
│  │Streamlit │ ────────▶ │  FastAPI  (api/)          │   │
│  │  (ui/)   │           │  /health  /upload  /ask   │   │
│  └──────────┘           └────────────┬─────────────┘   │
│                                      │                  │
│                               calls  │                  │
│                                      ▼                  │
│                          ┌───────────────────────┐      │
│                          │  RAGOrchestrator       │      │
│                          │  (src/orchestrator.py) │      │
│                          └──┬──────────┬──────────┘      │
│                             │          │                  │
│                  ┌──────────▼──┐  ┌────▼──────────┐      │
│                  │  Retriever  │  │  Generator    │      │
│                  │  Protocol   │  │  Protocol     │      │
│                  └──────────┬──┘  └────┬──────────┘      │
│                             │          │                  │
│              ┌──────────────▼──┐  ┌────▼──────────┐      │
│              │ VectorStore     │  │  OpenAI GPT   │      │
│              │ (Phase 2: Chroma│  │  (Phase 3)    │      │
│              │  / pgvector)    │  └───────────────┘      │
│              └─────────────────┘                         │
│                                                          │
│  src/ = business logic (pure Python, no HTTP)            │
│  api/ = HTTP adapter  │  ui/ = Streamlit adapter         │
└─────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI 0.115 + Uvicorn |
| UI | Streamlit 1.45 |
| Config | Pydantic-Settings + .env |
| Logging | Structlog (JSON) |
| Embedding | OpenAI text-embedding-3-small |
| LLM | OpenAI gpt-4o-mini |
| Vector DB | Chroma (Phase 2) |
| Testing | Pytest + HTTPX |
| Linting | Ruff |
| Typing | Mypy (strict) |
| Container | Docker + Docker Compose |

---

## Running Locally

### Prerequisites

- Python 3.12+
- Docker & Docker Compose (for containerised run)

### Quick start (local)

```bash
# 1. Clone and enter the repo
git clone https://github.com/your-username/docuverse.git
cd docuverse

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install all dependencies
make install-dev

# 4. Copy env template and add your OpenAI key
cp .env.example .env
# edit .env: set OPENAI_API_KEY=sk-...

# 5a. Start the API server
make api         # http://localhost:8000

# 5b. In a separate terminal, start the UI
make ui          # http://localhost:8501
```

### Docker Compose

```bash
cp .env.example .env   # fill in OPENAI_API_KEY
make docker-up         # builds and starts api + ui
```

### Run tests

```bash
make test
```

---

## Planned Phases

| Phase | Goal |
|---|---|
| **0** | Project scaffold, health endpoint, Streamlit shell ✅ |
| **1** | PDF ingestion — parse, chunk, validate |
| **2** | Embedding + Chroma vector store + retrieval |
| **3** | LLM generation with cited answers |
| **4** | Evaluation harness (RAGAS metrics) |
| **5** | CI/CD, observability, production hardening |

---

## Project Structure

```
docuverse/
├── api/            # FastAPI HTTP adapter
├── config/         # Pydantic-Settings + YAML component selection
├── data/           # Sample PDFs (gitignored after Phase 1)
├── docs/           # Architecture notes and iteration log
├── scripts/        # One-off utility scripts
├── src/            # Business logic — all Protocols + implementations
│   ├── ingestion/  # Parser, Chunker protocols
│   ├── retrieval/  # Embedder, VectorStore, Retriever protocols
│   ├── generation/ # Generator protocol
│   └── utils/      # Logger, exceptions, Pydantic models
├── tests/          # Pytest unit + integration suites
└── ui/             # Streamlit frontend adapter
```

---

## License

MIT © 2025 DocuVerse Contributors
