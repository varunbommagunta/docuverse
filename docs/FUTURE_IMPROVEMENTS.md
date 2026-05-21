# Future Improvements — ML/AI Upgrade Roadmap

The current DocuVerse system demonstrates production engineering depth: modular architecture, containerized deployment, scientific evaluation, and 95 passing unit tests. This document plans the next phase, focused on ML modeling depth. Improvements are organized by pipeline stage — from PDF parsing through answer evaluation — and ranked by the ML signal each upgrade adds to the portfolio. Each stage lists its current implementation, concrete limitations, and prioritized upgrade options with realistic effort estimates.

---

## The 10-Stage ML/AI Pipeline

DocuVerse has 10 distinct ML/AI stages from raw PDF to final answer, each with independent upgrade opportunities.

```
              RAW PDF
                 │
                 ▼
       ┌──────────────────┐
       │  1. PARSING      │  ← PyPDF, text extraction
       └────────┬─────────┘
                ▼
       ┌──────────────────┐
       │  2. CHUNKING     │  ← RecursiveChunker, 500/50
       └────────┬─────────┘
                ▼
       ┌──────────────────┐
       │  3. EMBEDDING    │  ← OpenAI text-embedding-3-small
       └────────┬─────────┘
                ▼
       ┌──────────────────┐
       │  4. STORAGE      │  ← ChromaDB (vector store)
       └────────┬─────────┘
                │
                │  (query time)
                │
    QUERY ──────┤
                ▼
       ┌──────────────────┐
       │  5. RETRIEVAL    │  ← Dense / BM25 / Hybrid (RRF)
       └────────┬─────────┘
                ▼
       ┌──────────────────┐
       │  6. RERANKING    │  ← Cross-encoder MS-MARCO
       └────────┬─────────┘
                ▼
       ┌──────────────────┐
       │  7. PROMPT BUILD │  ← Citation template
       └────────┬─────────┘
                ▼
       ┌──────────────────┐
       │  8. GENERATION   │  ← gpt-4o-mini API
       └────────┬─────────┘
                ▼
       ┌──────────────────┐
       │  9. POST-PROCESS │  ← Regex citation parsing
       └────────┬─────────┘
                ▼
       ┌──────────────────┐
       │ 10. EVALUATION   │  ← RAGAS metrics
       └──────────────────┘
                ▼
              ANSWER
```

---

## Stage-by-Stage Improvements

---

### Stage 1 — Parsing

**Current:** PyPDF library extracts plain text from PDFs.

**Limitations:**
- Loses table structure (tables become text blobs)
- Loses figure/image content entirely
- Doesn't handle scanned PDFs (no OCR)
- Reading order can be wrong on multi-column layouts (Constitution has these)

**Improvement Options:**

| Improvement | ML Signal | Effort | Description |
|---|---|---|---|
| Unstructured.io chunked partition | 🟢🟢 | 1 weekend | Library that extracts tables, figures, headers with structure preserved; maintains element type metadata |
| LayoutLMv3 / Donut | 🟢🟢🟢 | 2 weekends | Vision-language models trained on document layout; correctly extracts text + structure + tables + figures |
| OCR via Tesseract or PaddleOCR | 🟢 | 1 weekend | For scanned PDFs; lower complexity, less novel |

**Recommendation:** Start with Unstructured.io (quick + high-value). LayoutLM later for image-heavy documents.

---

### Stage 2 — Chunking

**Current:** RecursiveChunker splits at 500 characters with 50 char overlap.

**Limitations:**
- Article boundaries are ignored (Article gets split mid-sentence)
- No semantic awareness (splits at character counts, not meaning)
- Constitution structure is lost (Article 21's text might be split across 3 chunks)
- No metadata for hierarchy

**Improvement Options:**

| Improvement | ML Signal | Effort | Description |
|---|---|---|---|
| Semantic chunking | 🟢🟢🟢 | 1 weekend | Use embedding similarity between adjacent sentences; split at topic boundaries |
| Document-aware chunking for legal text | 🟢🟢🟢 | 1–2 weekends | Parse Article/Chapter structure; each Article = one chunk with hierarchical metadata |
| Hierarchical chunking (parent-child) | 🟢🟢 | 1 weekend | Small chunks for retrieval, large parent chunks for LLM context |

**Recommendation:** Document-aware chunking for Constitution + ARC. Distinctive (most projects don't do this), works perfectly for legal text, strong interview story.

---

### Stage 3 — Embedding

**Current:** OpenAI text-embedding-3-small (1536-dim, API call per chunk).

**Limitations:**
- Not domain-adapted (general English, doesn't understand Indian legal terminology)
- Expensive (every chunk costs API calls)
- Latency (round-trip to OpenAI)
- Vendor lock-in

**Improvement Options:**

| Improvement | ML Signal | Effort | Description |
|---|---|---|---|
| Fine-tune BGE-small embedder | 🟢🟢🟢 | 1–2 weekends | Contrastive learning on Indian government text; BAAI/bge-small-en-v1.5 (33MB); MultipleNegativesRankingLoss |
| Better off-the-shelf embedder | 🟢🟢 | 1 day | Swap to BAAI/bge-large-en-v1.5 or all-mpnet-base-v2; no training needed |
| Knowledge distillation (mimic OpenAI) | 🟢🟢🟢 | 2 weekends | Train a tiny model to produce embeddings similar to text-embedding-3-small |

**Recommendation:** Fine-tune BGE-small — this is the single highest-ML-signal upgrade. Demonstrates real model training and reduces inference cost ~99%.

---

### Stage 4 — Storage

**Current:** ChromaDB vector store on disk.

**Limitations:**
- No metadata indexing (can't pre-filter by chapter or article)
- Single vector per chunk
- No hybrid index structures (vector and BM25 are separate)

**Improvement Options:**

| Improvement | ML Signal | Effort | Description |
|---|---|---|---|
| Metadata filtering | 🟢🟢 | 1 weekend | Use ChromaDB where filter; combine with vector search for hybrid pre-filtered retrieval |
| ColBERT-style multi-vector | 🟢🟢🟢 | 2 weekends | One embedding per token; late interaction at query time; significantly more accurate |
| Better vector DB (Qdrant/Weaviate) | 🟢 | 1 weekend | More infrastructure than ML |

**Recommendation:** Add metadata filtering after document-aware chunking. Quick, sets up other improvements, demonstrates retrieval engineering depth.

---

### Stage 5 — Retrieval

**Current:** Dense + BM25 + RRF fusion (Phase 4 work).

**Limitations:**
- Static fusion weights (RRF treats dense/BM25 equally regardless of query type)
- No query understanding (same strategy for all query types)
- No query expansion (short queries underperform)

**Improvement Options:**

| Improvement | ML Signal | Effort | Description |
|---|---|---|---|
| HyDE (Hypothetical Document Embeddings) | 🟢🟢🟢 | 1 weekend | LLM generates fake answer; embed that; retrieve using fake answer's embedding |
| Query routing with classifier | 🟢🟢🟢 | 1–2 weekends | Train/use classifier that detects query type; route to optimal retrieval strategy |
| Multi-query retrieval | 🟢🟢 | 1 weekend | LLM generates paraphrases; retrieve for each; merge |
| Learned-to-rank fusion weights | 🟢🟢🟢 | 2 weekends | Train a model to predict optimal RRF weights per query |

**Recommendation:** HyDE first (cheap, single weekend, big impact on cross-chunk queries). Query routing second.

---

### Stage 6 — Reranking

**Current:** MS-MARCO cross-encoder (point-wise scoring).

**Limitations:**
- Point-wise scoring kills diversity (caused -10.7% cross-chunk regression)
- General-domain training (MS-MARCO is web search, not legal text)
- No multi-vector consideration

**Improvement Options:**

| Improvement | ML Signal | Effort | Description |
|---|---|---|---|
| Fine-tune custom reranker | 🟢🟢🟢 | 1–2 weekends | Fine-tune cross-encoder/ms-marco-MiniLM-L-6-v2 or BAAI/bge-reranker-base on your corpus |
| List-wise reranker swap | 🟢🟢🟢 | 4 hours | BGE-Reranker-v2 or ColBERT; scores chunks considering full candidate set |
| LLM-as-reranker | 🟢🟢 | 1 weekend | Small LLM scores each chunk; expensive per query but flexible |

**Recommendation:** Train custom reranker after fine-tuning embedder. Combined with custom embedder, gives two custom ML components in your portfolio.

---

### Stage 7 — Prompt Building

**Current:** Static citation-forcing template.

**Limitations:**
- No few-shot examples
- No context compression (feeds raw chunks even if redundant)
- No query-specific adaptation

**Improvement Options:**

| Improvement | ML Signal | Effort | Description |
|---|---|---|---|
| DSPy for prompt optimization | 🟢🟢 | 1 weekend | Library that systematically optimizes prompts via examples |
| Contextual compression | 🟢🟢 | 1 weekend | Smaller model extracts only relevant sentences before LLM call |
| Chain-of-thought with verification | 🟢🟢🟢 | 1–2 weekends | LLM generates answer + reasoning; second LLM verifies citations |

**Recommendation:** Contextual compression is the strong middle ground. DSPy is impressive but adds dependency complexity.

---

### Stage 8 — Generation

**Current:** gpt-4o-mini API.

**Limitations:**
- Vendor lock-in (quality depends on OpenAI)
- Cost (every query costs money)
- Latency (network round-trip)
- No fine-tuning (can't adapt to domain)

**Improvement Options:**

| Improvement | ML Signal | Effort | Description |
|---|---|---|---|
| Fine-tune Phi-3-mini or Llama-3.2-3B with LoRA | 🟢🟢🟢 | 2–3 weekends | Train on (query, retrieved_chunks, ground_truth_answer) triples; runs locally |
| Multi-provider abstraction (Claude/Gemini) | 🟢🟢 | 1 weekend | LLM_STRATEGY env var; compare empirically; shows tool selection skill |
| Constrained decoding for citations | 🟢 | 1 weekend | Force LLM to output specific [chunk_N] syntax via grammar-constrained generation |

**Recommendation:** LoRA fine-tuning on Phi-3-mini is the moonshot. Massive ML signal. 92% of API quality at 1/100th the cost.

---

### Stage 9 — Post-Processing

**Current:** Regex extraction of [chunk_N] patterns.

**Limitations:**
- Brittle (depends on LLM following format)
- No verification (doesn't check if citations actually support the answer)
- No deduplication

**Improvement Options:**

| Improvement | ML Signal | Effort | Description |
|---|---|---|---|
| NLI faithfulness verification | 🟢🟢🟢 | 1 weekend | NLI model checks: does chunk N actually support claim X? Flag unsupported claims |
| Citation deduplication and ranking | 🟢 | 1 day | Order by relevance, remove duplicates; pure engineering |

**Recommendation:** NLI faithfulness verification is the strong addition. No training required, catches real hallucinations, demonstrates ML literacy.

---

### Stage 10 — Evaluation

**Current:** RAGAS with 4 standard metrics.

**Limitations:**
- Generic metrics (don't catch all hallucination types)
- No human evaluation
- Limited query diversity (40 test queries)
- No A/B framework

**Improvement Options:**

| Improvement | ML Signal | Effort | Description |
|---|---|---|---|
| Custom domain-specific metrics | 🟢🟢🟢 | 1–2 weekends | Citation correctness, refusal correctness, multi-hop accuracy; each metric is Python function + LLM judge |
| Synthetic evaluation queries | 🟢🟢🟢 | 1 weekend | LLM generates (query, ground_truth) pairs; expand from 40 to 400 queries |
| A/B testing framework | 🟢🟢 | 1 weekend | Two strategies in parallel; track which RAGAS prefers |

**Recommendation:** Custom metrics + synthetic data generation would dramatically strengthen the evaluation story.

---

## Recommended Implementation Order

| Weekend | Stage | Improvement | Why This Order |
|---|---|---|---|
| 1 | Stage 2 | Document-aware chunking | Foundation that improves all downstream stages |
| 2–3 | Stage 3 | Fine-tune BGE-small embedder | Highest ML-signal upgrade; needs good chunks first |
| 4 | Stage 5 | Add HyDE query expansion | Quick win for cross-chunk queries |
| 5–6 | Stage 6 | Train custom reranker | Fixes cross-chunk regression at root |
| 7–8 | Stage 10 | Custom metrics + synthetic eval | Validates all prior improvements rigorously |

After 8 focused weekends, the project would have 3 custom-trained ML components, sophisticated retrieval techniques, domain-specific evaluation, the same clean architecture, and an interview story that demonstrates ML engineering depth alongside production engineering.

---

## ML Signal Rating Guide

- 🟢🟢🟢 = Significant ML signal — demonstrates model training, novel techniques, or substantial domain expertise
- 🟢🟢 = Moderate ML signal — demonstrates retrieval engineering or applied ML thoughtfulness
- 🟢 = Low ML signal — useful improvements but mostly engineering rather than ML

---

## See Also

- [../README.md](../README.md) — Project overview
- [./ITERATION_LOG.md](./ITERATION_LOG.md) — Engineering decisions and measured improvements to date
- [./ARCHITECTURE.md](./ARCHITECTURE.md) — System architecture deep-dive
- [./EVALUATION.md](./EVALUATION.md) — Current evaluation methodology
