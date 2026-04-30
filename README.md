# Production RAG System

A production-ready RAG system built with **Haystack**, featuring CRAG, semantic caching, and hybrid retrieval.

## Architecture

- **Framework**: Haystack for document processing and retrieval
- **Runtime**: FastAPI backend with SSE streaming
- **Data Pipeline**: Haystack indexing pipeline (offline)
- **Infrastructure**: Qdrant (vector DB), Redis (semantic cache)

## Features

- **Haystack Pipelines**: Built-in document processing, embedding, and retrieval
- **CRAG**: Self-correcting RAG with citations
- **Semantic Cache**: Redis HNSW for query caching
- **Hybrid Retrieval**: Dense + BM25 + RRF fusion (Haystack components)
- **Adaptive Routing**: Multi-source query routing
- **Security**: Input/output guards, PII redaction

## Quick Start

```bash
# Start infrastructure
docker-compose up -d

# Install dependencies
uv sync

# Run API
uv run uvicorn app.main:app --reload
```

## Project Structure

```
advanced-rag/
├── app/              # FastAPI application
├── routes/           # API endpoints
├── services/         # Custom business logic (wraps Haystack)
├── retrieval/        # Haystack retrieval pipelines
├── agents/           # Agentic layer (CRAG, routing)
├── prompts/          # Versioned prompts
├── security/         # Guards & filters
└── pipeline/         # Haystack indexing pipeline
```

## Design Decisions

### Chunking Strategy

**Decision:** Hybrid — embed whole document for short files, split by H2 headers for longer ones.

**Context:** The knowledge base is structured Markdown files (300–1,200 words each), organized as single-topic documents with consistent H2 sections (`Purpose`, `Who This Is For`, `Boundaries`, etc.) and a JSON metadata block at the end.

**Options considered:**

| Option | Reasoning |
|---|---|
| Fixed-size / recursive splitting | Generic default. Ignores the document's own structure. Would split mid-section, fragmenting intentionally authored units. |
| Document-aware (header splitting only) | Respects semantic boundaries. Risk: some H2 sections are very short (3–5 lines), producing tiny fragments that hurt answer quality — the same failure mode documented in FloTorch 2026 (43-token chunks → 54% accuracy). |
| Embed whole document, no chunking | Best for short, focused docs per Firecrawl analysis and the 2026 benchmark guide. Risk: `all-MiniLM-L6-v2` has a 256-token limit and silently truncates longer files. |
| Semantic chunking | High retrieval recall (91.9% in Chroma) but poor end-to-end accuracy on short corpora. Vectara NAACL 2025 found fixed-size consistently outperformed it on realistic datasets. Not justified here. |

**Chosen approach:**
- Files ≤ ~400 words → embed whole, no chunking (document is already a self-contained unit)
- Files > ~400 words → split by H2 headers, merging any section under ~100 words with the next to avoid tiny fragments
- Strip the JSON metadata block from chunk content; store its fields (`module`, `journey_stage`, `user_type`) as Qdrant payload for filter-at-query-time

**Why this works for this corpus:** Each file is a single focused topic written with intentional section boundaries. Splitting by those boundaries gives retrieval precision (a query about "who is MIA for" retrieves that section, not the whole doc) while the merge rule prevents the fragment problem.

**References:**
- [RAG Chunking Strategies: The 2026 Benchmark Guide](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/)
- Vectara / FloTorch 2026 benchmark — recursive 512t: 69%, semantic: 54%
- Firecrawl: chunking actively hurts retrieval on short, focused documents

---

## Haystack Components Used

- **Document Processing**: PDFToTextConverter, DocumentCleaner, DocumentSplitter
- **Embedding**: SentenceTransformersDocumentEmbedder
- **Retrieval**: QdrantEmbeddingRetriever, InMemoryBM25Retriever, DocumentJoiner
- **Generation**: OpenAIGenerator with PromptBuilder
