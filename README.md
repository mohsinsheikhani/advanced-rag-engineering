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
| Fixed-size / recursive splitting | Splits on token count, not meaning. This corpus has intentional H2 section boundaries — recursive splitting ignores them and cuts mid-section. Sections use implicit references ("This program", "These users") that only resolve with the section header present; splitting breaks that context. Also unnecessary: files are 300–1,200 words with multiple sections, so individual H2 chunks already fit well within the model's 256-token limit without further splitting. |
| Document-aware (header splitting only) | Respects semantic boundaries. Risk: some H2 sections are very short (3–5 lines), producing tiny fragments that hurt answer quality — the same failure mode documented in FloTorch 2026 (43-token chunks → 54% accuracy). |
| Embed whole document, no chunking | Best for short, focused docs per Firecrawl analysis and the 2026 benchmark guide. Risk: `all-MiniLM-L6-v2` has a 256-token limit and silently truncates longer files. |
| Semantic chunking | Embeds every sentence to detect topic shifts — computationally expensive at index time. Also unnecessary for this corpus: topic shifts are already marked by H2 headers, so semantic similarity detection adds cost with no benefit. |

**Chosen approach:**
- Files ≤ ~400 words → embed whole, no chunking (document is already a self-contained unit)
- Files > ~400 words → split by H2 headers using Haystack's `MarkdownHeaderSplitter`
- Filter boilerplate sections by header text (`Document Header`, `Gentle Invitation`, `Gentle Next-Step Framing`) — these contain no retrieval signal
- Strip the JSON metadata block from chunk content; store its fields (`module`, `journey_stage`, `user_type`) as Qdrant payload for filter-at-query-time

**Validation — the core question:** *What is the minimum unit of text that, handed alone to an LLM, produces a correct answer to a realistic user query?*

Answered by reading the corpus directly: each H2 section is self-contained and answers a distinct question. Sections do not reference each other or rely on surrounding context. The isolation problem (chunks that only make sense with parent context) does not apply here.

**Multi-question queries** (e.g. "what is MIA and who is it for?") span multiple chunks by design — handled at retrieval time via `top_k` and query decomposition, not by changing chunk boundaries.

**Why this works for this corpus:** Each file is a single focused topic written with intentional section boundaries. Splitting by those boundaries gives retrieval precision (a query about "who is MIA for" retrieves that section, not the whole doc). Boilerplate filtering removes noise before it reaches the vector store.

**References:**
- [RAG Chunking Strategies: The 2026 Benchmark Guide](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/)
- Vectara / FloTorch 2026 benchmark — recursive 512t: 69%, semantic: 54%
- Firecrawl: chunking actively hurts retrieval on short, focused documents

---

### Embedding Model

**Decision:** `text-embedding-3-small` (OpenAI API, 1536 dimensions)

**Initial choice:** Started with `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` (local, 384 dimensions) — same size and cost as `all-MiniLM-L6-v2` but trained specifically for asymmetric query-to-passage retrieval, which matches this use case (short user queries vs paragraph-length chunks).

**What worked:** Most queries retrieved correctly with high confidence (0.66–0.76). Fast, free, no API dependency.

**What didn't work:** Vocabulary mismatch on queries like "Who is MIA for?" and "How do I make money with MIA?" The model couldn't bridge the gap between plain user language ("make money") and the corpus's abstract phrasing ("earn income", "generate residual income", "representation income"). The correct chunks existed but never surfaced in top 15 results. Query expansion (3 LLM-generated rephrasings) didn't resolve it — the vocabulary gap was too systematic.

**Root cause:** The corpus is ChatGPT-generated and uses aspirational/abstract language throughout. A 22M-parameter model trained on curated QA pairs hasn't seen enough examples of "make money" and "earn income" used interchangeably to place them close in vector space.

**Switch to `text-embedding-3-small`:** OpenAI's model is orders of magnitude larger, trained on massive internet-scale data. It has seen "make money", "earn income", "ways to earn", "profit" used as near-synonyms in thousands of contexts, so they're genuinely close in its vector space.

**Results after switch:**
- "How do I make money with MIA?" — top 3 chunks all from `how-income-is-generated-in-mia.md`, scores 0.629–0.595 (vs 0.46–0.51 bunched with noise before)
- "Who is MIA for?" — `who-mia-is-for-and-not-for.md` chunks surface in top 3, correct chunk from `what-is-mia.md` at position 4 (vs not in top 15 before)

**Tradeoff accepted:** API cost and latency per query instead of local/free. At current scale (300 queries/month), cost is negligible (~$0.02/1M tokens = fractions of a cent). At 1M+ queries/month, this becomes a real cost line to monitor.

**Token limit:** 8191 tokens. All chunks in this corpus are H2 sections (30–180 words, ~40–240 tokens) — well within the limit.

---

### Retrieval Strategy

**Current:** Dense-only retrieval via Qdrant (`QdrantEmbeddingRetriever`, `top_k=5`)

**Findings from manual evaluation (6 test queries):**

With `text-embedding-3-small`, most queries now retrieve correctly with strong confidence scores and clean relevance separation. The vocabulary mismatch problems observed with the smaller model are largely resolved.

**Remaining edge case — "Who is MIA for?"**
The most specific chunk (`## 4. Who This Is For` from `what-is-mia.md`) ranks at position 4–5, while more general chunks about MIA rank higher. However, the top 3 results (`who-mia-is-for-and-not-for.md` sections) are still directly relevant and arguably provide better answers than the generic section.

**Why hybrid retrieval (BM25 + dense) is still planned:** While the better embedding model closed most vocabulary gaps, BM25 would provide additional precision for keyword-heavy queries and act as a safety net for edge cases where semantic similarity alone isn't sufficient. This is the standard production pattern for robust retrieval.

---

- **Document Processing**: PDFToTextConverter, DocumentCleaner, DocumentSplitter
- **Embedding**: SentenceTransformersDocumentEmbedder
- **Retrieval**: QdrantEmbeddingRetriever, InMemoryBM25Retriever, DocumentJoiner
- **Generation**: OpenAIGenerator with PromptBuilder
