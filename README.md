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

---

### Evaluation — Diagnostic Framework

**Decision:** Run the four standard RAG metrics (Faithfulness, Contextual Precision, Contextual Recall, Answer Relevancy) on a small golden set, then deliberately break individual pipeline components and observe which metrics react. The point isn't the absolute scores — it's building the intuition for *which metric flags which failure mode*.

**What each metric actually measures (and what it doesn't):**

| Metric | What it computes | What it does NOT detect |
|---|---|---|
| **Faithfulness** | Decomposes the *answer* into atomic claims; for each claim, asks the judge LLM whether retrieved context supports it. Score = supported claims / total claims. | Whether the retrieved context is *correct*. A confidently wrong answer grounded in a wrong-but-retrieved chunk scores 1.00. |
| **Contextual Precision** | For each retrieved chunk, asks "is this relevant to the input?" Weighted by rank (top positions matter more). Reflects reranker quality. | Whether the *right* chunks were retrieved at all (that's recall). |
| **Contextual Recall** | Decomposes the *ground-truth answer* into claims; checks each against retrieved chunks. Score = attributable claims / total claims. **Requires `expected_output`.** | Anything about the generated answer — purely a retrieval-side metric. |
| **Answer Relevancy** | Generates N hypothetical questions the answer *could* be answering; measures cosine similarity to the original input. | Whether the answer is *correct* — only whether it's *on-topic* for the question. |

**Failure-mode experiment (Config B2: bad embedder, row 1 of synthetic testset):**

Query: *"What Mia can help with?"*  Reference: §9 of `book-ordering-process.md`.

| Config | Change | Faithfulness | Ctx Precision | Ctx Recall | Answer Relevancy |
|---|---|---|---|---|---|
| Baseline | `text-embedding-3-small` + `gpt-4o-mini` + neutral prompt | 1.00 | 1.00 | 1.00 | 1.00 |
| B2 — bad embedder | swap embedder → `multi-qa-MiniLM-L6-cos-v1` (384-D) | **1.00** | 0.95 | **0.00** | **1.00** |
| D — bad generator | swap LLM → `gpt-3.5-turbo` + "be creative, fill gaps" prompt | 1.00 | 1.00 | 1.00 | **0.88** |

**The dangerous result:** Faithfulness and Answer Relevancy both stayed at 1.00 *while the retriever was completely failing*. Recall went to 0.00 — the gold chunk was not in top-5; the small model surfaced lexically-adjacent chunks ("explaining processes", "routing to support") from unrelated docs instead.

**Why this matters:** Faithfulness can't catch retriever failures because it only checks answer-vs-context consistency, not context-vs-truth. Answer Relevancy can't catch them either because it only measures topical alignment, not correctness. **Contextual Recall is the only one of the four that requires ground truth, and therefore the only one that flags this failure mode.**

**Operational takeaway:** Production RAG eval needs a golden set with ground-truth answers — not just LLM-as-judge on free-form output. Without recall, a system can score 1.00 on three metrics and still be silently wrong.

**Why Config D didn't break Faithfulness (the failed-prediction lesson):**

The prediction was: weaker LLM + a "be creative, fill gaps" prompt should drop Faithfulness via hallucination. It didn't. Faithfulness stayed at 1.00.

Reason: Faithfulness only drops when the model makes claims **not in the retrieved context**. Row 1's gold chunk (§9 of `book-ordering-process.md`) was retrieved cleanly and contains a complete answer to "What Mia can help with?" — so even with `gpt-3.5-turbo` and an aggressive creative-license prompt, the model had **no gap to fill**. It paraphrased the chunk instead of inventing.

The only signal of generator degradation was a **0.88 Answer Relevancy** (vs 1.00 baseline) — likely from creative padding pulling the answer slightly off-topic for the input. Faithfulness is thus a *necessary but not sufficient* check on the generator: it can only flag hallucination when retrieval has left room for it.

**Methodological consequence:** to stress-test the generator in isolation, pair the bad-generator config with a query whose gold chunk is **partial** or **missing** from retrieval — then the model is forced to either say "I don't know" or hallucinate, and Faithfulness becomes diagnostic again.

**Faithfulness — additional behaviors worth knowing:**

- **Weak model, good chunks → Faithfulness can drop while retrieval is fine.** A weaker LLM may ignore or paraphrase past the relevant chunk, producing claims that aren't grounded even though the right context was retrieved. Precision/Recall will look healthy; Faithfulness alone reflects the generator weakness.
- **Strong model, bad prompt → Faithfulness can still drop.** A capable model with a poorly-written prompt (e.g. one that encourages generalization, summarization, or tone-shifting) may emit claims that don't match the retrieved chunks. The retriever isn't at fault; the prompt is.
- **A faithful answer is not always a correct answer; an unfaithful answer is not always a wrong one.** Faithfulness only checks consistency between answer and retrieved context. An answer that doesn't match the retrieved chunks may still be correct (e.g. it draws on the model's parametric knowledge to address the query). In that case, Faithfulness drops but the system actually behaved well — and conversely, a perfectly faithful answer can still be wrong if the chunks themselves were wrong. **Faithfulness is a consistency check, not a correctness check, in either direction.**

**Third gap in the four-metric framework (Config D' finding):**

| Gap | Symptom | What to add |
|---|---|---|
| Faithfulness can't catch retriever failure | Faithfulness 1.00 while answer is silently wrong | Contextual Recall (needs ground truth) |
| Faithfulness can't catch generator drift when retrieval covers the answer | Faithfulness 1.00; only Answer Relevancy wobbles | Direct correctness metric (e.g. `GEval` answer-vs-expected) |
| Recall can't evaluate "answer should be 'I don't know'" cases | Recall 0.00 even when the system correctly refuses | Refusal/abstention check (assert the answer contains a hedge or "not specified" phrase) |
