# Production RAG System

A working RAG backend that I used as a sandbox to think through, measure, and write down the decisions that actually matter in production. The chat path is small on purpose. Most of the work lives in the engineering decisions section below, where each choice is backed by numbers, tradeoffs, and a short note on what I'd change next.

If you're a recruiter, hiring manager, or founder evaluating me, the fastest read is the [Headline results](#headline-results) and then any one of the [engineering decision sections](#engineering-decisions). Each section is self-contained.

**Tech:** Python, FastAPI, Haystack, Qdrant, Redis Stack, OpenAI (`text-embedding-3-small`, `gpt-4o-mini`), Langfuse, DeepEval, SSE streaming, semantic caching, RAG evaluation, observability, Docker.

**Contact:** [linkedin.com/in/mohsin-sheikhani](https://www.linkedin.com/in/mohsin-sheikhani/)

---

> **A note on what this is.** This is a public, high level view of a project I built for a customer. The system, the methodology, and the eval harness are real and runnable. The full customer corpus, the production traces, and the calibrated final judge numbers are not exposed here. That is why `knowledge_base/` is in `.gitignore`. I took permission to share the bare minimum needed to make my work credible, and nothing more.
>
> Where this version uses a smaller synthetic dataset, a dev-split number, or a single-row failure-mode experiment instead of the full customer result, I say so plainly rather than dressing it up. The numbers in this README are the ones I actually measured on this code with the inputs I had permission to share.

---

## Table of contents

1. [Headline results](#headline-results)
2. [Stack at a glance](#stack-at-a-glance)
3. [What's built vs. what's planned](#whats-built-vs-whats-planned)
4. [Quick start](#quick-start)
5. [Project structure](#project-structure)
6. Engineering decisions
   - Index time
     - [Chunking strategy](#chunking-strategy)
     - [Embedding model](#embedding-model)
     - [Vector database](#vector-database)
   - Query time
     - [Retrieval strategy](#retrieval-strategy)
     - [Evaluation, the diagnostic framework](#evaluation-the-diagnostic-framework)
   - Serving and operations
     - [FastAPI streaming backend](#fastapi-streaming-backend)
     - [Caching layers](#caching-layers)
     - [Production architecture and cost at scale](#production-architecture-and-cost-at-scale)
     - [Observability with Langfuse](#observability-with-langfuse)
7. [What's next](#whats-next)
8. [References](#references)

---

## Headline results

The numbers below were all measured on this stack, not pulled from a blog. Each one links to the section that explains how it was produced and what it means.

- **Streaming endpoint feels 2.6x faster than the blocking one.** `stream TTFT P50 = 1737ms` vs `sync total P50 = 4483ms` over 50 paired requests. The LLM did not get faster, the user just sees tokens earlier. ([details](#fastapi-streaming-backend))
- **L1 exact-match cache cuts P50 by ~500x.** `sync P50: 3911ms → 7ms` on repeated questions. The tail barely moved, which is the honest takeaway. ([details](#l1-exact-match))
- **L2 semantic cache cuts P50 by ~6.4x on rephrased questions.** `sync P50: 3911ms → 610ms` across 50 paraphrased queries. Tuned the cosine threshold to 0.85 after watching the Langfuse similarity distribution on real misses. ([details](#l2-semantic-match))
- **Recall is the only one of the four standard RAG metrics that catches a broken retriever.** Faithfulness and Answer Relevancy both stayed at 1.00 while contextual recall went to 0.00. A system can score perfectly on three metrics and still be silently wrong. ([details](#evaluation-the-diagnostic-framework))
- **Switching from a small local embedder to `text-embedding-3-small` fixed retrieval failures the small model could not solve.** The smaller model could not bridge "make money" and "earn income" in this corpus. Query expansion did not save it. The cost difference at this scale is a coffee per month. ([details](#embedding-model))
- **At 1M queries/month, the all-in bill is around $400 to $600.** LLM input tokens dominate by an order of magnitude, not model choice. Cache hit rate is the second biggest lever. ([details](#production-architecture-and-cost-at-scale))

---

## Stack at a glance

| Layer | Choice | Why |
|---|---|---|
| Framework | Haystack | Built-in indexing + retrieval components, swappable pieces, fewer custom adapters to write. |
| Web layer | FastAPI + SSE | Async-first, native streaming, low ceremony. |
| Vector store | Qdrant (managed) | Filtered HNSW for metadata-aware search, managed so I don't own ops. |
| Cache | Redis Stack | Same box covers L1 GET and L2 vector KNN. One dependency, two cache types. |
| Embeddings | OpenAI `text-embedding-3-small` (1536-D) | Large enough vocabulary to bridge user phrasing and corpus phrasing. |
| Generation | OpenAI `gpt-4o-mini` | Cheap, good enough for this corpus. Eval is the gate for upgrading. |
| Eval | DeepEval | Four-metric framework plus deliberate failure-mode experiments. |
| Observability | Langfuse (drop-in OpenAI wrapper) | One-line integration, auto-captures cost, tokens, latency, errors. |

---

## What's built vs. what's planned

Being honest about this is more useful than overclaiming. The chat path is intentionally minimal so latency and cache experiments aren't muddied by extra components.

**Wired into the live request path (`/chat/sync`, `/chat/stream`):**
- Dense retrieval against Qdrant (`text-embedding-3-small`, `top_k=5`)
- L1 exact-match cache (Redis GET on hashed query text)
- L2 semantic cache (Redis Stack KNN over query embeddings)
- SSE streaming with a dedicated `ttft` event
- Langfuse tracing on every request (spans for cache-L1, embed, cache-L2, retrieve, generation)

**Scaffolded but not yet in the request path:**
- CRAG (self-correcting RAG): file in `agents/`, not wired in
- Adaptive multi-source query router: file in `agents/`, not wired in
- Input/output guards and PII redaction: files in `security/`, not called from chat routes
- Document grader, query decomposer, reranker: files in `services/` and `retrieval/`, used in offline scripts only

**Planned next:**
- Hybrid retrieval (BM25 + dense + RRF). See [Retrieval strategy](#retrieval-strategy).
- Embedding cache to retire L1 cleanly. See [Caching layers](#caching-layers).
- User-feedback scores attached to Langfuse trace IDs. See [Observability](#observability-with-langfuse).

---

## Quick start

```bash
# Start infrastructure (Qdrant + Redis Stack)
docker-compose up -d

# Install dependencies (uv)
uv sync

# Run the API
uv run uvicorn app.main:app --reload
```

Environment variables live in `.env.example`. The required keys are `OPENAI_API_KEY`, `QDRANT_*`, `REDIS_*`, and the three `LANGFUSE_*` keys.

To index the knowledge base:

```bash
uv run python scripts/index_documents.py
```

To run the latency benchmark:

```bash
uv run python scripts/bench_latency.py --rows 0,1,2,3,4 --repeats 10
```

---

## Project structure

```
advanced-rag/
├── app/              FastAPI app: main.py, config, document_store wiring
├── routes/           HTTP endpoints. The chat path lives in routes/chat.py
├── services/         Pipeline glue and helpers (semantic cache lives here)
├── retrieval/        Haystack retrieval pipelines and filter helpers
├── pipeline/         Offline indexing pipeline and chunkers
├── agents/           Agentic components (scaffolded, see "Built vs planned")
├── prompts/          Versioned prompt templates
├── security/         Guards and content filters (scaffolded)
├── eval/             DeepEval configs and failure-mode experiments
├── scripts/          Bench harness, indexing scripts, test scripts
└── knowledge_base/   Source markdown corpus (~226 files)
```

The chat path is short by design. `routes/chat.py` is the file to read if you want to see the wired flow end to end.

---

# Engineering decisions

Each section below is one decision, structured as: what I chose, what I considered, why, what the numbers say, and what I'd revisit. They're ordered by where they sit in the request lifecycle.

---

## Chunking strategy

**Decision:** Hybrid approach. Embed the whole document for short files, and split by H2 headers for longer ones.

**Context:** The knowledge base is structured Markdown files (300 to 1,200 words each). Each file is a single-topic document with consistent H2 sections (`Purpose`, `Who This Is For`, `Boundaries`, etc.) and a JSON metadata block at the end.

**Options considered:**

| Option | Reasoning |
|---|---|
| Fixed-size / recursive splitting | Splits on token count, not meaning. This corpus has intentional H2 section boundaries, and recursive splitting ignores them and cuts mid-section. Sections also use implicit references ("This program", "These users") that only resolve with the section header present, so splitting breaks that context. It's also unnecessary here: files are 300 to 1,200 words with multiple sections, so individual H2 chunks already fit well within the model's 256-token limit. |
| Document-aware (header splitting only) | Respects semantic boundaries. The risk is that some H2 sections are very short (3 to 5 lines), producing tiny fragments that hurt answer quality. This is the same failure mode documented in FloTorch 2026 (43-token chunks gave 54% accuracy). |
| Embed whole document, no chunking | Best for short, focused docs per Firecrawl analysis and the 2026 benchmark guide. The risk is that `all-MiniLM-L6-v2` has a 256-token limit and silently truncates longer files. |
| Semantic chunking | Embeds every sentence to detect topic shifts, which is computationally expensive at index time. Also unnecessary for this corpus: topic shifts are already marked by H2 headers, so semantic similarity detection adds cost with no benefit. |

**Chosen approach:**
- Files of ~400 words or less get embedded whole, with no chunking (the document is already a self-contained unit).
- Files over ~400 words get split by H2 headers using Haystack's `MarkdownHeaderSplitter`.
- Filter boilerplate sections by header text (`Document Header`, `Gentle Invitation`, `Gentle Next-Step Framing`). These contain no retrieval signal.
- Strip the JSON metadata block from chunk content. Store its fields (`module`, `journey_stage`, `user_type`) as Qdrant payload for filter-at-query-time.

**Validation, the core question:** *What is the minimum unit of text that, handed alone to an LLM, produces a correct answer to a realistic user query?*

This was answered by reading the corpus directly. Each H2 section is self-contained and answers a distinct question. Sections do not reference each other or rely on surrounding context. The isolation problem (chunks that only make sense with parent context) does not apply here.

**Multi-question queries** (e.g. "what is MIA and who is it for?") span multiple chunks by design. They're handled at retrieval time via `top_k` and query decomposition, not by changing chunk boundaries.

**Why this works for this corpus:** Each file is a single focused topic written with section boundaries. Splitting by those boundaries gives retrieval precision (a query about "who is MIA for" retrieves that section, not the whole doc). Boilerplate filtering removes noise before it reaches the vector store.

---

## Embedding model

**Decision:** `text-embedding-3-small` (OpenAI API, 1536 dimensions)

**Initial choice:** Started with `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` (local, 384 dimensions). It's the same size and cost as `all-MiniLM-L6-v2` but trained specifically for asymmetric query-to-passage retrieval, which matches this use case (short user queries vs paragraph-length chunks).

**What worked:** Most queries retrieved correctly with high confidence (0.66 to 0.76). Fast, free, no API dependency.

**What didn't work:** Vocabulary mismatch on queries like "Who is MIA for?" and "How do I make money with MIA?". The model couldn't bridge the gap between plain user language ("make money") and the corpus's abstract phrasing ("earn income", "generate residual income", "representation income"). The correct chunks existed but never surfaced in the top 15 results. Query expansion (3 LLM-generated rephrasings) didn't resolve it either. The vocabulary gap was too systematic.

**Root cause:** The corpus uses aspirational and abstract language throughout. A 22M-parameter model trained on curated QA pairs hasn't seen enough examples of "make money" and "earn income" used interchangeably to place them close in vector space.

**Switch to `text-embedding-3-small`:** OpenAI's model is orders of magnitude larger, trained on massive internet-scale data. It has seen "make money", "earn income", "ways to earn", and "profit" used as near-synonyms in thousands of contexts, so they're genuinely close in its vector space.

**Results after switch:**
- "How do I make money with MIA?" returned its top 3 chunks all from `how-income-is-generated-in-mia.md`, with scores 0.629 to 0.595 (vs 0.46 to 0.51 bunched with noise before).
- "Who is MIA for?" surfaced `who-mia-is-for-and-not-for.md` chunks in the top 3, with the correct chunk from `what-is-mia.md` at position 4 (vs not in the top 15 before).

**Tradeoff accepted:** API cost and latency per query instead of local and free. Cost looks like nothing at prototype scale, which is exactly how it blindsides later.

**Two cost surfaces to think about separately.** Embedding has two distinct lines, and they behave differently as the product grows:

1. *One-time indexing* of the corpus. Pay once, sleep well. Only re-runs when documents change.
2. *Per-query embedding*. Runs on every search. This is the line that scales with users and quietly compounds.

**Numbers on `text-embedding-3-small` at $0.02 per 1M tokens** ([pricing reference](https://developers.openai.com/api/docs/models/text-embedding-3-small)):

| Surface | Volume | Calc | Cost |
|---|---|---|---|
| One-time indexing | 50M tokens | 50 × $0.02 | **$1.00** (one-off) |
| Per-query, ~20 tokens/query | 1M queries/month → 20M tokens/month | 20 × $0.02 | **$0.40 / month** |
| Per-query at 10x growth | 10M queries/month → 200M tokens/month | 200 × $0.02 | **$4.00 / month** |

So at this corpus size, indexing is a coffee. Query embedding at 1M/month is also a coffee. The point of writing them down is the shape of the bill, not the dollar amount. Indexing is fixed (cheap forever unless the corpus changes), and per-query is linear in traffic (still cheap on `-small`, but worth knowing the slope before swapping to `-large` which is roughly 6.5x more expensive on the same workload).

**Token limit:** 8191 tokens. All chunks in this corpus are H2 sections (30 to 180 words, ~40 to 240 tokens), well within the limit.

---

## Vector database

**Decision:** Qdrant Cloud (managed).

**How the choice was made.** Three questions, in order:

1. *Self-hosted or managed?* Self-hosted means a VM, upgrades, backups, and a 3am page when it falls over. Managed means a credit card. At this stage of the project there's nothing interesting to learn from running the box, so managed wins.
2. *How much data, and how often does it change?* The corpus is ~226 structured markdown files, embedded at 1536 dimensions (`text-embedding-3-small`). Writes are rare and only triggered when source documents change. The workload sits comfortably inside a single managed cluster, with clear headroom before any sharding conversation needs to happen.
3. *What's the filter story?* Retrieval needs metadata filters (`module`, `journey_stage`, `user_type` stored as payload). Qdrant has first-class filtered HNSW, so filters happen inside the search, not as a post-filter step that wrecks recall.

**Options considered:**

| Option | Why not |
|---|---|
| pgvector on existing Postgres | No existing Postgres in this project, so it would mean adding a database to avoid adding a database. If the rest of the stack ever moves to Postgres, this becomes the obvious choice. |
| Self-hosted Qdrant (docker-compose) | Was the original choice during local development. Fine for the dev loop, but means owning ops in production for no real upside at this scale. |
| Pinecone | Filters are applied post-search, which can hurt recall when filters are selective. Also more expensive than Qdrant Cloud at the smallest paid tier. |
| Weaviate Cloud | Comparable on features, slightly heavier API surface. No strong reason to switch given Qdrant already works locally. |

**Cost surfaces, the ones that actually show up on the bill:**

1. **Storage.** Grows with `(vectors × dimension × 4 bytes)` plus payload and HNSW index overhead. Rough rule: a 1536-D float32 vector is ~6KB raw, ~10 to 12KB after index overhead. At 1M vectors that's ~10GB; the current corpus sits comfortably below the smallest managed tier.
2. **Queries / reads.** Qdrant Cloud prices on cluster size (RAM/CPU), not per-query, so reads are effectively bundled into the cluster line. On Pinecone, per-query pricing is explicit and you watch it.
3. **Writes / upserts.** Only matters if the corpus churns. This one doesn't.
4. **Egress.** Cross-region reads are the silent killer on managed services. Put the cluster in the same region as the API.
5. **Ops time.** Self-hosted Qdrant means you own backups, upgrades, and HA. Managed earns its subscription back the first time an outage gets handled without anyone on your team waking up at 3am.

**Where this stack actually sits today:**

| Item | Today | At 1M vectors (1536-D) |
|---|---|---|
| Storage footprint | well under 1GB | ~10GB |
| Qdrant Cloud tier | smallest managed cluster | larger managed cluster, roughly $50 to $150 / month depending on RAM |
| Query cost | bundled into cluster price | bundled into cluster price, not per-query |
| Ops effort | none | none (still managed) |

**When to revisit.** If the corpus crosses ~1M vectors, or if filters get complex enough that recall@k drops, or if the rest of the stack consolidates onto Postgres. None of those are close.

---

## Retrieval strategy

**Current:** Dense-only retrieval via Qdrant (`QdrantEmbeddingRetriever`, `top_k=5`)

**Findings from manual evaluation (6 test queries):**

With `text-embedding-3-small`, most queries now retrieve correctly with strong confidence scores and clean relevance separation. The vocabulary mismatch problems observed with the smaller model are largely resolved.

**Remaining edge case, "Who is MIA for?":**
The most specific chunk (`## 4. Who This Is For` from `what-is-mia.md`) ranks at position 4 or 5, while more general chunks about MIA rank higher. That said, the top 3 results (`who-mia-is-for-and-not-for.md` sections) are still directly relevant and arguably provide better answers than the generic section.

**Why hybrid retrieval (BM25 + dense) is still planned:** The better embedding model closed most vocabulary gaps, but BM25 would provide additional precision for keyword-heavy queries and act as a safety net for edge cases where semantic similarity alone isn't sufficient. This is the standard production pattern for robust retrieval.

---

## Evaluation, the diagnostic framework

**Decision:** Run the four standard RAG metrics (Faithfulness, Contextual Precision, Contextual Recall, Answer Relevancy) on a small golden set. Then deliberately break individual pipeline components and observe which metrics react. The point isn't the absolute scores. It's building the intuition for *which metric flags which failure mode*.

**What each metric actually measures (and what it doesn't):**

| Metric | What it computes | What it does NOT detect |
|---|---|---|
| **Faithfulness** | Decomposes the *answer* into atomic claims. For each claim, asks the judge LLM whether retrieved context supports it. Score = supported claims / total claims. | Whether the retrieved context is *correct*. A confidently wrong answer grounded in a wrong-but-retrieved chunk scores 1.00. |
| **Contextual Precision** | For each retrieved chunk, asks "is this relevant to the input?". Weighted by rank (top positions matter more). Reflects reranker quality. | Whether the *right* chunks were retrieved at all (that's recall). |
| **Contextual Recall** | Decomposes the *ground-truth answer* into claims, then checks each against retrieved chunks. Score = attributable claims / total claims. **Requires `expected_output`.** | Anything about the generated answer. It's purely a retrieval-side metric. |
| **Answer Relevancy** | Generates N hypothetical questions the answer *could* be answering, then measures cosine similarity to the original input. | Whether the answer is *correct*. Only whether it's *on-topic* for the question. |

**Failure-mode experiment (Config B2: bad embedder, row 1 of synthetic testset):**

Query: *"What Mia can help with?"*  Reference: §9 of `book-ordering-process.md`.

| Config | Change | Faithfulness | Ctx Precision | Ctx Recall | Answer Relevancy |
|---|---|---|---|---|---|
| Baseline | `text-embedding-3-small` + `gpt-4o-mini` + neutral prompt | 1.00 | 1.00 | 1.00 | 1.00 |
| B2, bad embedder | swap embedder to `multi-qa-MiniLM-L6-cos-v1` (384-D) | **1.00** | 0.95 | **0.00** | **1.00** |
| D, bad generator | swap LLM to `gpt-3.5-turbo` + "be creative, fill gaps" prompt | 1.00 | 1.00 | 1.00 | **0.88** |

**The dangerous result:** Faithfulness and Answer Relevancy both stayed at 1.00 *while the retriever was completely failing*. Recall went to 0.00. The gold chunk was not in top-5. The small model surfaced lexically-adjacent chunks ("explaining processes", "routing to support") from unrelated docs instead.

**Why this matters:** Faithfulness can't catch retriever failures because it only checks answer-vs-context consistency, not context-vs-truth. Answer Relevancy can't catch them either because it only measures topical alignment, not correctness. **Contextual Recall is the only one of the four that requires ground truth, and therefore the only one that flags this failure mode.**

**Operational takeaway:** Production RAG eval needs a golden set with ground-truth answers, not just LLM-as-judge on free-form output. Without recall, a system can score 1.00 on three metrics and still be silently wrong.

**Why Config D didn't break Faithfulness (the failed-prediction lesson):**

The prediction was that a weaker LLM plus a "be creative, fill gaps" prompt should drop Faithfulness via hallucination. It didn't. Faithfulness stayed at 1.00.

The reason: Faithfulness only drops when the model makes claims **not in the retrieved context**. Row 1's gold chunk (§9 of `book-ordering-process.md`) was retrieved cleanly and contains a complete answer to "What Mia can help with?". So even with `gpt-3.5-turbo` and an aggressive creative-license prompt, the model had **no gap to fill**. It paraphrased the chunk instead of inventing.

The only signal of generator degradation was a **0.88 Answer Relevancy** (vs 1.00 baseline), likely from creative padding pulling the answer slightly off-topic for the input. Faithfulness is thus a *necessary but not sufficient* check on the generator. It can only flag hallucination when retrieval has left room for it.

**Methodological consequence:** to stress-test the generator in isolation, pair the bad-generator config with a query whose gold chunk is **partial** or **missing** from retrieval. Then the model is forced to either say "I don't know" or hallucinate, and Faithfulness becomes diagnostic again.

**Faithfulness, additional behaviors worth knowing:**

- **Weak model, good chunks: Faithfulness can drop while retrieval is fine.** A weaker LLM may ignore or paraphrase past the relevant chunk, producing claims that aren't grounded even though the right context was retrieved. Precision/Recall will look healthy. Faithfulness alone reflects the generator weakness.
- **Strong model, bad prompt: Faithfulness can still drop.** A capable model with a poorly-written prompt (e.g. one that encourages generalization, summarization, or tone-shifting) may emit claims that don't match the retrieved chunks. The retriever isn't at fault. The prompt is.
- **A faithful answer is not always a correct answer, and an unfaithful answer is not always a wrong one.** Faithfulness only checks consistency between answer and retrieved context. An answer that doesn't match the retrieved chunks may still be correct (e.g. it draws on the model's parametric knowledge to address the query). In that case, Faithfulness drops but the system actually behaved well. Conversely, a perfectly faithful answer can still be wrong if the chunks themselves were wrong. **Faithfulness is a consistency check, not a correctness check, in either direction.**

**Third gap in the four-metric framework (Config D' finding):**

| Gap | Symptom | What to add |
|---|---|---|
| Faithfulness can't catch retriever failure | Faithfulness 1.00 while answer is silently wrong | Contextual Recall (needs ground truth) |
| Faithfulness can't catch generator drift when retrieval covers the answer | Faithfulness 1.00, only Answer Relevancy wobbles | Direct correctness metric (e.g. `GEval` answer-vs-expected) |
| Recall can't evaluate "answer should be 'I don't know'" cases | Recall 0.00 even when the system correctly refuses | Refusal/abstention check (assert the answer contains a hedge or "not specified" phrase) |

---

## FastAPI streaming backend

**Goal:** make the user-perceived latency of the chat endpoint a measured number, not a vibe. Ship two endpoints, bench both, prove the streaming win.

**Two endpoints, kept deliberately minimal** (no cache, no guards, no reranker, since those distort the comparison):

| Endpoint | Shape | Purpose |
|---|---|---|
| `POST /chat/sync` | `{query, top_k}` → `{answer, sources, retrieval_ms, total_ms}` | non-streaming baseline |
| `POST /chat/stream` | `{query, top_k}` → SSE stream | streaming, with explicit TTFT signal |

**SSE event contract** (`/chat/stream`):

| event | payload | when |
|---|---|---|
| `meta` | `{retrieval_ms, sources}` | after retrieval, before generation |
| `ttft` | `{ttft_ms}` (server-side) | first non-empty content chunk from the LLM |
| `token` | `{text}` | each generated token |
| `done` | `{ttft_ms, total_ms, retrieval_ms, tokens, tps}` | stream complete |

The dedicated `ttft` event matches the production pattern (CloudPortableTech RAG benchmark, vLLM). It lets server and client agree on exactly when generation started, without inferring it from the first `token` frame. The first non-empty content chunk is what counts. OpenAI emits a role-only chunk first, which NVIDIA GenAI-Perf and LLMPerf both ignore by convention.

**Why client-side timing is the source of truth.** Server-side `ttft_ms` can't see network/TLS, response buffering by uvicorn or a reverse proxy, or async scheduling jitter. All of those are things the user feels. The bench script records both: `client_ttft_ms` is the headline number, and `server_ttft_ms` is for attribution (`client - server` = network + buffering budget).

**Bench harness, `scripts/bench_latency.py`:**

```bash
# 1. Start the API
uv run uvicorn app.main:app --port 8000

# 2. In another shell
uv run python scripts/bench_latency.py --rows 0,1,2,3,4 --repeats 3
```

Defaults: 5 fixed rows from `eval/datasets/synthetic_testset_raw.csv`, 3 repeats each, 2 throwaway warmup requests, `temperature=0`, `top_k=5`. Sequential (concurrency=1), because concurrency adds queuing time that obscures the streaming-vs-sync delta on a small N. Raw timings are persisted to `scripts/bench_results.csv` so aggregations don't require re-running.

**Why 3 repeats × 5 queries (N=15).** Three repeats of the *same* query is for **statistical stability**, not simulating users. It smooths per-request noise (network jitter, OpenAI prefill variance) so the percentiles are stable across runs. Simulating concurrent users is a different axis (`asyncio.gather` over the inner loop, not yet wired). That measures system behavior under load, not single-request speed.

**Actual results, stable run (5 rows × 10 repeats, N=50 per endpoint):**

```
                                   P50           P95          mean       N
sync   total (client)            4483ms        6140ms        4546ms     50
stream TTFT (client)             1737ms        3178ms        1957ms     50
stream total (client)            4240ms        6615ms        4371ms     50
stream TTFT (server)             1730ms        3166ms        1948ms     50
  perceived-latency win (P50): sync_total / stream_ttft = 2.6x faster
```

The first pass at N=15 (3 repeats) had P50s within 10% of these, but P95 for stream TTFT undershot at 2333ms. The real tail is 3178ms. **N=15 P50 was solid, N=15 P95 was a lie of small numbers.** That's the operational lesson: SLO percentiles need N≥50 to be trustworthy.

**The win is the TTFT row vs. sync total.** Stream-total stayed roughly equal to sync-total. Streaming doesn't make the LLM faster, it just exposes tokens earlier. If stream-total had dropped materially, something else would have changed (model swap, cache hit) and the comparison would be dirty.

**Reading the numbers, five takeaways:**

1. **Retrieval is the elephant.** Per-request retrieval ranged from 440ms to **4132ms** for the *same query repeated*, with a median around 800ms. That's roughly half of stream-TTFT. The next bottleneck to attack isn't streaming. It's whatever is making retrieval slow and inconsistent. (Likely culprit: OpenAI's embedding API, not Qdrant. Confirmed in the Langfuse traces by inspecting the `retrieve` span's nested `OpenAI-embedding` child.)

   ```
   stream TTFT P50: 1737ms
                   ├─ retrieval median:  ~800ms   ← biggest single component
                   └─ LLM prefill+net:   ~900ms
   ```

2. **Stream TTFT P95/P50 ratio = 1.83×.** That's within the 2 to 3× range industry benchmarks (NVIDIA NIM, BentoML) report for streaming endpoints. Sync total P95/P50 = 1.37× (narrower) because waiting for full generation averages out prefill variance.

3. **N=15 to N=50 deltas tell a story.** P50 numbers held within ~10% (central tendency was right the first time), but stream TTFT P95 jumped 2333ms to 3178ms (+36%). The real tail only revealed itself with more samples. Sync P95 actually dropped 6568ms to 6140ms because the original N=15 had one 4132ms retrieval spike that dominated. Spread over 50 samples, it averaged in.

4. **Client-server TTFT gap is ~5 to 10ms** across the board. Loopback overhead is negligible, so the server-side `ttft_ms` is trustworthy in this environment. When deployed behind nginx/cloudflare, that gap will widen and the **client number becomes the only one to trust**.

5. **Tokens scale with query complexity** as expected. Short factual queries are ~130 tokens, and multi-part / philosophical queries are 200 to 250 tokens. Stream-total tracks tokens. TTFT does not. That's the structural argument for streaming: TTFT is **input-bound**, total is **output-bound**, so streaming wins more on long answers.

**Why the 2.5× win is honest but undersells streaming.** Both endpoints share retrieval, so the headline ratio is dragged down by the retrieval cost that streaming doesn't help with. Isolating generation only (`stream_ttft − retrieval_ms` vs `sync_total − retrieval_ms`), the win on the LLM portion is closer to **3.5 to 4×**. Stack-wide TTFT will improve more from fixing retrieval than from any further streaming work.

**Why P50 + P95, not mean.** Industry benchmarks (NVIDIA NIM, BentoML, Anyscale) report percentiles because the mean lies on tail-heavy distributions. P95/P50 routinely runs 2 to 3× on streaming endpoints. SLOs are written against P95 for that reason.

**Next steps (in priority order):**

1. **Investigate retrieval variance.** Read `embed_ms` vs `qdrant_ms` off the Langfuse trace (the `retrieve` span has an `OpenAI-embedding` child). If embedding dominates, batching/caching common queries gets a bigger TTFT win than streaming did.
2. **Layer concurrency** (`--concurrency N`). TTFT under load is where streaming's perceptual win matters most. Queued users still see *something* while waiting.

---

## Caching layers

There are three different cache layers worth knowing about. Two are built (L1, L2). One is planned (embedding cache). They sit at different stages of a request and they save different costs.

**Quick comparison.**

| Layer | Keyed by | Returns | Saves | Catches |
|---|---|---|---|---|
| **L1** | exact text hash | full answer | embed + retrieve + LLM | exact repeats |
| **Embedding cache** | exact text hash | embedding vector | OpenAI embedding call | exact repeats |
| **L2** | embedding vector | full answer | retrieve + LLM | rephrasings |

**Where each one sits in the request.**

```
question "how do I get paid"
   │
   ▼
L1 lookup (Redis GET on sha256 of normalized text)
   ├─ hit  → return cached answer. ~5ms. Done.
   └─ miss → keep going
   │
   ▼
Embedding cache lookup (Redis GET on sha256 of text)   ← planned, not built
   ├─ hit  → use the cached vector. No OpenAI call.
   └─ miss → call OpenAI to embed (~200 to 1200ms), store the vector for next time
   │
   ▼
L2 lookup (KNN over stored embeddings in Redis Stack)
   ├─ hit  → return cached answer (if similarity ≥ 0.92). ~5ms.
   └─ miss → keep going
   │
   ▼
Qdrant retrieval → LLM → store answer in L1 and L2
```

### L1 (exact match)

**The problem.** After the streaming work, P50 TTFT was around 1.4 seconds and P95 was about 1.9 seconds. The biggest single cost inside that was the OpenAI embedding call, which by itself can take a full second on a bad day. Calling OpenAI every single time a user asks the same question is wasteful. If two people ask "how do I get paid" five minutes apart, the answer is the same, and we already paid to compute it once.

**What L1 does.** Before doing any work, we check Redis to see if we already answered this exact question recently. The cache key is just a hash of the cleaned-up question (lowercased, extra spaces removed). If it's there, we return the stored answer in a few milliseconds. If it's not, we do the normal pipeline and save the answer for next time. The stored value lives for one hour and then expires on its own.

This is the simplest kind of cache. Same question in, same answer out. It does not understand that "how do I get paid" and "how do I receive payment" mean the same thing. That's what L2 handles.

**Where it sits in the request flow.**

```
question → check Redis (a few ms)
            ├─ hit  → return cached answer. Done.
            └─ miss → embed → search Qdrant → call LLM → save to Redis → return.
```

**What it stores.**

```
Redis entry:
  key:   cache:l1:<sha256("how do i get paid")>
  value: {"answer": "...", "sources": [...]}
```

**What it saves.** Everything below it in the stack. On a hit you pay one Redis GET, around 5ms. Skips OpenAI embed + Qdrant + OpenAI completion.

**What it catches.** Exact text repeats. Page refresh, retry button, copy-paste, the same FAQ asked twice.

**Why it's worth having.** The cost saved is huge, and the implementation is tiny (~10 lines, no schema, no index). Even at modest exact-repeat rates the cost-benefit is overwhelming.

**The numbers, cache off vs cache on** (50 calls each, 5 unique questions repeated 10 times):

```
                       Cache OFF        Cache ON         What changed
                       ----------       ----------       --------------------
sync   P50               3911 ms             7 ms        ~500x faster
sync   P95               6542 ms          4342 ms        ~33% faster
stream TTFT P50          1426 ms             8 ms        ~180x faster
stream TTFT P95          1917 ms          1418 ms        ~25% faster
```

**Reading this without jargon.** P50 is the typical experience. P95 is the worst experience that 1 in 20 users will see. The cache made the typical experience way better, but the worst-case only got a little better. Why?

Because 5 out of those 50 calls are "first time we've ever seen this question". The cache can't help those. They still pay the full ~4 second cost. P95 lands right on top of those slow ones, so P95 stays slow.

If you think about it from a user's seat: the first person to ask "how do I get paid" today waits 4 seconds. Everyone else who asks the same thing for the next hour waits 7 milliseconds. That's the win L1 buys.

**Where L1 helps and where it doesn't.**

| Situation | Does L1 help? |
|---|---|
| User asks the same question multiple times | Yes, huge. |
| Two users ask the exact same question | Yes. |
| User asks "how do I get paid" and later "how do I receive payment" | No. Different text, cache misses. L2 fixes this. |
| Brand new question nobody has asked before | No. First request always pays full cost. |

**The honest takeaway.** L1 moves the median latency a lot. It barely moves the tail. To move the tail you have to either make the cache catch near-duplicate questions too (that's L2), or make the slow path itself faster (cache the embedding call separately, use a smaller model, etc).

**What came next.** L2 semantic cache. Same idea, but instead of matching on exact text, match on meaning using the question's embedding. Starting threshold 0.92 cosine similarity (raise it if we start seeing wrong cached answers come back). That's where the real production hit rate comes from, because in real traffic almost nobody asks the same exact words twice.

**How to turn it off.** Set `CACHE_ENABLED=false` before starting the API. Useful for benchmarking the cold path or debugging.

```bash
CACHE_ENABLED=false uv run uvicorn app.main:app --port 8000
```

**Bench reproducibility note.** The benchmark used to run sync and stream interleaved per question, which meant the second endpoint always saw a warm cache populated by the first. That made the second endpoint's numbers look unrealistically good. The bench now flushes Redis between the sync phase and the stream phase, so each phase sees its own cold-path misses. Pass `--no-flush-between-phases` to get the old behavior back if you want to simulate a session that hits both endpoints.

### L2 (semantic match)

**What it stores.** The full answer, keyed by the *embedding* of the original question.

```
Redis entry (Redis Stack HASH):
  key:   cache:l2:<random-uuid>
  value: {
    embedding: <1536 float32 bytes>   ← this is the actual lookup key
    answer:    "..."
    sources:   [...]
    query:     "how do I get paid"     ← original text, kept for debugging
  }
```

A Redis Stack vector index (`cache:l2:idx`) sits on top of the `embedding` field. The UUID in the Redis key is just storage plumbing.

**What it does.** Embeds the incoming question, then runs a K-nearest-neighbor search over the stored embeddings. If the closest stored entry is within the similarity threshold (currently 0.92 cosine similarity), return that entry's answer.

**What it saves.** Retrieval (Qdrant) + LLM call. Still pays the embedding cost because you need the embedding to do the lookup.

**What it catches.** Rephrasings. "how do I get paid", "how can I receive payment", "what's the payment process" should all match the same entry if their embeddings are close enough.

**The threshold knob.** Currently 0.92. Lower means more hits but more risk of returning a wrong cached answer for a query that's only superficially similar. Higher means safer but you miss more rephrasings. Watch Langfuse `cache-l2.output.similarity` on real traffic to tune it. For RAG (where a wrong cached answer is worse than a slow correct one) something in the 0.92 to 0.97 range is normal.

**Why we built it.** L1 only helps on exact repeats, which in real traffic is a small fraction. Most users phrase the same question slightly differently each time. L2 is what actually lifts the production hit rate from "small" to "30 to 70%".

**The numbers, measured on rephrased queries.** The original L1 bench can't show L2 doing anything, because it asks the exact same query 10 times in a row. L1 catches all of those before L2 ever gets a turn. So I added a second bench mode (`--paraphrase-mode`) that sends 1 original plus 9 hand-written rephrasings per base query, 50 unique queries per endpoint with no two identical. That forces L1 to miss every single time, which is exactly what we need to see what L2 is actually doing. The paraphrases live in `eval/datasets/bench_paraphrases.json` if you want to read or change them.

```
                       Cache OFF        L1+L2 on (paraphrases)    What changed
                       ----------       ----------------------    --------------------
sync   P50               3911 ms             610 ms               ~6.4x faster
sync   P95               6542 ms            4675 ms               ~30% faster
stream TTFT P50          1426 ms             597 ms               ~2.4x faster
stream TTFT P95          1917 ms            1881 ms               basically unchanged
```

**Why an L2 hit costs ~600ms instead of a few.** When L2 hits, we don't have to call the LLM and we don't have to hit Qdrant. But we still have to embed the incoming question, because the embedding is the lookup key. That OpenAI embedding call by itself is around 500ms. Add the Redis vector search on top and you land near 600ms. So 600ms is the floor for an L2-only setup. Getting under that requires caching the embedding too, which is the embedding cache work mentioned below.

**Why P95 barely moved.** Two things are happening. The first paraphrase in each group has nothing to match against, because L2 is empty for that topic, so it pays full cold cost (around 4 seconds). And then one or two of the looser paraphrases per group land just below the similarity threshold and also pay the full cost. Out of 50 calls, you only need a few cold ones to land on top of the P95 bucket. L2 moves the median a lot. It does not move the tail at this hit rate. To move the tail you either need a higher hit rate (lower threshold, more seed coverage) or you need the cold path itself to be faster.

**Why sync and stream now look basically the same.** When L2 hits, there's no LLM generation to stream. The answer is already sitting in Redis. So "time to first token" and "time to full answer" collapse to the same number, which is whatever embed+lookup cost. Streaming only buys you something when there's actually a generation happening behind it.

**How I picked the threshold.** Started at 0.92, which is the safe default for cached RAG, the kind of default you pick when you'd rather miss than serve a wrong answer. At 0.92 the bench logged zero L2 hits. Pulled up the Langfuse traces and looked at the `cache-l2.output.top_similarity` field on each miss. The legitimate paraphrases were landing at 0.86, 0.89, 0.91, all just under the line. A cross-topic comparison (one base query checked against a different topic's stored entries) landed at 0.34, which is comfortably far away. So 0.85 catches the real paraphrase cluster and still has plenty of room above the cross-topic floor. On real traffic the same loop applies: log similarities for a week or two, look at the distribution, set the threshold one notch above whatever cluster you don't want to match into.

**To reproduce.**

```bash
# 1. Restart Redis and the API so the L2 index is fresh
docker compose restart redis
L2_THRESHOLD=0.85 uv run uvicorn app.main:app --port 8000

# 2. Run the paraphrase-mode bench
uv run python scripts/bench_latency.py --rows 0,1,2,3,4 --repeats 10 --paraphrase-mode
```

### Embedding cache (planned, not built)

**What it stores.** Just the embedding vector, keyed by a hash of the question text.

```
Redis entry:
  key:   cache:embed:<sha256("how do i get paid")>
  value: <1536 float32 bytes>      ← just the vector
```

**What it does.** Hashes the incoming question. If we've embedded this exact text before, return the stored vector instead of calling OpenAI. No answer is involved at all, only the vector.

**What it saves.** The OpenAI embedding API call (~200 to 1200ms depending on the day). This is currently the slowest single step in the cold path of this system, so it's a big lever.

**What it catches.** Exact text repeats. Same vocabulary as L1.

**Why it would be added.** Right now, every L1 miss pays the embedding cost even if we've embedded that exact text minutes ago. Embedding cache makes embed-on-repeat nearly free.

**The relationship with L1.** Once embedding cache exists, L1 becomes redundant. The reason L1 exists today is that without an embedding cache, every L1 miss pays the embedding tax. With both an embedding cache and L2, the flow for an exact repeat would be: embed lookup (~5ms) → L2 lookup (~5ms) → return. ~10ms total, basically as fast as L1's 5ms, and one fewer cache to maintain.

### Why the three are not interchangeable

They cache different things at different stages. The shortest way to say it:

- **L1** = "I've seen this exact question. Here's the answer I gave."
- **Embedding cache** = "I've embedded this exact text. Here's the vector."
- **L2** = "I've answered something very similar. Here's that answer."

L1 and the embedding cache both key on exact text, but they return different things (final answer vs intermediate vector). L2 and L1 both return answers, but L2 is keyed by meaning (the embedding) while L1 is keyed by exact text.

### The honest tradeoffs

- **L1 alone** is fast on exact repeats, useless on rephrasings.
- **L2 alone** catches rephrasings, but every request pays the embedding cost upfront (no way to skip it, you need the embedding to do the L2 lookup).
- **L1 + L2** (current state) covers both, but exact repeats still benefit from L1's speed because L2 alone would pay the embedding cost on them.
- **Embedding cache + L2** (future state) is the cleanest design: one place to cache vectors, one place to cache answers, L1 becomes redundant.

The migration plan is: build L1 first (cheap, immediate win), build L2 next (real production hit rate), then add embedding cache and retire L1.

---

## Production architecture and cost at scale

**Latency budget (target: TTFT P90 < 2s).** A RAG response spends time across embed, search, optional rerank, generate, and network. LLM generation dominates (60 to 80% of total), which is why streaming is the single biggest UX win. It does not cut total time, it cuts perceived latency to the first token. Measured numbers on this stack are in the streaming-backend section above (`stream TTFT P50 = 1737ms`, `sync total P50 = 4483ms`, **2.6x perceived win**).

**Cost drivers, ranked.** LLM calls dominate by an order of magnitude. Embedding refresh is small but easy to forget. Vector DB hosting is fixed cost. Rerankers (if hosted) become meaningful at volume.

**Back-of-envelope cost at 1M queries/month on this stack** (`gpt-4o-mini` + `text-embedding-3-small`, `top_k=5`):

| Component | Tokens per query | Tokens / month | Unit price | Monthly cost |
|---|---|---|---|---|
| Query embedding (`text-embedding-3-small`) | ~30 in | 30M | $0.020 / 1M | **$0.60** |
| LLM input (instruction + 5 chunks + query) | ~1,500 in | 1.5B | $0.15 / 1M | **$225** |
| LLM output (`gpt-4o-mini`) | ~250 out | 250M | $0.60 / 1M | **$150** |
| | | | **Total** | **~$375 / month** |

That works out to **~$0.000375 per query**. Pricing reference: https://developers.openai.com/api/docs/pricing

**What this number is sensitive to.**
- *Chunk size and `top_k`.* Doubling either roughly doubles LLM input cost. The dominant input line is retrieved context, not the user query or instruction.
- *Output length.* A chatty 500-token answer doubles the output line ($150 to $300). Capping `max_tokens` is a direct cost lever.
- *Model swap.* Moving the same workload to `gpt-4o` (full) is ~16x input and ~25x output, so the bill jumps from ~$375 to **~$7,300/month**. Stay on `mini` until eval shows it's the bottleneck.
- *Cache hit rate.* A 50% semantic-cache hit rate roughly halves the LLM lines. At 1M queries, that's ~$190 saved per month.
- *Reranker.* Not currently in the pipeline. If added (e.g. Cohere Rerank at ~$2/1M docs), 5M reranked docs/month adds ~$10. Negligible vs the LLM bill.

**What's not in the table.** Vector DB hosting (Qdrant self-hosted: VM cost only; managed: ~$100 to $600/month at this scale), Langfuse (free tier likely sufficient at 1M traces with 10% sampling), Redis (small instance ~$15/month). Add a flat ~$50 to $200 infra line on top.

**Takeaway.** At 1M queries/month, this stack is a **~$400 to $600/month all-in** workload. The single biggest cost lever is LLM input tokens, which is controlled by chunk count and chunk size, not model choice. Cache is the second lever once hit rate gets measured.

---

## Observability with Langfuse

**Why this is a different tool than the bench script.** The bench (`scripts/bench_latency.py`) is a *pre-deploy regression gate* with fixed inputs that runs on demand. Langfuse is for *online traffic*, where every real request emits a trace and dashboards aggregate over real users. Both coexist. They answer different questions.

**Setup.** Three env vars (already in `.env.example`):

```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com   # or self-hosted URL
```

The Langfuse skill (cloned into `.claude/skills/langfuse/`) defines best practices, and Claude Code can pick it up via the slash command on next session start.

**Integration approach: drop-in OpenAI replacement.** The skill's first recommendation is "prefer framework integrations over manual instrumentation." For OpenAI-Python that's a one-line import swap:

```diff
- from openai import AsyncOpenAI
+ from langfuse.openai import AsyncOpenAI
```

That single change auto-captures **model name, full prompt/response, token counts (in/out), cost in USD, latency, errors**. Every "baseline requirement" the skill lists. Streaming is supported transparently.

**Trace shape per request:**

```
chat-sync | chat-stream                  ← parent span (input = user query, output = answer)
    ├─ retrieve                          ← child span (output = chunk ids + scores)
    └─ generation (OpenAI)               ← auto-instrumented; tokens, cost, model
```

**Trace attributes** (set via `propagate_attributes` so they propagate to all child spans):

| Attribute | Source | What it powers |
|---|---|---|
| `user_id` | `ChatRequest.user_id` (defaults to `"anonymous"`) | Per-user filtering, cost attribution, Users view |
| `session_id` | `ChatRequest.session_id` | Sessions view, groups multi-turn conversations |
| `tags` | `["chat", "sync"]` or `["chat", "stream"]` | Filter dashboards by feature/endpoint |

**Two views you'll use:**

1. **Trace inspector.** Pick a slow request, drill in. The `retrieve` span shows exactly how much of TTFT was retrieval vs. LLM prefill. This is what flagged retrieval as the bottleneck in the streaming-backend bench. Replaces print debugging.
2. **Dashboards.** P50/P95 latency, cost per user, error rate. Sliced by tag (`stream` vs `sync`) or by `user_id`.

**Lifespan flush.** `get_client().flush()` is called on FastAPI shutdown (`app/main.py`). Without it, queued traces in short-lived processes get dropped. The skill flags this as a common mistake.

**Things deliberately NOT instrumented** (yet):
- **User feedback scores.** Needs a `/feedback` endpoint that posts a thumbs up/down attached to the trace ID. Trace ID would need to be returned to the client first (currently isn't).
- **Sampling.** At low traffic, sample 100%. Once the project sees real volume, set `LANGFUSE_SAMPLE_RATE=0.1` to keep the cost/storage bill reasonable while still getting representative percentiles.
- **PII masking.** Knowledge base is non-sensitive, so query/answer logging is safe. If this ever ingests user-provided documents, revisit.

---

## What's next

In rough priority order, what I'd build next if this kept being my main project:

1. **Fix the retrieval tail.** Per-request retrieval ranges from 440ms to 4132ms for the same query, dominated by the OpenAI embedding call. An embedding cache is the cleanest fix and would also let me retire L1.
2. **Wire hybrid retrieval (BM25 + dense + RRF).** The Haystack pieces are there. BM25 catches the keyword-heavy edge cases dense retrieval misses, and RRF fuses them without tuning weights.
3. **Add concurrent load to the bench.** TTFT under load is where streaming's perceptual win matters most. Right now I only measure single-request latency.
4. **Return trace IDs to the client and add a `/feedback` endpoint.** Langfuse needs feedback scores to close the eval loop on real users.
5. **Wire CRAG, the router, and guards into the live request path.** The scaffolding exists; they need to be plugged in and benched before being claimed as features.

---

## References

**Chunking and embeddings**
- [RAG Chunking Strategies: The 2026 Benchmark Guide](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/)
- Vectara / FloTorch 2026 benchmark (recursive 512t: 69%, semantic: 54%)
- Firecrawl: chunking actively hurts retrieval on short, focused documents
- [OpenAI `text-embedding-3-small` pricing](https://developers.openai.com/api/docs/models/text-embedding-3-small)

**Streaming and latency**
- [BentoML, LLM inference metrics](https://bentoml.com/llm/inference-optimization/llm-inference-metrics)
- [NVIDIA NIM benchmarking metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html)
- [Artificial Analysis, performance benchmarking methodology](https://artificialanalysis.ai/methodology/performance-benchmarking)
- [Cloud Portable Tech, RAG benchmark SSE contract](https://www.cloudportabletech.com)

**Cost and observability**
- [OpenAI API pricing](https://developers.openai.com/api/docs/pricing)
- [Langfuse, OpenAI Python integration](https://langfuse.com/integrations/model-providers/openai-py)
- [Langfuse skill repo](https://github.com/langfuse/skills) (installed at `.claude/skills/langfuse/`)

---

Built by Mohsin Sheikhani. If anything here is interesting and you'd like to talk, [linkedin.com/in/mohsin-sheikhani](https://www.linkedin.com/in/mohsin-sheikhani/).
