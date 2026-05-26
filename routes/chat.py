"""Session 2 — minimal chat endpoints for streaming benchmark.
Session 4 — instrumented with Langfuse via the OpenAI drop-in integration.
Session 5 — two-tier semantic cache (L1 exact, L2 vector).

Two endpoints:
- POST /chat/sync   : non-streaming baseline. Returns full answer + server timings.
- POST /chat/stream : SSE stream. Emits meta / ttft / token / done events.

Tracing shape (per request):
    chat-sync | chat-stream            ← parent span (input = user query)
        ├─ cache-l1                    ← exact-match Redis GET
        ├─ embed                       ← OpenAI embedding (only if L1 missed)
        ├─ cache-l2                    ← KNN over query embeddings
        ├─ retrieve                    ← Qdrant search (only if L2 missed)
        └─ generation                  ← auto-instrumented by langfuse.openai
"""
from __future__ import annotations

import json
import time
from typing import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langfuse import get_client, propagate_attributes
from langfuse.openai import AsyncOpenAI
from pydantic import BaseModel, Field

from app.config import settings
from app.document_store import get_document_store
from haystack.components.embedders import OpenAITextEmbedder
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from services.semantic_cache import get_cache


router = APIRouter(prefix="/chat", tags=["chat"])

LLM_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
TOP_K = 5


def _build_prompt(query: str, chunks: list[str]) -> str:
    context = "\n---\n".join(chunks)
    return (
        "You are a helpful assistant answering questions about MIA "
        "(Make Income Anywhere).\n\n"
        "Use the following context to answer the user's question. If the "
        "context doesn't contain enough information, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\nAnswer:"
    )


# Module-level singletons (built lazily, after env is loaded).
_embedder: OpenAITextEmbedder | None = None
_retriever: QdrantEmbeddingRetriever | None = None
_openai_client: AsyncOpenAI | None = None


def _get_embedder() -> OpenAITextEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = OpenAITextEmbedder(model=EMBED_MODEL)
    return _embedder


def _get_retriever() -> QdrantEmbeddingRetriever:
    global _retriever
    if _retriever is None:
        _retriever = QdrantEmbeddingRetriever(
            document_store=get_document_store(), top_k=TOP_K
        )
    return _retriever


def _get_openai() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI()
    return _openai_client


# ---------------------------------------------------------------------------
# Cache + embed + retrieve helpers — each wraps a Langfuse span
# ---------------------------------------------------------------------------

async def _cache_l1(query: str) -> dict | None:
    langfuse = get_client()
    with langfuse.start_as_current_observation(name="cache-l1", as_type="span") as span:
        span.update(input={"query": query})
        hit = await get_cache().lookup_exact(query)
        span.update(output={"hit": hit is not None})
        return hit


def _embed(query: str) -> list[float]:
    langfuse = get_client()
    with langfuse.start_as_current_observation(name="embed", as_type="span") as span:
        span.update(input={"query": query, "model": EMBED_MODEL})
        embedding = _get_embedder().run(text=query)["embedding"]
        span.update(output={"dim": len(embedding)})
        return embedding


async def _cache_l2(embedding: list[float]) -> tuple[dict, float] | None:
    langfuse = get_client()
    with langfuse.start_as_current_observation(name="cache-l2", as_type="span") as span:
        payload, similarity = await get_cache().lookup_semantic(embedding)
        if payload is None:
            span.update(output={
                "hit": False,
                "top_similarity": round(similarity, 4) if similarity is not None else None,
                "threshold": settings.l2_threshold,
            })
            return None
        span.update(output={
            "hit": True,
            "similarity": round(similarity, 4),
            "matched_query": payload["matched_query"],
        })
        return payload, similarity


def _retrieve_with_embedding(embedding: list[float]) -> tuple[list[str], list[dict]]:
    langfuse = get_client()
    with langfuse.start_as_current_observation(name="retrieve", as_type="span") as span:
        span.update(input={"top_k": TOP_K})
        docs = _get_retriever().run(query_embedding=embedding)["documents"]
        chunks = [d.content for d in docs]
        sources = [{"id": d.id, "score": d.score, "meta": d.meta} for d in docs]
        span.update(output={
            "n_chunks": len(chunks),
            "ids": [s["id"] for s in sources],
            "scores": [s["score"] for s in sources],
        })
        return chunks, sources


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(TOP_K, ge=1, le=20)
    user_id: str = "anonymous"
    session_id: str | None = None


class ChatSyncResponse(BaseModel):
    answer: str
    sources: list[dict]
    retrieval_ms: float
    total_ms: float


# ---------------------------------------------------------------------------
# /chat/sync — baseline, non-streaming
# ---------------------------------------------------------------------------

@router.post("/sync", response_model=ChatSyncResponse)
async def chat_sync(req: ChatRequest) -> ChatSyncResponse:
    langfuse = get_client()

    with langfuse.start_as_current_observation(name="chat-sync", as_type="span") as span:
        span.update(input={"query": req.query})

        with propagate_attributes(
            user_id=req.user_id,
            session_id=req.session_id,
            tags=["chat", "sync"],
        ):
            t0 = time.perf_counter()

            # L1
            l1 = await _cache_l1(req.query)
            if l1 is not None:
                t_done = time.perf_counter()
                span.update(output={"answer": l1["answer"], "cache": "l1"})
                return ChatSyncResponse(
                    answer=l1["answer"],
                    sources=l1["sources"],
                    retrieval_ms=0.0,
                    total_ms=(t_done - t0) * 1000,
                )

            # Embed once. Reused for L2 lookup and (if L2 miss) Qdrant retrieval.
            embedding = _embed(req.query)

            # L2
            l2 = await _cache_l2(embedding)
            if l2 is not None:
                payload, _sim = l2
                t_done = time.perf_counter()
                span.update(output={"answer": payload["answer"], "cache": "l2"})
                return ChatSyncResponse(
                    answer=payload["answer"],
                    sources=payload["sources"],
                    retrieval_ms=0.0,
                    total_ms=(t_done - t0) * 1000,
                )

            chunks, sources = _retrieve_with_embedding(embedding)
            t_retrieved = time.perf_counter()

            prompt = _build_prompt(req.query, chunks)
            completion = await _get_openai().chat.completions.create(
                model=LLM_MODEL,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
                name="generation",
            )
            answer = completion.choices[0].message.content or ""
            t_done = time.perf_counter()

            await get_cache().store_exact(req.query, answer, sources)
            await get_cache().store_semantic(req.query, embedding, answer, sources)

        span.update(output={"answer": answer})

    return ChatSyncResponse(
        answer=answer,
        sources=sources,
        retrieval_ms=(t_retrieved - t0) * 1000,
        total_ms=(t_done - t0) * 1000,
    )


# ---------------------------------------------------------------------------
# /chat/stream — SSE: meta / ttft / token / done
# ---------------------------------------------------------------------------

def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _cached_done_payload(ttft_ms: float, total_ms: float, tier: str) -> dict:
    return {
        "ttft_ms": ttft_ms,
        "total_ms": total_ms,
        "retrieval_ms": 0.0,
        "tokens": 1,
        "tps": 0.0,
        "cache_hit": tier,
    }


@router.post("/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    async def event_generator() -> AsyncIterator[str]:
        langfuse = get_client()

        with langfuse.start_as_current_observation(name="chat-stream", as_type="span") as span:
            span.update(input={"query": req.query})

            with propagate_attributes(
                user_id=req.user_id,
                session_id=req.session_id,
                tags=["chat", "stream"],
            ):
                t0 = time.perf_counter()

                # L1
                l1 = await _cache_l1(req.query)
                if l1 is not None:
                    ttft_ms = (time.perf_counter() - t0) * 1000
                    yield _sse("meta", {"retrieval_ms": 0.0, "sources": l1["sources"], "cache_hit": "l1"})
                    yield _sse("ttft", {"ttft_ms": ttft_ms})
                    yield _sse("token", {"text": l1["answer"]})
                    total_ms = (time.perf_counter() - t0) * 1000
                    yield _sse("done", _cached_done_payload(ttft_ms, total_ms, "l1"))
                    span.update(output={"answer": l1["answer"]}, metadata={"cache": "l1", "ttft_ms": ttft_ms})
                    return

                embedding = _embed(req.query)

                # L2
                l2 = await _cache_l2(embedding)
                if l2 is not None:
                    payload, _sim = l2
                    ttft_ms = (time.perf_counter() - t0) * 1000
                    yield _sse("meta", {"retrieval_ms": 0.0, "sources": payload["sources"], "cache_hit": "l2"})
                    yield _sse("ttft", {"ttft_ms": ttft_ms})
                    yield _sse("token", {"text": payload["answer"]})
                    total_ms = (time.perf_counter() - t0) * 1000
                    yield _sse("done", _cached_done_payload(ttft_ms, total_ms, "l2"))
                    span.update(output={"answer": payload["answer"]}, metadata={"cache": "l2", "ttft_ms": ttft_ms})
                    return

                chunks, sources = _retrieve_with_embedding(embedding)
                retrieval_ms = (time.perf_counter() - t0) * 1000
                yield _sse("meta", {"retrieval_ms": retrieval_ms, "sources": sources})

                prompt = _build_prompt(req.query, chunks)
                stream = await _get_openai().chat.completions.create(
                    model=LLM_MODEL,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                    name="generation",
                )

                ttft_ms: float | None = None
                token_count = 0
                pieces: list[str] = []
                async for chunk in stream:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if not delta:
                        continue
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t0) * 1000
                        yield _sse("ttft", {"ttft_ms": ttft_ms})
                    token_count += 1
                    pieces.append(delta)
                    yield _sse("token", {"text": delta})

                total_ms = (time.perf_counter() - t0) * 1000
                tps = (token_count / (total_ms / 1000)) if total_ms > 0 else 0.0
                answer = "".join(pieces)
                await get_cache().store_exact(req.query, answer, sources)
                await get_cache().store_semantic(req.query, embedding, answer, sources)
                yield _sse(
                    "done",
                    {
                        "ttft_ms": ttft_ms,
                        "total_ms": total_ms,
                        "retrieval_ms": retrieval_ms,
                        "tokens": token_count,
                        "tps": tps,
                    },
                )

            span.update(
                output={"answer": "".join(pieces)},
                metadata={"ttft_ms": ttft_ms, "total_ms": total_ms, "tokens": token_count},
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
