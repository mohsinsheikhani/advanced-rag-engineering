"""Session 2 — minimal chat endpoints for streaming benchmark.
Session 4 — instrumented with Langfuse via the OpenAI drop-in integration.

Two endpoints:
- POST /chat/sync   : non-streaming baseline. Returns full answer + server timings.
- POST /chat/stream : SSE stream. Emits meta / ttft / token / done events.

Tracing shape (per request):
    chat-sync | chat-stream            ← parent span (input = user query)
        ├─ retrieve                    ← child span (output = chunk count + ids)
        └─ generation                  ← auto-instrumented by langfuse.openai
"""
from __future__ import annotations

import json
import time
from typing import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langfuse import get_client, propagate_attributes
from langfuse.openai import AsyncOpenAI  # drop-in: auto-traces model, tokens, latency
from pydantic import BaseModel, Field

from app.document_store import get_document_store
from haystack import Pipeline
from haystack.components.embedders import OpenAITextEmbedder
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever


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
_retrieval_pipeline: Pipeline | None = None
_openai_client: AsyncOpenAI | None = None


def _get_retrieval() -> Pipeline:
    global _retrieval_pipeline
    if _retrieval_pipeline is None:
        store = get_document_store()
        p = Pipeline()
        p.add_component("text_embedder", OpenAITextEmbedder(model=EMBED_MODEL))
        p.add_component("retriever", QdrantEmbeddingRetriever(document_store=store, top_k=TOP_K))
        p.connect("text_embedder.embedding", "retriever.query_embedding")
        _retrieval_pipeline = p
    return _retrieval_pipeline


def _get_openai() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI()
    return _openai_client


def _retrieve(query: str) -> tuple[list[str], list[dict]]:
    """Run Haystack retrieval inside its own Langfuse span.

    Returns (chunk_texts, source_metadata).
    """
    langfuse = get_client()
    with langfuse.start_as_current_observation(name="retrieve", as_type="span") as span:
        span.update(input={"query": query, "top_k": TOP_K})
        result = _get_retrieval().run({"text_embedder": {"text": query}})
        docs = result["retriever"]["documents"]
        chunks = [d.content for d in docs]
        sources = [{"id": d.id, "score": d.score, "meta": d.meta} for d in docs]
        # Output kept compact — full chunk text is on the trace via input/output
        # of the next step (generation), no need to duplicate it here.
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
        # Explicit input — avoids leaking the full request body (incl. defaults).
        span.update(input={"query": req.query})

        with propagate_attributes(
            user_id=req.user_id,
            session_id=req.session_id,
            tags=["chat", "sync"],
        ):
            t0 = time.perf_counter()
            chunks, sources = _retrieve(req.query)
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

                chunks, sources = _retrieve(req.query)
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
