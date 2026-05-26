"""FastAPI entry point with CORS, lifespan, and request/response schemas."""
from dotenv import load_dotenv
load_dotenv()  # must run before any module that reads OPENAI_API_KEY at import

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import redis.asyncio as redis

from langfuse import get_client

from app.document_store import get_document_store
from app.config import settings
from routes.chat import router as chat_router
from services.semantic_cache import get_cache

# Global state
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize Haystack document store and Redis
    app_state["document_store"] = get_document_store()
    app_state["redis"] = await redis.from_url(f"redis://{settings.redis_host}:{settings.redis_port}")
    await get_cache().ensure_index()

    yield

    # Shutdown: flush Langfuse traces, then cleanup resources.
    # Without flush() short-lived events queued in the SDK can be dropped.
    get_client().flush()
    await app_state["redis"].close()

app = FastAPI(title="Production RAG System", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)


@app.get("/")
async def root():
    return {"status": "ok", "service": "production-rag-system", "framework": "haystack"}
