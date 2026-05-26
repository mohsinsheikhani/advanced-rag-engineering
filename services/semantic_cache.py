"""Two-tier cache.

L1 — exact match. Key = sha256(normalized_query). Sub-ms GET.
L2 — semantic match. Key = HNSW vector index over query embeddings in Redis Stack.

Flow per request:
    L1 lookup -> if miss: embed -> L2 lookup -> if miss: full pipeline -> store both.
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from typing import Any

import numpy as np
import redis.asyncio as redis
from redis.commands.search.field import VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.exceptions import ResponseError

from app.config import settings

_L1_PREFIX = "cache:l1:"
_L2_PREFIX = "cache:l2:"
_L2_INDEX = "cache:l2:idx"
_DEFAULT_TTL = 3600
_EMBED_DIM = 1536  # text-embedding-3-small


def _normalize(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower())


def _l1_key(query: str) -> str:
    h = hashlib.sha256(_normalize(query).encode("utf-8")).hexdigest()
    return f"{_L1_PREFIX}{h}"


def _vec_bytes(embedding: list[float]) -> bytes:
    return np.asarray(embedding, dtype=np.float32).tobytes()


class SemanticCache:
    def __init__(self, client: redis.Redis):
        self.redis = client

    # ---- L1 ----------------------------------------------------------------

    async def lookup_exact(self, query: str) -> dict[str, Any] | None:
        if not settings.cache_enabled:
            return None
        raw = await self.redis.get(_l1_key(query))
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)

    async def store_exact(
        self,
        query: str,
        answer: str,
        sources: list[dict],
        ttl: int = _DEFAULT_TTL,
    ) -> None:
        if not settings.cache_enabled:
            return
        payload = json.dumps({"answer": answer, "sources": sources})
        await self.redis.set(_l1_key(query), payload, ex=ttl)

    # ---- L2 ----------------------------------------------------------------

    async def ensure_index(self) -> None:
        """Create the HNSW vector index once on startup. Idempotent."""
        try:
            await self.redis.ft(_L2_INDEX).info()
            return  # already exists
        except ResponseError:
            pass

        schema = (
            VectorField(
                "embedding",
                "HNSW",
                {
                    "TYPE": "FLOAT32",
                    "DIM": _EMBED_DIM,
                    "DISTANCE_METRIC": "COSINE",
                },
            ),
        )
        definition = IndexDefinition(prefix=[_L2_PREFIX], index_type=IndexType.HASH)
        await self.redis.ft(_L2_INDEX).create_index(
            fields=schema, definition=definition
        )

    async def lookup_semantic(
        self, embedding: list[float]
    ) -> tuple[dict[str, Any] | None, float | None]:
        """KNN-1 search. Returns (payload_or_None, top_similarity_or_None).

        Payload is None when the top match is below threshold (or no matches exist),
        but the similarity is still returned so the caller can log how close it was.
        """
        if not settings.cache_enabled or not settings.l2_enabled:
            return (None, None)

        q = (
            Query("*=>[KNN 1 @embedding $vec AS dist]")
            .return_fields("dist", "answer", "sources", "query")
            .sort_by("dist")
            .dialect(2)
        )
        try:
            res = await self.redis.ft(_L2_INDEX).search(
                q, query_params={"vec": _vec_bytes(embedding)}
            )
        except ResponseError:
            # Index missing (first run before ensure_index)
            return (None, None)

        if not res.docs:
            return (None, None)

        top = res.docs[0]

        def _s(v: Any) -> str:
            return v.decode("utf-8") if isinstance(v, bytes) else v

        # Redis returns cosine *distance* in [0, 2]; similarity = 1 - distance.
        similarity = 1.0 - float(top.dist)
        if similarity < settings.l2_threshold:
            return (None, similarity)

        return (
            {
                "answer": _s(top.answer),
                "sources": json.loads(_s(top.sources)),
                "matched_query": _s(top.query),
            },
            similarity,
        )

    async def store_semantic(
        self,
        query: str,
        embedding: list[float],
        answer: str,
        sources: list[dict],
        ttl: int = _DEFAULT_TTL,
    ) -> None:
        if not settings.cache_enabled or not settings.l2_enabled:
            return
        key = f"{_L2_PREFIX}{uuid.uuid4().hex}"
        await self.redis.hset(
            key,
            mapping={
                "embedding": _vec_bytes(embedding),
                "answer": answer,
                "sources": json.dumps(sources),
                "query": query,
            },
        )
        await self.redis.expire(key, ttl)


_client: redis.Redis | None = None
_cache: SemanticCache | None = None


def get_cache() -> SemanticCache:
    global _client, _cache
    if _cache is None:
        _client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            decode_responses=False,  # vector bytes must stay binary
        )
        _cache = SemanticCache(_client)
    return _cache