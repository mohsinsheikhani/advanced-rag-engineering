"""Readiness + dependency checks."""
from fastapi import APIRouter

router = APIRouter(tags=["health"])

@router.get("/health")
async def health():
    # TODO: Check vector DB, Redis, LLM availability
    return {"status": "healthy", "dependencies": {}}
