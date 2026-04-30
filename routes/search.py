"""Retrieval debugger endpoint."""
from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["search"])

@router.get("/search")
async def search(query: str, top_k: int = 10):
    # TODO: Direct vector search for debugging
    return {"results": []}
