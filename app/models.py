"""Pydantic request/response schemas."""
from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    stream: bool = False

class Source(BaseModel):
    content: str
    metadata: dict
    score: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    trace_id: Optional[str] = None
