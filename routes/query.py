"""API layer - SSE streaming /api/query endpoint."""
from fastapi import APIRouter
from app.models import QueryRequest, QueryResponse

router = APIRouter(prefix="/api", tags=["query"])

@router.post("/query")
async def query(request: QueryRequest) -> QueryResponse:
    # TODO: Integrate with RAG pipeline
    return QueryResponse(
        answer="Response placeholder",
        sources=[],
        trace_id="trace-123"
    )
