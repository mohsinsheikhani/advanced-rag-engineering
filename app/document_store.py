"""Qdrant document store initialization."""
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from app.config import settings

def get_document_store() -> QdrantDocumentStore:
    """Initialize Qdrant document store."""
    return QdrantDocumentStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        index="documents",
        embedding_dim=384,  # all-MiniLM-L6-v2 dimension
        recreate_index=False
    )
