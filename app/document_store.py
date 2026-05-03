"""Qdrant document store initialization."""
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from app.config import settings

def get_document_store() -> QdrantDocumentStore:
    """Initialize Qdrant document store."""
    return QdrantDocumentStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        index="documents",
        embedding_dim=1536,  # text-embedding-3-small dimension
        recreate_index=False  # collection already exists with 1536-dim vectors
    )
