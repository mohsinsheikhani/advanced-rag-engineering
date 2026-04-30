"""Haystack hybrid retrieval: BM25 + Dense + RRF fusion."""
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.components.joiners import DocumentJoiner
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

def create_retrieval_pipeline(document_store: QdrantDocumentStore) -> Pipeline:
    """Build hybrid retrieval pipeline with RRF fusion."""
    pipeline = Pipeline()
    
    # Add components
    pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"))
    pipeline.add_component("dense_retriever", QdrantEmbeddingRetriever(document_store=document_store))
    pipeline.add_component("bm25_retriever", InMemoryBM25Retriever(document_store=document_store))
    pipeline.add_component("joiner", DocumentJoiner(join_mode="reciprocal_rank_fusion"))
    
    # Connect components
    pipeline.connect("text_embedder.embedding", "dense_retriever.query_embedding")
    pipeline.connect("dense_retriever", "joiner")
    pipeline.connect("bm25_retriever", "joiner")
    
    return pipeline
