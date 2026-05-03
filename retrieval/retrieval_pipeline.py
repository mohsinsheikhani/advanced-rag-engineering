"""Haystack dense retrieval pipeline."""
from haystack import Pipeline
from haystack.components.embedders import OpenAITextEmbedder
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore


def create_retrieval_pipeline(document_store: QdrantDocumentStore) -> Pipeline:
    pipeline = Pipeline()
    pipeline.add_component("text_embedder", OpenAITextEmbedder(model="text-embedding-3-small"))
    pipeline.add_component("retriever", QdrantEmbeddingRetriever(document_store=document_store, top_k=5))

    pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

    return pipeline
