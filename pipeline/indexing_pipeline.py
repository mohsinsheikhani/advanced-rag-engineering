"""Haystack indexing pipeline: extract → chunk → embed → index."""
from haystack import Pipeline
from haystack.components.converters import TextFileToDocument
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from pipeline.hybrid_chunker import HybridMarkdownChunker


def create_indexing_pipeline(document_store: QdrantDocumentStore) -> Pipeline:
    pipeline = Pipeline()
    pipeline.add_component("converter", TextFileToDocument())
    pipeline.add_component("chunker", HybridMarkdownChunker())
    pipeline.add_component("embedder", OpenAIDocumentEmbedder(model="text-embedding-3-small"))
    pipeline.add_component("writer", DocumentWriter(document_store=document_store))

    pipeline.connect("converter", "chunker")
    pipeline.connect("chunker", "embedder")
    pipeline.connect("embedder", "writer")

    return pipeline
