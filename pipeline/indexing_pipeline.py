"""Haystack indexing pipeline: extract → preprocess → chunk → embed → index."""
from haystack import Pipeline
from haystack.components.converters import TextFileToDocument
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from pipeline.hybrid_chunker import HybridMarkdownChunker


def create_indexing_pipeline(document_store: QdrantDocumentStore) -> Pipeline:
    pipeline = Pipeline()
    pipeline.add_component("converter", TextFileToDocument())
    pipeline.add_component("cleaner", DocumentCleaner())
    pipeline.add_component("chunker", HybridMarkdownChunker())
    pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"))
    pipeline.add_component("writer", DocumentWriter(document_store=document_store))

    pipeline.connect("converter", "cleaner")
    pipeline.connect("cleaner", "chunker")
    pipeline.connect("chunker", "embedder")
    pipeline.connect("embedder", "writer")

    return pipeline
