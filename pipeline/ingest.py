"""Haystack-based document ingestion."""
from pathlib import Path
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from pipeline.indexing_pipeline import create_indexing_pipeline

class IngestionPipeline:
    def __init__(self, document_store: QdrantDocumentStore):
        self.pipeline = create_indexing_pipeline(document_store)
    
    def ingest(self, file_path: str):
        """Ingest markdown document using Haystack pipeline."""
        result = self.pipeline.run({"converter": {"sources": [Path(file_path)]}})
        return result
    
    def ingest_directory(self, directory: str):
        """Ingest all markdown files from a directory."""
        md_files = list(Path(directory).rglob("*.md"))
        results = []
        for file in md_files:
            result = self.ingest(str(file))
            results.append(result)
        return results
