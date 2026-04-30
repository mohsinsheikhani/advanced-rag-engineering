"""Haystack document writer wrapper."""
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

class Indexer:
    def __init__(self, document_store: QdrantDocumentStore):
        self.writer = DocumentWriter(document_store=document_store)
    
    def index(self, documents: list):
        """Index documents using Haystack writer."""
        return self.writer.run(documents=documents)
