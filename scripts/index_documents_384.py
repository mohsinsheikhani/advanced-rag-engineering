"""
Re-index the knowledge_base into a separate 384-D Qdrant collection using
`sentence-transformers/multi-qa-MiniLM-L6-cos-v1` — for failure-mode eval
(Config B2: bad embedder).

The production collection ('documents', 1536-D) is left untouched.
Run once before eval_bad_embedder_row1.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from haystack import Pipeline
from haystack.components.converters import TextFileToDocument
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from app.config import settings
from pipeline.hybrid_chunker import HybridMarkdownChunker


COLLECTION = "documents_384"
MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
DIM = 384


store = QdrantDocumentStore(
    host=settings.qdrant_host,
    port=settings.qdrant_port,
    index=COLLECTION,
    embedding_dim=DIM,
    recreate_index=True,
)

pipeline = Pipeline()
pipeline.add_component("converter", TextFileToDocument())
pipeline.add_component("chunker", HybridMarkdownChunker())
pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder(model=MODEL))
pipeline.add_component("writer", DocumentWriter(document_store=store))
pipeline.connect("converter", "chunker")
pipeline.connect("chunker", "embedder")
pipeline.connect("embedder", "writer")

docs_dir = Path(__file__).parent.parent / "knowledge_base"
md_files = [str(p) for p in docs_dir.rglob("*.md")]
print(f"Indexing {len(md_files)} files into '{COLLECTION}' ({DIM}-D, {MODEL})")

result = pipeline.run({"converter": {"sources": md_files}})
written = result.get("writer", {}).get("documents_written", "?")
print(f"Done. {written} chunks written.")
