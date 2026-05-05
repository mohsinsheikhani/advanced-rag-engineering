"""Session 3 — Config B2 (bad embedder).

Swap text-embedding-3-small (1536-D, OpenAI) → multi-qa-MiniLM-L6-cos-v1
(384-D, local) — the model the README documents as having a vocabulary
mismatch on this corpus.

Prerequisite: run `uv run python scripts/index_documents_384.py` once to
populate the 'documents_384' Qdrant collection.

Prediction:
- Contextual Recall:    DROP — relevant chunks fall out of top-5.
- Contextual Precision: DROP — irrelevant chunks crowd top positions.
- Faithfulness:         may STAY HIGH (the dangerous case).
- Answer Relevancy:     may DROP if context is off-topic.
"""
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from app.config import settings
from eval.common import load_row, run_and_evaluate

ROW_INDEX = 1
COLLECTION = "documents_384"
EMBED_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"


def small_embedder():
    return SentenceTransformersTextEmbedder(model=EMBED_MODEL)


store_384 = QdrantDocumentStore(
    host=settings.qdrant_host,
    port=settings.qdrant_port,
    index=COLLECTION,
    embedding_dim=384,
    recreate_index=False,
)

query, expected = load_row(ROW_INDEX)
run_and_evaluate(
    label=f"CONFIG B2 — bad embedder ({EMBED_MODEL}) — row {ROW_INDEX}",
    query=query,
    expected=expected,
    store=store_384,
    embedder_factory=small_embedder,
)
