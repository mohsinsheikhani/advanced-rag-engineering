"""Test retrieval with query expansion."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from app.document_store import get_document_store
from retrieval.retrieval_pipeline import create_retrieval_pipeline
from services.query_decomposer import QueryDecomposer

document_store = get_document_store()
pipeline = create_retrieval_pipeline(document_store)
decomposer = QueryDecomposer()

queries = [
    "Who is MIA for?",
    "How do I make money with MIA?",
]

for query in queries:
    # expanded = decomposer.decompose(query)
    # print(f"\nQ: {query}")
    # print(f"   Expanded: {expanded[1:]}")
    print(f"\nQ: {query}")

    # Retrieve for each rephrasing, merge by doc id, keep highest score
    seen, merged = {}, []
    # for q in expanded:
    result = pipeline.run({"text_embedder": {"text": query}})
    for doc in result["retriever"]["documents"]:
        if doc.id not in seen or doc.score > seen[doc.id].score:
            seen[doc.id] = doc

    top5 = sorted(seen.values(), key=lambda d: d.score, reverse=True)[:5]
    for i, doc in enumerate(top5, 1):
        preview = doc.content[:100].replace("\n", " ")
        print(f"  [{i}] score={round(doc.score,3)} | {doc.meta.get('file_path','')} | {preview}...")
