"""Test hybrid chunking on one short doc and one long doc."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.indexing_pipeline import create_indexing_pipeline

pipeline = create_indexing_pipeline(document_store=None)

kb = Path(__file__).parent.parent / "knowledge_base"
all_files = list(kb.rglob("*.md"))

# Pick one short and one long file based on word count
short_file = min(all_files, key=lambda f: len(f.read_text().split()))
long_file  = max(all_files, key=lambda f: len(f.read_text().split()))

for label, path in [("SHORT", short_file), ("LONG", long_file)]:
    result = pipeline.run({"converter": {"sources": [path]}})
    chunks = result["embedder"]["documents"]

    print(f"\n{'='*60}")
    print(f"{label}: {path.name}  ({len(path.read_text().split())} words → {len(chunks)} chunk(s))")
    for i, doc in enumerate(chunks, 1):
        words = len(doc.content.split())
        preview = doc.content[:120].replace("\n", " ")
        emb = f"embedding: {len(doc.embedding)}d" if doc.embedding else "no embedding"
        print(f"  [{i}] {words} words | {emb} | meta: {doc.meta}")
        print(f"       {preview}...")
