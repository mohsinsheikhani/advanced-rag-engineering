"""Index all markdown documents into Qdrant."""
import sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from app.document_store import get_document_store
from pipeline.ingest import IngestionPipeline

document_store = get_document_store()
pipeline = IngestionPipeline(document_store)

docs_dir = Path(__file__).parent.parent / "knowledge_base"
md_files = list(docs_dir.rglob("*.md"))
print(f"Found {len(md_files)} files")

for f in md_files:
    result = pipeline.ingest(str(f))
    written = result.get("writer", {}).get("documents_written", "?")
    print(f"  {f.name}: {written} chunks written")

print("\nDone.")
