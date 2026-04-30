"""Script to index markdown documents into Qdrant."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.document_store import get_document_store
from pipeline.ingest import IngestionPipeline

def main():
    # Initialize
    document_store = get_document_store()
    pipeline = IngestionPipeline(document_store)
    
    # Get first markdown file to test
    docs_dir = Path(__file__).parent.parent / "knowledge-base"
    md_files = list(docs_dir.rglob("*.md"))
    
    if md_files:
        print(f"Testing with: {md_files[0]}")
        result = pipeline.ingest(str(md_files[0]))
        
        # Print converted documents (with markdown preserved)
        print("\n=== CONVERTED DOCUMENTS ===")
        for doc in result.get("converter", {}).get("documents", []):
            print(f"\nFull Content:\n{doc.content}")
            print(f"\nMetadata: {doc.meta}")
    else:
        print("No markdown files found")

if __name__ == "__main__":
    main()
