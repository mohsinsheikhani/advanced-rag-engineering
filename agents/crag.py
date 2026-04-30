"""Self-correcting RAG loop with citations."""

class CRAGAgent:
    def run(self, query: str):
        """
        1. Retrieve
        2. Grade relevance
        3. Generate or decompose
        4. Re-retrieve if needed
        5. Merge with citations
        """
        pass
