"""Orchestrates full query flow with Haystack pipelines."""
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from retrieval.retrieval_pipeline import create_retrieval_pipeline
from retrieval.reranker import Reranker
from security.input_guard import InputGuard
from security.output_guard import OutputGuard
from services.semantic_cache import SemanticCache

class RAGPipeline:
    def __init__(self, document_store: QdrantDocumentStore, redis_client):
        self.input_guard = InputGuard()
        self.output_guard = OutputGuard()
        self.cache = SemanticCache(redis_client)
        self.reranker = Reranker()
        
        # Build Haystack RAG pipeline
        self.pipeline = self._build_pipeline(document_store)
    
    def _build_pipeline(self, document_store: QdrantDocumentStore) -> Pipeline:
        """Build complete RAG pipeline."""
        retrieval_pipeline = create_retrieval_pipeline(document_store)
        
        pipeline = Pipeline()
        pipeline.add_component("retrieval", retrieval_pipeline)
        pipeline.add_component("prompt_builder", PromptBuilder(template="Context: {{documents}}\n\nQuestion: {{query}}\n\nAnswer:"))
        pipeline.add_component("llm", OpenAIGenerator(model="gpt-4"))
        
        pipeline.connect("retrieval.joiner", "prompt_builder.documents")
        pipeline.connect("prompt_builder", "llm")
        
        return pipeline
    
    async def query(self, text: str, top_k: int = 5):
        # 1. Input guard
        valid, clean_text = self.input_guard.validate(text)
        if not valid:
            return {"error": "Invalid input"}
        
        # 2. Check semantic cache
        cached = await self.cache.get(clean_text)
        if cached:
            return cached
        
        # 3. Retrieve with Haystack
        result = self.pipeline.run({"query": clean_text, "top_k": top_k})
        
        # 4. Rerank
        reranked = self.reranker.rerank(clean_text, result["documents"], top_k)
        
        # 5. Output guard
        answer = self.output_guard.sanitize(result["llm"]["replies"][0])
        
        # 6. Cache result
        await self.cache.set(clean_text, answer)
        
        return {"answer": answer, "sources": reranked}
