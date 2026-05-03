from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.components.embedders import OpenAITextEmbedder
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore


PROMPT_TEMPLATE = """
You are a helpful assistant answering questions about MIA (Make Income Anywhere).

Use the following context to answer the user's question. If the context doesn't contain enough information, say so.

Context:
{% for doc in documents %}
{{ doc.content }}
---
{% endfor %}

Question: {{ query }}

Answer:
"""


def create_rag_pipeline(document_store: QdrantDocumentStore) -> Pipeline:
    pipeline = Pipeline()
    
    pipeline.add_component("text_embedder", OpenAITextEmbedder(model="text-embedding-3-small"))
    pipeline.add_component("retriever", QdrantEmbeddingRetriever(document_store=document_store, top_k=5))
    pipeline.add_component("prompt_builder", PromptBuilder(template=PROMPT_TEMPLATE))
    pipeline.add_component("llm", OpenAIGenerator(model="gpt-4o-mini"))
    
    pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder", "llm")
    
    return pipeline
