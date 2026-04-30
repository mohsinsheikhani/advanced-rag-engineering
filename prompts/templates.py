"""Prompt templates with versioning."""

RAG_GENERATION_V1 = """Answer the question based on the context below.

Context:
{context}

Question: {question}

Answer:"""

QUERY_CLASSIFICATION_V1 = """Classify the query type: factual, analytical, or conversational.

Query: {query}

Type:"""

DOCUMENT_GRADING_V1 = """Grade document relevance: relevant, partially_relevant, or irrelevant.

Query: {query}
Document: {document}

Grade:"""
