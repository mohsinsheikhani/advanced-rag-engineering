"""Shared helpers for Session 3 RAG evaluation scripts.

Each per-config script in this package describes one cell of the diagnostic
matrix (baseline / bad retriever / bad embedder / bad generator). Boilerplate
— env loading, pipeline construction, metric evaluation — lives here so the
config files stay short and the only meaningful diff between them is the
component being broken.
"""
import csv
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load env before deepeval is imported by callers.
from dotenv import dotenv_values
_env = dotenv_values(REPO_ROOT / ".env")
for _k in ("CONFIDENT_API_KEY", "OPENAI_API_KEY"):
    if _k in _env:
        os.environ[_k] = _env[_k]

from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.components.embedders import OpenAITextEmbedder
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever

from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)
from deepeval import evaluate

from app.document_store import get_document_store
from retrieval.rag_pipeline import PROMPT_TEMPLATE as DEFAULT_PROMPT


DEFAULT_TESTSET = REPO_ROOT / "eval" / "datasets" / "synthetic_testset_raw.csv"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_LLM = "gpt-4o-mini"


def load_row(idx: int, path: Path = DEFAULT_TESTSET) -> tuple[str, str]:
    """Return (user_input, reference) for the given row of the testset CSV."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    r = rows[idx]
    return r["user_input"], r["reference"]


def default_embedder():
    return OpenAITextEmbedder(model=DEFAULT_EMBED_MODEL)


def build_pipelines(
    *,
    store=None,
    embedder_factory=default_embedder,
    top_k: int = 5,
    prompt_template: str = DEFAULT_PROMPT,
    llm_model: str = DEFAULT_LLM,
):
    """Build retrieval + RAG pipelines.

    Each pipeline gets its own embedder instance via ``embedder_factory``
    because Haystack components can't be shared across pipelines.
    """
    if store is None:
        store = get_document_store()

    retrieval = Pipeline()
    retrieval.add_component("text_embedder", embedder_factory())
    retrieval.add_component("retriever", QdrantEmbeddingRetriever(document_store=store, top_k=top_k))
    retrieval.connect("text_embedder.embedding", "retriever.query_embedding")

    rag = Pipeline()
    rag.add_component("text_embedder", embedder_factory())
    rag.add_component("retriever", QdrantEmbeddingRetriever(document_store=store, top_k=top_k))
    rag.add_component("prompt_builder", PromptBuilder(template=prompt_template))
    rag.add_component("llm", OpenAIGenerator(model=llm_model))
    rag.connect("text_embedder.embedding", "retriever.query_embedding")
    rag.connect("retriever.documents", "prompt_builder.documents")
    rag.connect("prompt_builder", "llm")

    return retrieval, rag


def run_and_evaluate(
    *,
    label: str,
    query: str,
    expected: str,
    store=None,
    **pipeline_kwargs,
):
    """Run retrieval + RAG, print the answer, then run the four-metric eval."""
    retrieval, rag = build_pipelines(store=store, **pipeline_kwargs)

    ret = retrieval.run({"text_embedder": {"text": query}})
    chunks = [d.content for d in ret["retriever"]["documents"]]

    out = rag.run({"text_embedder": {"text": query}, "prompt_builder": {"query": query}})
    answer = out["llm"]["replies"][0]

    print("=" * 70)
    print(label)
    print("=" * 70)
    print(f"Query:    {query}")
    print(f"Expected: {expected}")
    print("-" * 70)
    print(f"Answer:   {answer}")
    print(f"Retrieved {len(chunks)} chunk(s)")
    print("=" * 70)

    case = LLMTestCase(
        input=query,
        actual_output=answer,
        expected_output=expected,
        retrieval_context=chunks,
    )
    evaluate(
        test_cases=[case],
        metrics=[
            FaithfulnessMetric(threshold=0.7),
            ContextualPrecisionMetric(threshold=0.7),
            ContextualRecallMetric(threshold=0.7),
            AnswerRelevancyMetric(threshold=0.7),
        ],
    )
