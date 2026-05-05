"""
Session 1 eval — 5 hand-written questions with known answers.

Question coverage:
  Q1  easy / single-chunk  — what MIA stands for (what-is-mia.md §3)
  Q2  factual lookup       — what qualifies an agent referral (mia-referral-program.md §4)
  Q3  multi-chunk          — all the ways to earn inside MIA (how-income-is-generated-in-mia.md §5)
  Q4  reasoning            — why Mia routes to a human instead of answering directly (mia-boundaries-and-limitations.md §8)
  Q5  negative / boundary  — who MIA is NOT for (who-mia-is-for-and-not-for.md §4)
"""

# Importing eval.common loads .env (CONFIDENT_API_KEY, OPENAI_API_KEY) and
# fixes sys.path so app/* and retrieval/* imports below resolve.
from eval import common  # noqa: F401

from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualRelevancyMetric, ContextualPrecisionMetric
from deepeval import evaluate

from app.document_store import get_document_store
from retrieval.rag_pipeline import create_rag_pipeline
from retrieval.retrieval_pipeline import create_retrieval_pipeline


def run_rag(query: str) -> tuple[str, list[str]]:
    store = get_document_store()
    # Run retrieval first to capture chunks
    retrieval = create_retrieval_pipeline(store)
    ret_result = retrieval.run({"text_embedder": {"text": query}})
    docs = ret_result["retriever"]["documents"]
    chunks = [doc.content for doc in docs]
    # Run full RAG pipeline for the answer
    pipeline = create_rag_pipeline(store)
    result = pipeline.run({"text_embedder": {"text": query}, "prompt_builder": {"query": query}})
    answer = result["llm"]["replies"][0]
    return answer, chunks


questions = [
    ("What does MIA stand for?",
     "MIA stands for Make Income Anywhere."),
    ("What three steps must someone complete for an agent referral to qualify?",
     "The referred person must become properly licensed, join Modern Insurance Advisors as an agent, and submit their first piece of new business through the agency."),
    ("What are all the different ways someone can earn income inside MIA?",
     "Representation and affiliate income, service-based income through licensed insurance, compounding insurance residual income, product-based income, and licensing and platform income."),
    ("Why does Mia route users to a human instead of answering certain questions herself?",
     "Because certain areas require a license, involve compliance decisions, pricing, contracts, or personal accountability — situations where human judgment and responsibility are required."),
    ("Who is MIA not a good fit for?",
     "People looking for guaranteed results, fast money or shortcuts, to be told what to do, or pressure-based selling."),
]

test_cases = []
for q, expected in questions:
    answer, chunks = run_rag(q)
    test_cases.append(LLMTestCase(input=q, actual_output=answer, expected_output=expected, retrieval_context=chunks))

evaluate(
    test_cases=test_cases,
    metrics=[
        AnswerRelevancyMetric(threshold=0.7),
        FaithfulnessMetric(threshold=0.7),
        ContextualRelevancyMetric(threshold=0.7),
        ContextualPrecisionMetric(threshold=0.7),
    ],
)
