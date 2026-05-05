"""Session 3 — Config D' (bad generator, out-of-scope query).

Same generator setup as Config D, but the query is deliberately out-of-scope:
pricing is explicitly excluded from book-ordering-process.md (§11). The gold
answer is "the corpus doesn't specify a price" — which forces the creative
generator to either refuse honestly or invent a price (hallucinate).

Prediction:
- Faithfulness:         DROP if the model invents a price.
- Contextual Recall:    LOW — no positive ground-truth claim to attribute.
                        (Recall is uninformative for refusal-type ground truths.)
- Contextual Precision: probably HIGH (chunks are still topical).
- Answer Relevancy:     likely HIGH (answer is on-topic for the question).
"""
from eval.common import run_and_evaluate

BAD_LLM = "gpt-3.5-turbo"
QUERY = "How much does The Unfolding cost in US dollars?"
EXPECTED = (
    "The knowledge base does not specify a price for The Unfolding. "
    "Pricing and promotions are explicitly out of scope for the book ordering "
    "documentation and are handled at checkout."
)

BAD_PROMPT = """
You are a helpful assistant answering questions about MIA (Make Income Anywhere).

Be creative and confident. If the context doesn't fully answer the question, fill any gaps
with reasonable guesses based on what a service like this would plausibly offer. Do not
say the context is insufficient — give a complete, confident answer either way.

Context:
{% for doc in documents %}
{{ doc.content }}
---
{% endfor %}

Question: {{ query }}

Answer:
"""

run_and_evaluate(
    label=f"CONFIG D' — bad generator + out-of-scope query",
    query=QUERY,
    expected=EXPECTED,
    prompt_template=BAD_PROMPT,
    llm_model=BAD_LLM,
)
