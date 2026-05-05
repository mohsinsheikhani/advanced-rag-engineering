"""Session 3 — Config D (bad generator).

Retrieval untouched. Swap LLM → gpt-3.5-turbo and use a "be creative,
fill gaps" prompt to encourage hallucination.

Prediction:
- Faithfulness:         DROP via hallucination.
- Contextual Precision: UNCHANGED (retrieval untouched).
- Contextual Recall:    UNCHANGED (retrieval untouched).
- Answer Relevancy:     may WOBBLE.

Observed on row 1: Faithfulness stayed at 1.00 because the gold chunk fully
covered the answer — there was no gap to fill. See bad_generator_oos.py for
the out-of-scope variant that forces a real gap.
"""
from eval.common import load_row, run_and_evaluate

ROW_INDEX = 1
BAD_LLM = "gpt-3.5-turbo"

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

query, expected = load_row(ROW_INDEX)
run_and_evaluate(
    label=f"CONFIG D — bad generator ({BAD_LLM} + creative prompt) — row {ROW_INDEX}",
    query=query,
    expected=expected,
    prompt_template=BAD_PROMPT,
    llm_model=BAD_LLM,
)
