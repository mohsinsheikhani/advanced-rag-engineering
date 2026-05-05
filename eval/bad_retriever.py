"""Session 3 — Config B (bad retriever). top_k=1.

Prediction:
- Contextual Recall:    DROP if gold info needs >1 chunk.
- Contextual Precision: HIGH if top-1 is relevant; LOW otherwise.
- Faithfulness:         likely STAYS HIGH — model faithfully reports
                        the (possibly incomplete) single chunk.
- Answer Relevancy:     mostly UNCHANGED.
"""
from eval.common import load_row, run_and_evaluate

ROW_INDEX = 1
TOP_K = 1

query, expected = load_row(ROW_INDEX)
run_and_evaluate(
    label=f"CONFIG B — bad retriever (top_k={TOP_K}) — row {ROW_INDEX}",
    query=query,
    expected=expected,
    top_k=TOP_K,
)
