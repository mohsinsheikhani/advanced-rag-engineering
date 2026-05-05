"""Session 3 — Baseline. Default pipeline, no failure injected."""
from eval.common import load_row, run_and_evaluate

ROW_INDEX = 1

query, expected = load_row(ROW_INDEX)
run_and_evaluate(label=f"BASELINE — row {ROW_INDEX}", query=query, expected=expected)
