"""Judge prompts for evaluation."""

ANSWER_RELEVANCE = """Rate answer relevance to the question (1-5).

Question: {question}
Answer: {answer}

Score:"""

FAITHFULNESS = """Is the answer faithful to the context? (yes/no)

Context: {context}
Answer: {answer}

Faithful:"""
