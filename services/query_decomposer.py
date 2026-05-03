"""Expand a query into multiple rephrasings to improve retrieval recall."""
from openai import OpenAI
from app.config import settings

_client = OpenAI(api_key=settings.openai_api_key)

_PROMPT = """Generate 3 different rephrasings of the following search query.
Use plain, direct language. Vary the vocabulary.
Return only the rephrasings, one per line, no numbering or extra text.

Query: {query}"""


class QueryDecomposer:
    def decompose(self, query: str) -> list[str]:
        """Return original query + 3 rephrasings."""
        response = _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": _PROMPT.format(query=query)}],
            temperature=0.3,
        )
        rephrasings = response.choices[0].message.content.strip().splitlines()
        return [query] + [r.strip() for r in rephrasings if r.strip()]
