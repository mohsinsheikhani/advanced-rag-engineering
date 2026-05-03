"""Hybrid chunker: whole-doc for short files, H2-header split for longer ones."""
import re
import json
from typing import List
from haystack import component, Document
from haystack.components.preprocessors import MarkdownHeaderSplitter

WORD_THRESHOLD = 400  # files under this → embed whole

_splitter = MarkdownHeaderSplitter(header_split_levels=[2], keep_headers=True)

# Section headers that add no retrieval value — filtered regardless of section number
SKIP_HEADERS = {"document header", "metadata block", "gentle invitation", "gentle next-step framing"}


def _is_boilerplate(chunk_content: str) -> bool:
    first_line = chunk_content.strip().splitlines()[0].lstrip("#").strip().lower()
    return any(skip in first_line for skip in SKIP_HEADERS)


def _extract_metadata(text: str) -> dict:
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not match:
        return {}
    try:
        data = json.loads(match.group(1))
        return {k: data[k] for k in ("module", "journey_stage", "user_type") if k in data}
    except json.JSONDecodeError:
        return {}


def _strip_metadata_block(text: str) -> str:
    return re.sub(r"\n*```json\s*\{.*?\}\s*```\s*$", "", text, flags=re.DOTALL).strip()


@component
class HybridMarkdownChunker:
    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> dict:
        output = []
        for doc in documents:
            meta_fields = _extract_metadata(doc.content)
            base_meta = {**doc.meta, **meta_fields}
            clean = _strip_metadata_block(doc.content)

            if len(clean.split()) <= WORD_THRESHOLD:
                output.append(Document(content=clean, meta=base_meta))
            else:
                chunks = _splitter.run(documents=[Document(content=clean, meta=base_meta)])["documents"]
                output.extend(c for c in chunks if not _is_boilerplate(c.content))

        return {"documents": output}
