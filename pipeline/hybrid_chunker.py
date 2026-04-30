"""Hybrid chunker: whole-doc for short files, H2-header split for longer ones."""
import re
import json
from typing import List
from haystack import component, Document


WORD_THRESHOLD = 400   # files under this → embed whole
MIN_SECTION_WORDS = 100  # sections under this → merge with next


def _extract_metadata(text: str) -> dict:
    """Pull fields from the trailing JSON metadata block, if present."""
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not match:
        return {}
    try:
        data = json.loads(match.group(1))
        return {k: data[k] for k in ("module", "journey_stage", "user_type") if k in data}
    except json.JSONDecodeError:
        return {}


def _strip_metadata_block(text: str) -> str:
    """Remove the trailing JSON metadata block from content."""
    return re.sub(r"\n*```json\s*\{.*?\}\s*```\s*$", "", text, flags=re.DOTALL).strip()


def _split_by_h2(text: str) -> List[str]:
    """Split on ## headers, merge sections under MIN_SECTION_WORDS with the next."""
    parts = re.split(r"(?=^## )", text, flags=re.MULTILINE)
    parts = [p.strip() for p in parts if p.strip()]

    merged, i = [], 0
    while i < len(parts):
        section = parts[i]
        # keep merging forward while this section is too short and there's a next one
        while len(section.split()) < MIN_SECTION_WORDS and i + 1 < len(parts):
            i += 1
            section = section + "\n\n" + parts[i]
        merged.append(section)
        i += 1
    return merged


@component
class HybridMarkdownChunker:
    """
    Hybrid chunking for structured single-topic Markdown files.
    - Files ≤ WORD_THRESHOLD words  → one chunk (whole document)
    - Files >  WORD_THRESHOLD words → split by H2 headers, merging short sections
    JSON metadata block is stripped from content and stored as Qdrant payload.
    """

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
                for section in _split_by_h2(clean):
                    output.append(Document(content=section, meta=base_meta))

        return {"documents": output}
