import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import dotenv_values
_env = dotenv_values(REPO_ROOT / ".env")
if "OPENAI_API_KEY" in _env:
    os.environ["OPENAI_API_KEY"] = _env["OPENAI_API_KEY"]

import openai
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from ragas.testset import TestsetGenerator
from ragas.testset.transforms import default_transforms
from ragas.testset.transforms.splitters.headline import HeadlineSplitter

loader = DirectoryLoader(
    str(REPO_ROOT / "knowledge_base"), glob="**/*.md",
    loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}
)
docs = loader.load()
print(f"Loaded {len(docs)} documents")

client = openai.OpenAI()
generator_llm = llm_factory("gpt-4o-mini", client=client)
generator_embeddings = OpenAIEmbeddings(client=client, model="text-embedding-3-small")

transforms = default_transforms(documents=docs, llm=generator_llm, embedding_model=generator_embeddings)

for t in transforms:
    if isinstance(t, HeadlineSplitter):
        t.filter_nodes = lambda node: node.properties.get("headlines") is not None

generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
dataset = generator.generate_with_langchain_docs(docs, testset_size=20, transforms=transforms)

output_path = REPO_ROOT / "eval" / "datasets" / "synthetic_testset_raw.csv"
df = dataset.to_pandas()
df.to_csv(output_path, index=False)
print(f"\nWrote {output_path}")
print("\n=== Generated Questions ===\n")
print(df[["user_input", "reference"]].to_string())
