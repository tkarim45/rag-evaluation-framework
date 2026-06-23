"""The RAG pipeline: retrieve top-k contexts -> stuff into prompt -> generate answer.

`answer()` returns {answer, contexts} — RAGAS needs both the generated answer and the
exact retrieved contexts to compute Faithfulness / Answer Relevancy / Context Recall.
"""
from __future__ import annotations

import argparse

from langchain_chroma import Chroma
from langchain_core.documents import Document

from . import config as C
from .bedrock import embeddings, gen_llm
from .config import ExperimentConfig, load_config
from .prompts import get_prompt


class RAGPipeline:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.store = Chroma(
            collection_name=cfg.collection_name,
            embedding_function=embeddings(cfg.embed_model),
            persist_directory=str(C.CHROMA_DIR),
        )
        if self.store._collection.count() == 0:
            raise SystemExit(
                f"Collection '{cfg.collection_name}' is empty. Run: "
                f"python -m src.ingest --config {cfg.name}"
            )
        self.llm = gen_llm(cfg.gen_model)
        self.prompt = get_prompt(cfg.prompt)

    def retrieve(self, question: str) -> list[Document]:
        return self.store.similarity_search(question, k=self.cfg.top_k)

    def answer(self, question: str) -> dict:
        docs = self.retrieve(question)
        context = "\n\n---\n\n".join(d.page_content for d in docs)
        msg = self.prompt.format(context=context, question=question)
        resp = self.llm.invoke(msg)
        return {
            "answer": resp.content if isinstance(resp.content, str) else str(resp.content),
            "contexts": [d.page_content for d in docs],
            "sources": [d.metadata.get("source_name", "") for d in docs],
        }


def main() -> None:
    ap = argparse.ArgumentParser(description="Ask the RAG pipeline one question (smoke test)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--q", required=True, help="Question to ask")
    args = ap.parse_args()
    pipe = RAGPipeline(load_config(args.config))
    out = pipe.answer(args.q)
    print("\n=== ANSWER ===\n" + out["answer"])
    print(f"\n=== {len(out['contexts'])} CONTEXTS (sources: {out['sources']}) ===")
    for i, c in enumerate(out["contexts"], 1):
        print(f"\n[{i}] {c[:300]}{'...' if len(c) > 300 else ''}")


if __name__ == "__main__":
    main()
