"""Ingest: load corpus -> split -> embed -> persist a ChromaDB collection.

The collection is named by ingest-affecting params (chunk_size, overlap, embed model) so
experiments that differ only in top_k / prompt transparently reuse an existing store
instead of re-embedding (saves API calls — see ExperimentConfig.collection_name).

  python -m src.ingest --config configs/baseline.yaml
  python -m src.ingest --config configs/baseline.yaml --rebuild   # force re-embed
"""
from __future__ import annotations

import argparse

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config as C
from .bedrock import embeddings
from .config import ExperimentConfig, load_config


def load_corpus() -> list[Document]:
    if not any(C.CORPUS_DIR.glob("*.txt")):
        raise SystemExit(f"No .txt files in {C.CORPUS_DIR}. Run: python -m src.fetch_corpus")
    loader = DirectoryLoader(
        str(C.CORPUS_DIR), glob="*.txt",
        loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    docs = loader.load()
    for d in docs:
        # carry a short source name for per-slice analysis later
        d.metadata["source_name"] = d.metadata.get("source", "").split("/")[-1]
    return docs


def split(docs: list[Document], cfg: ExperimentConfig) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap,
        length_function=len, add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    for ch in chunks:
        ch.metadata["doc_len"] = len(ch.page_content)
    return chunks


def collection_exists(cfg: ExperimentConfig) -> bool:
    store = Chroma(
        collection_name=cfg.collection_name,
        embedding_function=embeddings(cfg.embed_model),
        persist_directory=str(C.CHROMA_DIR),
    )
    try:
        return store._collection.count() > 0
    except Exception:
        return False


def build_store(cfg: ExperimentConfig, rebuild: bool = False) -> Chroma:
    emb = embeddings(cfg.embed_model)
    if collection_exists(cfg) and not rebuild:
        print(f"Collection '{cfg.collection_name}' already populated — reusing (use --rebuild to force).")
        return Chroma(collection_name=cfg.collection_name, embedding_function=emb,
                      persist_directory=str(C.CHROMA_DIR))

    if rebuild:
        try:
            Chroma(collection_name=cfg.collection_name, embedding_function=emb,
                   persist_directory=str(C.CHROMA_DIR)).delete_collection()
            print(f"Deleted existing collection '{cfg.collection_name}'.")
        except Exception:
            pass

    docs = load_corpus()
    chunks = split(docs, cfg)
    print(f"Loaded {len(docs)} docs -> {len(chunks)} chunks "
          f"(chunk_size={cfg.chunk_size}, overlap={cfg.chunk_overlap}). Embedding with {cfg.embed_model}...")
    store = Chroma.from_documents(
        documents=chunks, embedding=emb,
        collection_name=cfg.collection_name, persist_directory=str(C.CHROMA_DIR),
    )
    print(f"Persisted {store._collection.count()} vectors to collection '{cfg.collection_name}'.")
    return store


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the ChromaDB vector store for a config")
    ap.add_argument("--config", required=True)
    ap.add_argument("--rebuild", action="store_true", help="Delete and re-embed even if the collection exists")
    args = ap.parse_args()
    build_store(load_config(args.config), rebuild=args.rebuild)


if __name__ == "__main__":
    main()
