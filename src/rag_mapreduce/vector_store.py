"""Vector store helpers built on FAISS."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

from .config import VectorStoreConfig
from .embeddings import EmbeddedDocument


class VectorStoreManager:
    def __init__(self, embedding: Embeddings, config: VectorStoreConfig, index_name: str):
        self.embedding = embedding
        self.config = config
        self.index_name = index_name
        self.persist_path = Path(config.persist_dir) / index_name
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)

    def build_from_documents(self, documents) -> FAISS:
        store = FAISS.from_documents(list(documents), self.embedding)
        store.save_local(str(self.persist_path))
        return store

    def build_from_embedded(self, embedded_docs: Iterable[EmbeddedDocument]) -> FAISS:
        embedded_list = list(embedded_docs)
        if not embedded_list:
            raise ValueError("No embedded documents provided to build the vector store.")
        text_embeddings = [
            (doc.content, doc.embedding) for doc in embedded_list
        ]
        metadatas = [doc.metadata for doc in embedded_list]
        store = FAISS.from_embeddings(
            text_embeddings=text_embeddings,
            embedding=self.embedding,
            metadatas=metadatas,
        )
        store.save_local(str(self.persist_path))
        return store

    def load(self) -> FAISS:
        if not self.persist_path.exists():
            raise FileNotFoundError(
                f"Vector store not found at {self.persist_path}. Run the ingest command first."
            )
        return FAISS.load_local(
            str(self.persist_path),
            self.embedding,
            allow_dangerous_deserialization=True,
        )

    def as_retriever(self, top_k: int):
        store = self.load()
        return store.as_retriever(search_kwargs={"k": top_k})
