"""Embedding helpers and serialization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

from .config import EmbeddingConfig


@dataclass
class EmbeddedDocument:
    content: str
    metadata: dict
    embedding: List[float]


class EmbeddingFactory:
    @staticmethod
    def build(config: EmbeddingConfig) -> Embeddings:
        if config.provider != "huggingface":
            raise ValueError(f"Unsupported embedding provider: {config.provider}")
        return HuggingFaceEmbeddings(
            model_name=config.model_name,
            encode_kwargs={"normalize_embeddings": config.normalize},
            model_kwargs={"device": "cpu"},
        )


def embed_documents(embedder: Embeddings, documents: List[Document]) -> List[EmbeddedDocument]:
    if not documents:
        return []
    texts = [doc.page_content for doc in documents]
    vectors = embedder.embed_documents(texts)
    return [
        EmbeddedDocument(content=text, metadata=doc.metadata, embedding=vector)
        for text, doc, vector in zip(texts, documents, vectors)
    ]
