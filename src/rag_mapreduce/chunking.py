"""Document chunking helpers."""

from __future__ import annotations

from typing import Iterable, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import ChunkConfig


def build_text_splitter(config: ChunkConfig) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=config.separators,
    )


def chunk_documents(documents: Iterable[Document], config: ChunkConfig) -> List[Document]:
    splitter = build_text_splitter(config)
    return splitter.split_documents(list(documents))
