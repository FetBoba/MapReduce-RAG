"""Single-process RAG ingestion pipeline."""

from __future__ import annotations

from time import perf_counter
from typing import List

from langchain_core.documents import Document

from ..chunking import chunk_documents
from ..config import PipelineConfig
from ..embeddings import embed_documents
from ..embeddings import EmbeddingFactory
from ..logging_utils import get_logger
from ..vector_store import VectorStoreManager
from .base import PipelineRunStats


class SequentialPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = get_logger("SequentialPipeline")
        self.embedding = EmbeddingFactory.build(config.embedding)
        self.vector_manager = VectorStoreManager(
            embedding=self.embedding,
            config=config.vector_store,
            index_name=config.index_name,
        )

    def build_index(self, documents: List[Document]) -> PipelineRunStats:
        timings = {}
        start = perf_counter()

        chunk_start = perf_counter()
        chunks = chunk_documents(documents, self.config.chunk)
        timings["chunk_seconds"] = perf_counter() - chunk_start

        embed_start = perf_counter()
        embedded = embed_documents(self.embedding, chunks)
        timings["embedding_seconds"] = perf_counter() - embed_start

        index_start = perf_counter()
        self.vector_manager.build_from_embedded(embedded)
        timings["index_seconds"] = perf_counter() - index_start

        duration = perf_counter() - start
        stats = PipelineRunStats(
            pipeline_type="sequential",
            documents_read=len(documents),
            chunks_processed=len(chunks),
            duration_seconds=duration,
            index_path=self.vector_manager.persist_path,
            stage_timings=timings,
        )
        self.logger.info(
            "Sequential pipeline indexed %s chunks in %.2fs", len(chunks), duration
        )
        return stats

    def as_retriever(self):
        return self.vector_manager.as_retriever(self.config.retrieval.top_k)
