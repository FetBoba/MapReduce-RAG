"""Ray-powered MapReduce ingestion pipeline."""

from __future__ import annotations

from dataclasses import asdict
from time import perf_counter
from typing import Iterable, List

import ray
from langchain_core.documents import Document

from ..chunking import chunk_documents
from ..config import PipelineConfig, ChunkConfig, EmbeddingConfig
from ..embeddings import EmbeddedDocument, EmbeddingFactory, embed_documents
from ..logging_utils import get_logger
from ..vector_store import VectorStoreManager
from .base import PipelineRunStats


def _batch_documents(documents: List[Document], batch_size: int) -> List[List[Document]]:
    batches: List[List[Document]] = []
    current: List[Document] = []
    for doc in documents:
        current.append(doc)
        if len(current) >= batch_size:
            batches.append(current)
            current = []
    if current:
        batches.append(current)
    return batches


@ray.remote
def _map_chunk_and_embed(
    batch: List[Document], chunk_config: dict, embedding_config: dict
) -> List[EmbeddedDocument]:
    chunk = ChunkConfig(**chunk_config)
    embed_cfg = EmbeddingConfig(**embedding_config)
    chunks = chunk_documents(batch, chunk)
    embedder = EmbeddingFactory.build(embed_cfg)
    return embed_documents(embedder, chunks)


class MapReducePipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = get_logger("MapReducePipeline")
        self.embedding = EmbeddingFactory.build(config.embedding)
        self.vector_manager = VectorStoreManager(
            embedding=self.embedding,
            config=config.vector_store,
            index_name=config.index_name,
        )
        if not ray.is_initialized():
            ray.init(
                address=config.mapreduce.ray_address,
                include_dashboard=False,
                ignore_reinit_error=True,
                log_to_driver=False,
            )
        self.chunk_dict = asdict(config.chunk)
        self.embedding_dict = asdict(config.embedding)

    def build_index(self, documents: List[Document]) -> PipelineRunStats:
        start = perf_counter()
        batches = _batch_documents(documents, self.config.mapreduce.batch_size)
        futures = [
            _map_chunk_and_embed.remote(batch, self.chunk_dict, self.embedding_dict)
            for batch in batches
        ]
        embedded_batches: List[List[EmbeddedDocument]] = ray.get(futures)
        merged: List[EmbeddedDocument] = [item for batch in embedded_batches for item in batch]
        self.vector_manager.build_from_embedded(merged)
        duration = perf_counter() - start
        stats = PipelineRunStats(
            pipeline_type="mapreduce",
            documents_read=len(documents),
            chunks_processed=len(merged),
            duration_seconds=duration,
            index_path=self.vector_manager.persist_path,
        )
        self.logger.info(
            "MapReduce pipeline indexed %s chunks across %s Ray tasks in %.2fs",
            len(merged),
            len(futures),
            duration,
        )
        return stats

    def as_retriever(self):
        return self.vector_manager.as_retriever(self.config.retrieval.top_k)
