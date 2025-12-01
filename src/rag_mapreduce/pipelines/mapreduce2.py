"""Ray-powered MapReduce ingestion pipeline."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Iterable, List, Dict

import ray
from langchain_core.documents import Document

from ..chunking import chunk_documents
from ..config import PipelineConfig, ChunkConfig, EmbeddingConfig
from ..embeddings import EmbeddedDocument, EmbeddingFactory, embed_documents
from ..logging_utils import get_logger
from ..vector_store import VectorStoreManager
from .base import PipelineRunStats


@dataclass
class EmbeddedBatchResult:
    documents: List[EmbeddedDocument]
    timings: Dict[str, float]


def _batch_documents(documents: List[Document], batch_size: int) -> List[List[Document]]:
    start_time = perf_counter()
    
    batches: List[List[Document]] = []
    current: List[Document] = []
    for doc in documents:
        current.append(doc)
        if len(current) >= batch_size:
            batches.append(current)
            current = []
    if current:
        batches.append(current)
    
    duration = perf_counter() - start_time
    print(f"_batch_documents executed in {duration:.4f} seconds")
    return batches


@ray.remote
def _map_chunk_and_embed(
    batch: List[Document], chunk_config: dict, embedding_config: dict
) -> EmbeddedBatchResult:
    start_time = perf_counter()
    
    timings: Dict[str, float] = {}
    chunk = ChunkConfig(**chunk_config)
    embed_cfg = EmbeddingConfig(**embedding_config)
    
    chunk_start = perf_counter()
    chunks = chunk_documents(batch, chunk)
    timings["chunk_seconds"] = perf_counter() - chunk_start
    
    embedder = EmbeddingFactory.build(embed_cfg)
    embed_start = perf_counter()
    embedded = embed_documents(embedder, chunks)
    timings["embedding_seconds"] = perf_counter() - embed_start
    
    duration = perf_counter() - start_time
    timings["total_remote_seconds"] = duration  # Store in timings to return to main process
    
    # Also print from remote process (might not show up, but worth trying)
    print(f"[REMOTE] _map_chunk_and_embed executed in {duration:.4f} seconds")
    
    return EmbeddedBatchResult(documents=embedded, timings=timings)


class MapReducePipeline:
    def __init__(self, config: PipelineConfig):
        start_time = perf_counter()
        
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
                num_cpus=8,  # Force 8 CPUs to ensure 8 workers
                include_dashboard=False,
                ignore_reinit_error=True,
                log_to_driver=False,
            )
        self.chunk_dict = asdict(config.chunk)
        self.embedding_dict = asdict(config.embedding)
        
        duration = perf_counter() - start_time
        print(f"MapReducePipeline.__init__ executed in {duration:.4f} seconds")

    def build_index(self, documents: List[Document]) -> PipelineRunStats:
        start_time = perf_counter()
        
        start = perf_counter()
        
        # Calculate batch size to ensure exactly 8 batches
        target_workers = 8
        batch_size = max(1, len(documents) // target_workers)
        print(f"Creating exactly {target_workers} batches with batch size {batch_size}")
        
        batches = _batch_documents(documents, batch_size)
        
        # If we have fewer batches than target workers, create empty batches to reach 8
        if len(batches) < target_workers:
            print(f"Only {len(batches)} batches created, adding {target_workers - len(batches)} empty batches")
            batches.extend([[] for _ in range(target_workers - len(batches))])
        # If we have more batches than target workers, redistribute to exactly 8
        elif len(batches) > target_workers:
            print(f"Redistributing {len(batches)} batches into exactly {target_workers} batches")
            new_batches = [[] for _ in range(target_workers)]
            for i, batch in enumerate(batches):
                new_batches[i % target_workers].extend(batch)
            batches = new_batches
        
        print(f"Final batch distribution: {[len(batch) for batch in batches]} documents per batch")
        
        futures = [
            _map_chunk_and_embed.remote(batch, self.chunk_dict, self.embedding_dict)
            for batch in batches
        ]
        embedded_batches: List[EmbeddedBatchResult] = ray.get(futures)
        merged: List[EmbeddedDocument] = []
        stage_timings = defaultdict(float)
        
        # Print remote function timings and collect stats
        remote_timings = []
        for i, batch in enumerate(embedded_batches):
            merged.extend(batch.documents)
            for name, value in batch.timings.items():
                stage_timings[name] += value
            
            # Check for the remote timing in the returned timings dict
            remote_time = batch.timings.get("total_remote_seconds")
            if remote_time is not None:
                remote_timings.append(remote_time)
                print(f"_map_chunk_and_embed task {i} executed in {remote_time:.4f} seconds")
            else:
                # Fallback: calculate from individual components
                total_remote = batch.timings.get("chunk_seconds", 0) + batch.timings.get("embedding_seconds", 0)
                remote_timings.append(total_remote)
                print(f"_map_chunk_and_embed task {i} executed in {total_remote:.4f} seconds (calculated)")
        
        index_start = perf_counter()
        self.vector_manager.build_from_embedded(merged)
        stage_timings["index_seconds"] = perf_counter() - index_start
        duration = perf_counter() - start
        
        if remote_timings:
            avg_remote_time = sum(remote_timings) / len(remote_timings)
            print(f"Average _map_chunk_and_embed time: {avg_remote_time:.4f} seconds")
            print(f"Total remote processing time: {sum(remote_timings):.4f} seconds across {len(remote_timings)} tasks")
        
        stats = PipelineRunStats(
            pipeline_type="mapreduce",
            documents_read=len(documents),
            chunks_processed=len(merged),
            duration_seconds=duration,
            index_path=self.vector_manager.persist_path,
            stage_timings=dict(stage_timings),
        )
        self.logger.info(
            "MapReduce pipeline indexed %s chunks across %s Ray tasks in %.2fs",
            len(merged),
            len(futures),
            duration,
        )
        
        build_index_duration = perf_counter() - start_time
        print(f"MapReducePipeline.build_index executed in {build_index_duration:.4f} seconds")
        
        # Print detailed breakdown
        print("\n=== TIMING BREAKDOWN ===")
        print(f"Batch creation: {stage_timings.get('chunk_seconds', 0):.4f}s")
        print(f"Embedding: {stage_timings.get('embedding_seconds', 0):.4f}s")
        print(f"Index building: {stage_timings.get('index_seconds', 0):.4f}s")
        print(f"Number of Ray workers used: {len(futures)}")
        print("========================")
        
        return stats

    def as_retriever(self):
        start_time = perf_counter()
        
        result = self.vector_manager.as_retriever(self.config.retrieval.top_k)
        
        duration = perf_counter() - start_time
        print(f"MapReducePipeline.as_retriever executed in {duration:.4f} seconds")
        return result


# Example of how you would use this and see all timings:
if __name__ == "__main__":
    # This would typically be in your main execution code
    print("=== MapReduce Pipeline Execution Timings ===")
    # Your pipeline setup and execution would go here
    # config = PipelineConfig(...)
    # pipeline = MapReducePipeline(config)  # Will print __init__ timing
    # stats = pipeline.build_index(documents)  # Will print build_index and _batch_documents timings
    # retriever = pipeline.as_retriever()  # Will print as_retriever timing
    print("============================================")
