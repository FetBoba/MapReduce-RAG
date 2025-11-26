"""Benchmark helpers for comparing pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Dict, Iterable, List

import pandas as pd
from langchain_core.documents import Document

from .config import PipelineConfig
from .pipelines.base import PipelineRunStats
from .pipelines.mapreduce import MapReducePipeline
from .pipelines.sequential import SequentialPipeline


@dataclass
class BenchmarkScenario:
    name: str
    config_path: str
    pipeline_type: str


@dataclass
class BenchmarkResult:
    scenario: str
    stats: PipelineRunStats


def _build_pipeline(config: PipelineConfig):
    if config.pipeline_type == "mapreduce":
        return MapReducePipeline(config)
    return SequentialPipeline(config)


def ingest_and_time(config: PipelineConfig, documents: List[Document]) -> PipelineRunStats:
    pipeline = _build_pipeline(config)
    return pipeline.build_index(documents)


def benchmark_pipelines(
    configs: Iterable[PipelineConfig],
    document_loader: Callable[[PipelineConfig], List[Document]],
) -> List[BenchmarkResult]:
    results: List[BenchmarkResult] = []
    for config in configs:
        docs = document_loader(config)
        stats = ingest_and_time(config, docs)
        results.append(
            BenchmarkResult(
                scenario=f"{config.index_name}-{config.pipeline_type}", stats=stats
            )
        )
    return results


def results_to_frame(results: List[BenchmarkResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        stats = result.stats
        rows.append(
            {
                "scenario": result.scenario,
                "pipeline": stats.pipeline_type,
                "documents": stats.documents_read,
                "chunks": stats.chunks_processed,
                "seconds": stats.duration_seconds,
                "index_path": str(stats.index_path),
            }
        )
    return pd.DataFrame(rows)
