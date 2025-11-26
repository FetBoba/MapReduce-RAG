"""Configuration dataclasses and helpers for the RAG pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import yaml


@dataclass
class ChunkConfig:
    chunk_size: int = 900
    chunk_overlap: int = 120
    separators: list[str] = field(
        default_factory=lambda: ["\n\n", "\n", " ", ""]
    )


@dataclass
class EmbeddingConfig:
    provider: Literal["huggingface"] = "huggingface"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 64
    normalize: bool = True


@dataclass
class VectorStoreConfig:
    persist_dir: str = "artifacts/vectorstores"
    distance_metric: Literal["l2", "ip"] = "l2"


@dataclass
class MapReduceConfig:
    num_workers: int = 4
    batch_size: int = 256
    ray_address: Optional[str] = None


@dataclass
class RetrievalConfig:
    top_k: int = 5


@dataclass
class OllamaConfig:
    model: str = "llama3"
    temperature: float = 0.1
    base_url: Optional[str] = None


@dataclass
class PipelineConfig:
    dataset_path: str = "data/samples/mini_corpus.txt"
    artifact_dir: str = "artifacts/vectorstores"
    index_name: str = "classroom"
    pipeline_type: Literal["sequential", "mapreduce"] = "sequential"
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    mapreduce: MapReduceConfig = field(default_factory=MapReduceConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)

    @property
    def dataset(self) -> Path:
        return Path(self.dataset_path).expanduser()

    @property
    def persist_path(self) -> Path:
        directory = Path(self.vector_store.persist_dir).expanduser()
        return directory / self.index_name

    def ensure_artifacts(self) -> None:
        Path(self.artifact_dir).mkdir(parents=True, exist_ok=True)
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)


def _load_section(data: Dict[str, Any], section_key: str, target_type: Any) -> Any:
    section = data.get(section_key, {})
    return target_type(**section)


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw: Dict[str, Any] = yaml.safe_load(handle) or {}

    pipeline = PipelineConfig(
        dataset_path=raw.get("dataset_path", PipelineConfig.dataset_path),
        artifact_dir=raw.get("artifact_dir", PipelineConfig.artifact_dir),
        index_name=raw.get("index_name", PipelineConfig.index_name),
        pipeline_type=raw.get("pipeline_type", PipelineConfig.pipeline_type),
        chunk=_load_section(raw, "chunk", ChunkConfig),
        embedding=_load_section(raw, "embedding", EmbeddingConfig),
        vector_store=_load_section(raw, "vector_store", VectorStoreConfig),
        mapreduce=_load_section(raw, "mapreduce", MapReduceConfig),
        retrieval=_load_section(raw, "retrieval", RetrievalConfig),
        ollama=_load_section(raw, "ollama", OllamaConfig),
    )
    pipeline.ensure_artifacts()
    return pipeline
