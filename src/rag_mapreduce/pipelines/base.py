"""Shared pipeline models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PipelineRunStats:
    pipeline_type: str
    documents_read: int
    chunks_processed: int
    duration_seconds: float
    index_path: Path
