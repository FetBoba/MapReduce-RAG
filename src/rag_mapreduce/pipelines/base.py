"""Shared pipeline models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


@dataclass
class PipelineRunStats:
    pipeline_type: str
    documents_read: int
    chunks_processed: int
    duration_seconds: float
    index_path: Path
    stage_timings: Dict[str, float] = field(default_factory=dict)
