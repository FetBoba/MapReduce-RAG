"""Typer CLI entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import typer
from rich.console import Console
from rich.table import Table

from .config import PipelineConfig, load_pipeline_config
from .data_loader import load_documents
from .evaluation import benchmark_pipelines, results_to_frame
from .llm import build_llm
from .logging_utils import get_logger
from .rag_chain import build_rag_chain
from .pipelines.mapreduce import MapReducePipeline
from .pipelines.sequential import SequentialPipeline

app = typer.Typer(add_completion=False, help="Scalable RAG pipelines with LangChain + Ray")
console = Console()
logger = get_logger("CLI")


def _select_pipeline(config: PipelineConfig):
    if config.pipeline_type == "mapreduce":
        return MapReducePipeline(config)
    return SequentialPipeline(config)


def _load_configs(paths: Iterable[Path]) -> List[PipelineConfig]:
    return [load_pipeline_config(path) for path in paths]


@app.command()
def ingest(
    config_path: Path = typer.Option(
        Path("configs/default.yaml"), "--config", "-c", help="Path to YAML config"
    )
) -> None:
    config = load_pipeline_config(config_path)
    docs = load_documents(config.dataset)
    pipeline = _select_pipeline(config)
    stats = pipeline.build_index(docs)
    console.print(
        f"[bold green]Indexed[/bold green] {stats.chunks_processed} chunks in {stats.duration_seconds:.2f}s -> {stats.index_path}"
    )


@app.command()
def query(
    question: str = typer.Argument(..., help="User question to route through RAG"),
    config_path: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    prompt_template: Optional[str] = typer.Option(
        None,
        "--prompt",
        help="Custom prompt template overriding the default",
    ),
) -> None:
    config = load_pipeline_config(config_path)
    pipeline = _select_pipeline(config)
    retriever = pipeline.as_retriever()
    llm = build_llm(config.ollama)
    chain = build_rag_chain(llm, retriever, prompt_template)
    response = chain.invoke({"query": question})
    answer = response.get("result") if isinstance(response, dict) else response
    console.rule("Answer")
    console.print(answer)
    sources = response.get("source_documents", []) if isinstance(response, dict) else []
    if sources:
        table = Table(title="Sources", show_header=True, header_style="bold magenta")
        table.add_column("Rank", justify="right")
        table.add_column("Source")
        table.add_column("Preview")
        for idx, doc in enumerate(sources[:5], start=1):
            preview = doc.page_content[:120].replace("\n", " ") + "..."
            table.add_row(str(idx), str(doc.metadata.get("source")), preview)
        console.print(table)


@app.command()
def benchmark(
    config_paths: List[Path] = typer.Argument(
        ..., help="List of config files to benchmark", metavar="CONFIG"
    )
) -> None:
    configs = _load_configs(config_paths)
    results = benchmark_pipelines(configs, lambda cfg: load_documents(cfg.dataset))
    frame = results_to_frame(results)
    table = Table(title="Ingestion Benchmarks", show_lines=False)
    for column in frame.columns:
        table.add_column(column)
    for _, row in frame.iterrows():
        table.add_row(*(str(row[col]) for col in frame.columns))
    console.print(table)


def run() -> None:
    app()


if __name__ == "__main__":
    run()
