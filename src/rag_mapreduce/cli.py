"""Typer CLI entrypoints."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional

import typer
from rich.console import Console
from rich.table import Table

from .config import PipelineConfig, load_pipeline_config
from .data_loader import load_documents
from .evaluation import benchmark_pipelines, results_to_frame
from .llm import build_llm
from .logging_utils import get_logger
from .rag_chain import SimpleRAGChain, build_rag_chain, build_simple_chain
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


def _format_stage_name(key: str) -> str:
    label = key.replace("_seconds", "").replace("_", " ").strip()
    return label.title() if label else key


def _print_timings(title: str, timings: Dict[str, float]) -> None:
    if not timings:
        return
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Stage")
    table.add_column("Seconds", justify="right")
    for stage, seconds in timings.items():
        table.add_row(_format_stage_name(stage), f"{seconds:.2f}")
    console.print(table)


@app.command()
def ingest(
    config_path: Path = typer.Option(
        Path("configs/default.yaml"), "--config", "-c", help="Path to YAML config"
    ),
    measure: bool = typer.Option(
        False, "--measure", help="Print stage timings for embedding/indexing"
    ),
) -> None:
    config = load_pipeline_config(config_path)
    docs = load_documents(config.dataset)
    pipeline = _select_pipeline(config)
    stats = pipeline.build_index(docs)
    console.print(
        f"[bold green]Indexed[/bold green] {stats.chunks_processed} chunks in {stats.duration_seconds:.2f}s -> {stats.index_path}"
    )
    if measure and getattr(stats, "stage_timings", None):
        _print_timings("Ingestion Timings", stats.stage_timings)


@app.command()
def query(
    question: str = typer.Argument(..., help="User question to route through RAG"),
    config_path: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    prompt_template: Optional[str] = typer.Option(
        None,
        "--prompt",
        help="Custom prompt template overriding the default",
    ),
    measure: bool = typer.Option(
        False, "--measure", help="Capture retrieval and end-to-end query timings"
    ),
) -> None:
    config = load_pipeline_config(config_path)
    pipeline = _select_pipeline(config)
    retriever = pipeline.as_retriever()
    llm = build_llm(config.ollama)
    chain = build_rag_chain(llm, retriever, prompt_template)
    response: Any
    query_timings: Dict[str, float] = {}

    if measure:
        start = perf_counter()
        if isinstance(chain, SimpleRAGChain):
            response, chain_timings = chain.run_with_metrics({"query": question})
        else:
            manual_chain = build_simple_chain(llm, retriever, prompt_template)
            response, chain_timings = manual_chain.run_with_metrics({"query": question})
        query_timings.update(chain_timings)
        query_timings["end_to_end_seconds"] = perf_counter() - start
    else:
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
    if measure and query_timings:
        ordered_metrics = {
            key: query_timings[key]
            for key in (
                "retrieval_seconds",
                "generation_seconds",
                "total_seconds",
                "end_to_end_seconds",
            )
            if key in query_timings
        }
        _print_timings("Query Timings", ordered_metrics)


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
