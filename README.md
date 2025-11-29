# Scalable Retrieval-Augmented Generation with MapReduce

This project implements two Retrieval-Augmented Generation (RAG) ingestion paths using LangChain and Ollama:

1. **Basic / Sequential** – single-process pipeline for smaller corpora. Config: `configs/basic_sequential.yaml`.
2. **Parallel / MapReduce** – Ray-powered ingestion that distributes chunking + embedding before merging a FAISS index. Config: `configs/mapreduce_parallel.yaml`.

The goal is to quantify indexing speedups and query throughput as corpus sizes grow from megabytes to multi-gigabyte workloads for COMP4651.

## Features

- LangChain loaders, text splitters, FAISS vector stores, and RetrievalQA chains.
- Local LLM orchestration through Ollama (e.g., `llama3`, `mistral`).
- Ray MapReduce ingestion that parallelizes chunking + embedding.
- Config-driven workflow (`configs/*.yaml`) with ready-made basic vs mapreduce presets and tunable knobs.
- Typer CLI for ingestion, querying, and benchmarking.
- Benchmark helpers that output tidy tables for plotting build-time speedups.

## Quickstart

1. **Install system prerequisites**
   - Python 3.10+
   - [Ollama](https://ollama.ai) with at least one model pulled (e.g., `ollama run llama3` once).
   - Ray runtime (installed automatically via pip requirements).

2. **Create a virtual environment and install dependencies**
   ```bash
   cd /Users/fetullakh/Desktop/CPP/COMP4651/Project
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Verify Ollama is running**
   ```bash
   ollama serve # separate terminal or background service
   ```

4. **Ingest the sample corpus (pick a pipeline)**
   ```bash
   # Basic sequential RAG
   python -m rag_mapreduce.cli ingest --config configs/basic_sequential.yaml

   # Parallel MapReduce RAG
   python -m rag_mapreduce.cli ingest --config configs/mapreduce_parallel.yaml
   ```
   Point `dataset_path` at either a single file or a folder containing many `.txt`, `.md`, `.json`, `.csv`, or `.pdf` files—the loader recurses through nested directories automatically.

5. **Ask a question**
   ```bash
   python -m rag_mapreduce.cli query "How does Ray accelerate RAG?"
   ```

## Configuration

All pipeline knobs live in `configs/default.yaml`:

- `pipeline_type`: `sequential` or `mapreduce`.
- `dataset_path`: file or directory with `.txt`, `.md`, `.json`, `.csv`, or `.pdf` files (folders are walked recursively, so you can ingest multiple documents at once).
- `chunk`: size/overlap/separators for the LangChain text splitter.
- `embedding`: embedding provider + model (default: `sentence-transformers/all-MiniLM-L6-v2`).
- `vector_store`: FAISS persistence directory and distance metric.
- `mapreduce`: Ray worker count, batch size, and optional `ray_address` for remote clusters.
- `ollama`: model name, base URL, and temperature for the local LLM.
- `retrieval`: top-k neighbors when answering questions.

Create multiple config files to represent the 100 MB / 1 GB / 10 GB scenarios, point `dataset_path` to the appropriate blob (local mount or network drive), and switch `pipeline_type` per test.

## CLI Reference

```bash
python -m rag_mapreduce.cli [COMMAND] [OPTIONS]
```

| Command | Description |
| --- | --- |
| `ingest` | Load the dataset, build the FAISS index using `pipeline_type`, and persist artifacts under `artifacts/vectorstores/<index_name>` |
| `query` | Run RetrievalQA with Ollama against the latest index (no ingestion) |
| `benchmark CONFIG...` | Run ingestion benchmarks for each provided config file and print a comparison table |

Examples:

```bash
# Build + query basic RAG
python -m rag_mapreduce.cli ingest --config configs/basic_sequential.yaml
python -m rag_mapreduce.cli query "Summarize LangChain" --config configs/basic_sequential.yaml

# Build + query MapReduce RAG
python -m rag_mapreduce.cli ingest --config configs/mapreduce_parallel.yaml
python -m rag_mapreduce.cli query "Summarize LangChain" --config configs/mapreduce_parallel.yaml

# Benchmark both configs in one run
python -m rag_mapreduce.cli benchmark configs/basic_sequential.yaml configs/mapreduce_parallel.yaml
```

## Comparing Basic vs MapReduce RAG

| Aspect | Basic Sequential | Parallel MapReduce |
| --- | --- | --- |
| Config file | `configs/basic_sequential.yaml` | `configs/mapreduce_parallel.yaml` |
| Pipeline module | `SequentialPipeline` | `MapReducePipeline` |
| Execution | Single Python process | Ray tasks distributing chunk + embed work |
| Best for | <1 GB datasets, quick sanity checks | 1 GB–10 GB+ datasets, throughput experiments |

Use the same dataset path and index new names (e.g., `classroom-basic` vs `classroom-mapreduce`) so artifacts do not clash. When benchmarking, ingest both configs and compare the reported chunk counts + timings, or call `benchmark` to emit a table ready for plotting.

## Benchmarking Workloads

1. Prepare three config files (e.g., `configs/100mb.yaml`, `configs/1gb.yaml`, `configs/10gb.yaml`).
2. Duplicate settings but adjust `dataset_path`, `index_name`, and `pipeline_type`.
3. Run the sequential ingest per dataset, then mapreduce ingest. Record timings from CLI output or with `benchmark` command.
4. Use `results_to_frame` (see `src/rag_mapreduce/evaluation.py`) inside a notebook to build plots comparing throughput.

## Architecture Overview

```
Typer CLI → Pipeline (sequential | mapreduce) → FAISS Vector Store → LangChain Retriever → Ollama LLM
```

- `data_loader.py`: loads plaintext/JSON/CSV corpora.
- `chunking.py`: config-driven `RecursiveCharacterTextSplitter`.
- `embeddings.py`: HuggingFace sentence-transformers backend.
- `vector_store.py`: FAISS persistence helpers.
- `pipelines/sequential.py`: baseline ingestion path.
- `pipelines/mapreduce.py`: Ray remote tasks for chunking + embedding with reduce merge.
- `rag_chain.py`: constructs RetrievalQA chain with custom prompt.
- `evaluation.py`: benchmarking helpers returning tidy data frames.

## Testing

Run unit tests (stubs included under `tests/`).

```bash
pytest
```

## Next Steps

- Connect to a Ray cluster on AWS/GCP by setting `mapreduce.ray_address`.
- Swap embeddings (e.g., `text-embedding-3-small`) or add hybrid search.
- Add latency measurements for query-time comparisons.
- Implement custom RAG with MapReduce in retrieval stage.
- Wire Grafana or Ray Dashboard for live instrumentation during ingestion.
