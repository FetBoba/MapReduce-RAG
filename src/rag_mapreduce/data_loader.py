"""Utilities to read raw documents from disk."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, List

from langchain_core.documents import Document

SUPPORTED_SUFFIXES = {".txt", ".md", ".json", ".csv"}


def _load_text(path: Path) -> Document:
    content = path.read_text(encoding="utf-8")
    return Document(page_content=content, metadata={"source": str(path)})


def _load_json(path: Path) -> List[Document]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    docs: List[Document] = []
    if isinstance(payload, list):
        for idx, item in enumerate(payload):
            docs.append(
                Document(
                    page_content=json.dumps(item, ensure_ascii=False),
                    metadata={"source": str(path), "row": idx},
                )
            )
    else:
        docs.append(Document(page_content=json.dumps(payload), metadata={"source": str(path)}))
    return docs


def _load_csv(path: Path) -> List[Document]:
    docs: List[Document] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            docs.append(
                Document(
                    page_content=json.dumps(row, ensure_ascii=False),
                    metadata={"source": str(path), "row": idx},
                )
            )
    return docs


def _walk_files(dataset_path: Path) -> Iterable[Path]:
    if dataset_path.is_file():
        yield dataset_path
        return
    for file_path in dataset_path.rglob("*"):
        if file_path.is_file():
            yield file_path


def load_documents(dataset_path: Path | str) -> List[Document]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path not found: {path}")

    documents: List[Document] = []
    for file_path in _walk_files(path):
        if file_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if file_path.suffix.lower() in {".txt", ".md"}:
            documents.append(_load_text(file_path))
        elif file_path.suffix.lower() == ".json":
            documents.extend(_load_json(file_path))
        elif file_path.suffix.lower() == ".csv":
            documents.extend(_load_csv(file_path))
    if not documents:
        raise ValueError(
            f"No supported documents were found under {path}. Supported suffixes: {sorted(SUPPORTED_SUFFIXES)}"
        )
    return documents
