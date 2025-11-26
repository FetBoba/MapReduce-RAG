"""Ollama helper utilities."""

from __future__ import annotations

from langchain_community.llms import Ollama
from langchain_core.language_models import BaseLanguageModel

from .config import OllamaConfig


def build_llm(config: OllamaConfig) -> BaseLanguageModel:
    params = {
        "model": config.model,
        "temperature": config.temperature,
    }
    if config.base_url:
        params["base_url"] = config.base_url
    return Ollama(**params)
