"""Shared logging helpers."""

from __future__ import annotations

import logging
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


_CONSOLE = Console(width=120)
_LOGGER_CACHE: dict[str, logging.Logger] = {}


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    if name in _LOGGER_CACHE:
        return _LOGGER_CACHE[name]

    handler = RichHandler(console=_CONSOLE, show_path=False)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False

    _LOGGER_CACHE[name] = logger
    return logger


def set_global_log_level(level: int) -> None:
    for logger in _LOGGER_CACHE.values():
        logger.setLevel(level)
