"""Thesis ML pipeline — top-level public API surface.

This module re-exports the most commonly used symbols so that callers can use
``import thesis; thesis.run_pipeline(...)`` without reaching into sub-packages.
"""

from thesis.pipeline import run_pipeline
from thesis.shared.config import Config, get_config, load_config

__all__ = [
    "Config",
    "get_config",
    "load_config",
    "run_pipeline",
]
