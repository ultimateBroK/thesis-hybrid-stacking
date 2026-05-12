"""Shared walk-forward orchestration loop.

Provides a generic ``run_walk_forward`` that both LightGBM and stacking
trainers delegate to, eliminating the duplicated prepare → loop → save pattern.
"""

from __future__ import annotations

from collections.abc import Callable
import logging
import time
from typing import Any

from thesis.shared.config import Config
from thesis.shared.ui import console

logger = logging.getLogger("thesis.pipeline")


def run_walk_forward(
    config: Config,
    *,
    prepare_fn: Callable,
    window_fn: Callable,
    save_fn: Callable,
) -> None:
    """Execute the generic walk-forward loop.

    Args:
        config: Application configuration.
        prepare_fn: ``(config) -> (df, windows, feature_cols, extra_data)``
            Loads data and generates windows.  *extra_data* is unpacked
            as keyword arguments into each *window_fn* call.
        window_fn: ``(config, w_idx, window, df, feature_cols, **extra_data)
            -> dict | None`` — trains and predicts for one window.
            Returns ``None`` to skip.
        save_fn: ``(config, results, windows, stage_start) -> None``
            Persists all accumulated results.
    """
    df, windows, feature_cols, extra_data = prepare_fn(config)
    stage_start = time.perf_counter()

    results: list[dict[str, Any]] = []
    for w_idx, window in enumerate(windows):
        window_start = time.perf_counter()
        console.rule(f"[bold cyan]Window {w_idx + 1}/{len(windows)}[/]", style="cyan")
        logger.info(
            "=== Window %d/%d: train=[%d:%d] test=[%d:%d] ===",
            w_idx + 1,
            len(windows),
            window.train_start_idx,
            window.train_end_idx,
            window.test_start_idx,
            window.test_end_idx,
        )
        result = window_fn(config, w_idx, window, df, feature_cols, **extra_data)
        if result is None:
            continue
        results.append(result)
        logger.info(
            "Window %d done (%.1fs)",
            w_idx + 1,
            time.perf_counter() - window_start,
        )

    save_fn(config, results, windows, stage_start)
