"""Walk-forward validation windows for model experiments."""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
import polars as pl

from thesis.shared.config import Config

logger = logging.getLogger("thesis")


@dataclass(frozen=True)
class WalkForwardWindow:
    """One walk-forward fold."""

    train_start_idx: int
    train_end_idx: int
    test_start_idx: int
    test_end_idx: int

    @property
    def train_len(self) -> int:
        """Bars available for fitting (used by min_train_bars gate)."""
        return self.train_end_idx - self.train_start_idx

    @property
    def test_len(self) -> int:
        """Bars reserved for out-of-sample evaluation."""
        return self.test_end_idx - self.test_start_idx


def _apply_purge_embargo(
    train_start: int,
    raw_train_end: int,
    test_start: int,
    test_end: int,
    purge_bars: int,
    embargo_bars: int,
) -> WalkForwardWindow | None:
    adj_train_end = raw_train_end - purge_bars
    adj_test_start = test_start + purge_bars + embargo_bars

    if adj_train_end <= train_start or adj_test_start >= test_end:
        return None

    return WalkForwardWindow(train_start, adj_train_end, adj_test_start, test_end)


def _apply_event_purge(
    train_start: int,
    raw_train_end: int,
    test_start: int,
    test_end: int,
    event_end: np.ndarray,
    embargo_bars: int,
) -> WalkForwardWindow | None:
    if raw_train_end <= train_start:
        return None
    if len(event_end) < raw_train_end:
        raise ValueError(
            f"event_end length ({len(event_end)}) < raw_train_end ({raw_train_end})"
        )

    # Event-based purge is preferred because triple-barrier labels may extend
    # beyond a fixed horizon.
    train_events = event_end[train_start:raw_train_end]
    safe = np.flatnonzero(train_events < test_start)
    if safe.size == 0:
        return None

    adj_train_end = train_start + int(safe[-1]) + 1
    adj_test_start = test_start + embargo_bars
    if adj_test_start >= test_end:
        return None

    return WalkForwardWindow(train_start, adj_train_end, adj_test_start, test_end)


def generate_windows(
    total_bars: int,
    train_window_bars: int = 6240,
    test_window_bars: int = 1040,
    step_bars: int = 1040,
    purge_bars: int = 48,
    embargo_bars: int = 50,
    min_train_bars: int = 2000,
    event_end: np.ndarray | None = None,
) -> list[WalkForwardWindow]:
    """Build bar-count walk-forward windows."""
    windows: list[WalkForwardWindow] = []
    test_start = 0

    while test_start < total_bars:
        test_end = min(test_start + test_window_bars, total_bars)
        raw_train_end = test_start
        train_start = max(0, raw_train_end - train_window_bars)

        if event_end is None:
            window = _apply_purge_embargo(
                train_start,
                raw_train_end,
                test_start,
                test_end,
                purge_bars,
                embargo_bars,
            )
        else:
            window = _apply_event_purge(
                train_start,
                raw_train_end,
                test_start,
                test_end,
                event_end,
                embargo_bars,
            )

        if window is not None and window.train_len >= min_train_bars:
            windows.append(window)

        test_start += step_bars

    return windows


def build_walk_forward_windows(
    df: pl.DataFrame,
    config: Config,
) -> list[WalkForwardWindow]:
    """Build configured walk-forward windows from model dataset."""
    event_end = df["event_end"].to_numpy() if "event_end" in df.columns else None
    return generate_windows(
        total_bars=len(df),
        train_window_bars=config.validation.train_window_bars,
        test_window_bars=config.validation.test_window_bars,
        step_bars=config.validation.step_bars,
        purge_bars=config.validation.purge_bars,
        embargo_bars=config.validation.embargo_bars,
        min_train_bars=config.validation.min_train_bars,
        event_end=event_end,
    )


def log_windows(
    windows: list[WalkForwardWindow],
    df: pl.DataFrame,
    ts_col: str = "timestamp",
) -> None:
    """Log walk-forward date ranges."""
    if ts_col not in df.columns:
        return

    ts = df[ts_col]
    for i, w in enumerate(windows):
        logger.info(
            "Window %d | train [%d:%d] %s->%s (%d bars) | "
            "test [%d:%d] %s->%s (%d bars)",
            i + 1,
            w.train_start_idx,
            w.train_end_idx,
            ts[w.train_start_idx],
            ts[min(w.train_end_idx - 1, len(ts) - 1)],
            w.train_len,
            w.test_start_idx,
            w.test_end_idx,
            ts[w.test_start_idx],
            ts[min(w.test_end_idx - 1, len(ts) - 1)],
            w.test_len,
        )
