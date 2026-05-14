"""Walk-forward validation with purge and embargo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class WalkForwardWindow:
    """Train/test slice for one walk-forward fold.

    Attributes:
        train_start_idx: Inclusive start of training period.
        train_end_idx: Exclusive end of training period (after purge).
        test_start_idx: Inclusive start of test period (after embargo).
        test_end_idx: Exclusive end of test period.
    """

    train_start_idx: int
    train_end_idx: int
    test_start_idx: int
    test_end_idx: int

    @property
    def train_len(self) -> int:
        """Number of training bars."""
        return self.train_end_idx - self.train_start_idx

    @property
    def test_len(self) -> int:
        """Number of test bars."""
        return self.test_end_idx - self.test_start_idx


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
    """Create bar-count walk-forward windows across total_bars.

    Windows slide forward by step_bars. Purge trims training tail;
    embargo skips test head to prevent leakage.

    Args:
        total_bars: Total rows in dataset.
        train_window_bars: Desired training length in bars.
        test_window_bars: Desired test length in bars.
        step_bars: Bars between successive windows.
        purge_bars: Bars removed from training tail.
        embargo_bars: Bars skipped at test head after purge.
        min_train_bars: Minimum training bars required.
        event_end: Event-end indices for label-aware purging.

    Returns:
        List of WalkForwardWindow objects.
    """
    windows: list[WalkForwardWindow] = []
    test_start = 0

    while test_start < total_bars:
        test_end = min(test_start + test_window_bars, total_bars)
        raw_train_end = test_start
        train_start = max(0, raw_train_end - train_window_bars)

        if event_end is None:
            window = _apply_purge_embargo(
                train_start=train_start,
                raw_train_end=raw_train_end,
                test_start=test_start,
                test_end=test_end,
                purge_bars=purge_bars,
                embargo_bars=embargo_bars,
            )
        else:
            window = _apply_event_purge(
                train_start=train_start,
                raw_train_end=raw_train_end,
                test_start=test_start,
                test_end=test_end,
                event_end=event_end,
                embargo_bars=embargo_bars,
            )

        if window is not None and window.train_len >= min_train_bars:
            windows.append(window)

        test_start += step_bars

    return windows


def _apply_purge_embargo(
    train_start: int,
    raw_train_end: int,
    test_start: int,
    test_end: int,
    purge_bars: int,
    embargo_bars: int,
) -> WalkForwardWindow | None:
    """Adjust indices using fixed-bar purge and embargo.

    Gap between adjusted train end and adjusted test start =
    2*purge_bars + embargo_bars. Extra purge_bar on test side
    accounts for label lookahead.
    """
    adj_train_end = raw_train_end - purge_bars
    adj_test_start = test_start + purge_bars + embargo_bars

    if adj_train_end <= train_start:
        return None
    if adj_test_start >= test_end:
        return None

    return WalkForwardWindow(
        train_start_idx=train_start,
        train_end_idx=adj_train_end,
        test_start_idx=adj_test_start,
        test_end_idx=test_end,
    )


def _apply_event_purge(
    train_start: int,
    raw_train_end: int,
    test_start: int,
    test_end: int,
    event_end: np.ndarray,
    embargo_bars: int,
) -> WalkForwardWindow | None:
    """Adjust window using event-end times instead of fixed purge.

    Training samples kept only when event ends strictly before
    test boundary. Embargo still skips embargo_bars at test head.
    """
    if raw_train_end <= train_start:
        return None
    if len(event_end) < raw_train_end:
        raise ValueError(
            f"event_end length ({len(event_end)}) < raw_train_end ({raw_train_end})"
        )

    train_events = event_end[train_start:raw_train_end]
    safe = np.flatnonzero(train_events < test_start)
    if safe.size == 0:
        return None

    adj_train_end = train_start + int(safe[-1]) + 1
    adj_test_start = test_start + embargo_bars

    if adj_test_start >= test_end:
        return None

    return WalkForwardWindow(
        train_start_idx=train_start,
        train_end_idx=adj_train_end,
        test_start_idx=adj_test_start,
        test_end_idx=test_end,
    )


def log_windows(
    windows: list[WalkForwardWindow],
    df: pl.DataFrame,
    ts_col: str = "timestamp",
) -> None:
    """Log date ranges for every window."""
    import logging

    logger = logging.getLogger("thesis")
    if ts_col not in df.columns:
        return

    ts = df[ts_col]
    for i, w in enumerate(windows):
        t0 = ts[w.train_start_idx]
        t1 = ts[min(w.train_end_idx - 1, len(ts) - 1)]
        t2 = ts[w.test_start_idx]
        t3 = ts[min(w.test_end_idx - 1, len(ts) - 1)]
        logger.info(
            "Window %d | train [%d:%d] %s→%s (%d bars) | test [%d:%d] %s→%s (%d bars)",
            i + 1,
            w.train_start_idx,
            w.train_end_idx,
            t0,
            t1,
            w.train_len,
            w.test_start_idx,
            w.test_end_idx,
            t2,
            t3,
            w.test_len,
        )
