"""Walk-forward windows. Purge leakage, embargo spillover."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class WalkForwardWindow:
    """One walk-forward fold.

    Attributes:
        train_start_idx: Train start, inclusive.
        train_end_idx: Train end, exclusive, after purge.
        test_start_idx: Test start, inclusive, after embargo.
        test_end_idx: Test end, exclusive.
    """

    train_start_idx: int
    train_end_idx: int
    test_start_idx: int
    test_end_idx: int

    @property
    def train_len(self) -> int:
        """Training bar count."""
        return self.train_end_idx - self.train_start_idx

    @property
    def test_len(self) -> int:
        """Test bar count."""
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
    """Build bar-count walk-forward windows.

    Purge removes label overlap. Embargo blocks spillover.

    Args:
        total_bars: Dataset rows.
        train_window_bars: Target train bars.
        test_window_bars: Target test bars.
        step_bars: Window stride.
        purge_bars: Train-tail gap.
        embargo_bars: Test-head gap.
        min_train_bars: Minimum usable train bars.
        event_end: Label event-end indices.

    Returns:
        Valid windows.
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
    """Apply fixed purge/embargo gaps.

    Extra test-side purge covers label lookahead.
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
    """Apply event-end purge.

    Keep train rows only if event ends before test.
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
    """Log window date ranges."""
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
