"""Tests for validation module — walk-forward sliding window."""

import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from thesis.validation import (
    WalkForwardWindow,
    apply_purge_embargo,
    generate_windows,
    log_windows,
    split_data,
)


def _make_df(n: int = 500) -> pl.DataFrame:
    """Create a DataFrame with n hourly rows starting 2020-01-01."""
    return pl.DataFrame({
        "timestamp": pl.datetime_range(
            start=pl.datetime(2020, 1, 1),
            end=pl.datetime(2020, 1, 1) + pl.duration(hours=n - 1),
            interval="1h",
            eager=True,
        ),
        "value": list(range(n)),
    })


class TestWalkForwardWindow:
    def test_window_creation(self):
        w = WalkForwardWindow(
            train_start_idx=0,
            train_end_idx=100,
            test_start_idx=100,
            test_end_idx=120,
        )
        assert w.train_start_idx == 0
        assert w.train_end_idx == 100
        assert w.test_start_idx == 100
        assert w.test_end_idx == 120

    def test_window_is_frozen(self):
        w = WalkForwardWindow(0, 100, 100, 120)
        with pytest.raises(AttributeError):
            w.train_start_idx = 50


class TestApplyPurgeEmbargo:
    def test_basic_purge(self):
        result = apply_purge_embargo(
            train_start=0,
            raw_train_end=1000,
            test_start=1000,
            test_end=1200,
            purge_bars=25,
            embargo_bars=50,
        )
        assert result is not None
        assert result.train_end_idx == 975  # 1000 - 25
        assert result.test_start_idx == 1075  # 1000 + 25 + 50

    def test_no_purge(self):
        result = apply_purge_embargo(
            train_start=0,
            raw_train_end=1000,
            test_start=1000,
            test_end=1200,
            purge_bars=0,
            embargo_bars=0,
        )
        assert result is not None
        assert result.train_end_idx == 1000
        assert result.test_start_idx == 1000

    def test_returns_none_if_train_too_small(self):
        result = apply_purge_embargo(
            train_start=0,
            raw_train_end=10,
            test_start=10,
            test_end=100,
            purge_bars=25,
            embargo_bars=50,
        )
        assert result is None

    def test_returns_none_if_test_too_small(self):
        result = apply_purge_embargo(
            train_start=0,
            raw_train_end=1000,
            test_start=1000,
            test_end=1020,
            purge_bars=25,
            embargo_bars=50,
        )
        assert result is None


class TestGenerateWindows:
    def test_generates_windows(self):
        windows = generate_windows(
            total_bars=10000,
            train_window_bars=3000,
            test_window_bars=1000,
            step_bars=1000,
            purge_bars=10,
            embargo_bars=5,
            min_train_bars=500,
        )
        assert len(windows) >= 1

    def test_window_train_starts_at_zero(self):
        windows = generate_windows(
            total_bars=10000,
            train_window_bars=3000,
            test_window_bars=1000,
            step_bars=1000,
            purge_bars=10,
            embargo_bars=5,
            min_train_bars=500,
        )
        assert windows[0].train_start_idx == 0

    def test_windows_step_forward(self):
        windows = generate_windows(
            total_bars=10000,
            train_window_bars=3000,
            test_window_bars=1000,
            step_bars=1000,
            purge_bars=10,
            embargo_bars=5,
            min_train_bars=500,
        )
        if len(windows) >= 2:
            assert windows[1].test_start_idx > windows[0].test_start_idx

    def test_no_windows_if_too_small(self):
        windows = generate_windows(
            total_bars=100,
            train_window_bars=5000,
            test_window_bars=1000,
            step_bars=1000,
            purge_bars=10,
            embargo_bars=5,
            min_train_bars=1000,
        )
        assert len(windows) == 0

    def test_purge_embargo_adjusts_indices(self):
        windows = generate_windows(
            total_bars=10000,
            train_window_bars=3000,
            test_window_bars=1000,
            step_bars=1000,
            purge_bars=25,
            embargo_bars=50,
            min_train_bars=500,
        )
        assert len(windows) >= 1
        w = windows[0]
        # train_end should be less than the raw boundary
        assert w.train_end_idx < w.test_start_idx
        # test_start should be adjusted by purge+embargo
        assert w.test_start_idx - w.train_end_idx >= 25 + 50

    def test_windows_dont_exceed_total(self):
        windows = generate_windows(
            total_bars=5500,
            train_window_bars=3000,
            test_window_bars=1000,
            step_bars=1000,
            purge_bars=10,
            embargo_bars=5,
            min_train_bars=500,
        )
        for w in windows:
            assert w.test_end_idx <= 5500


class TestSplitData:
    def test_basic_split(self):
        df = _make_df(500)
        windows = [WalkForwardWindow(0, 300, 300, 400)]
        splits = split_data(df, windows, "timestamp")
        assert len(splits) == 1
        train_df, test_df = splits[0]
        assert len(train_df) == 300
        assert len(test_df) == 100

    def test_multiple_splits(self):
        df = _make_df(1000)
        windows = [
            WalkForwardWindow(0, 400, 400, 500),
            WalkForwardWindow(100, 500, 500, 600),
        ]
        splits = split_data(df, windows, "timestamp")
        assert len(splits) == 2

    def test_empty_windows(self):
        df = _make_df(500)
        splits = split_data(df, [], "timestamp")
        assert len(splits) == 0


class TestConsecutiveWindowsNoOverlap:
    """Verify walk-forward test windows are disjoint across folds."""

    def test_consecutive_test_windows_do_not_overlap(self):
        """No index can appear in two different test windows."""
        windows = generate_windows(
            total_bars=20000,
            train_window_bars=8000,
            test_window_bars=2000,
            step_bars=2000,  # step == test_window → non-overlapping
            purge_bars=10,
            embargo_bars=5,
            min_train_bars=2000,
        )
        assert len(windows) >= 2, "Need at least 2 windows to test overlap"

        for i in range(len(windows) - 1):
            w_a = windows[i]
            w_b = windows[i + 1]
            # test range of window i must end before test range of window i+1
            assert w_a.test_end_idx <= w_b.test_start_idx, (
                f"Window {i} test [{w_a.test_start_idx}, {w_a.test_end_idx}) "
                f"overlaps window {i + 1} test [{w_b.test_start_idx}, {w_b.test_end_idx})"
            )

    def test_all_test_indices_are_disjoint_sets(self):
        """Collect all test indices and verify zero duplicates."""
        windows = generate_windows(
            total_bars=15000,
            train_window_bars=5000,
            test_window_bars=1500,
            step_bars=1500,
            purge_bars=5,
            embargo_bars=3,
            min_train_bars=1000,
        )
        all_test_indices: set[int] = set()
        for w in windows:
            for idx in range(w.test_start_idx, w.test_end_idx):
                assert idx not in all_test_indices, (
                    f"Index {idx} appears in multiple test windows"
                )
                all_test_indices.add(idx)


class TestOOFUniquenessGuard:
    """Verify duplicate-timestamp detection logic for OOF predictions."""

    def test_oof_timestamps_unique_after_concat(self):
        """OOF predictions must have unique timestamps — no double-counting."""
        # Simulate two fold predictions with one overlapping timestamp
        fold1 = pl.DataFrame({
            "timestamp": [1, 2, 3],
            "pred_label": [1, -1, 0],
        })
        fold2 = pl.DataFrame({
            "timestamp": [3, 4, 5],  # timestamp 3 overlaps
            "pred_label": [1, -1, 0],
        })
        oof_df = pl.concat([fold1, fold2])

        ts_col = oof_df["timestamp"]
        assert ts_col.n_unique() < len(ts_col), "Expected duplicates"
        dup_count = len(ts_col) - ts_col.n_unique()
        assert dup_count == 1

    def test_oof_no_duplicates_passes(self):
        """Non-overlapping folds produce unique timestamps."""
        fold1 = pl.DataFrame({
            "timestamp": [1, 2, 3],
            "pred_label": [1, -1, 0],
        })
        fold2 = pl.DataFrame({
            "timestamp": [4, 5, 6],
            "pred_label": [1, -1, 0],
        })
        oof_df = pl.concat([fold1, fold2])

        ts_col = oof_df["timestamp"]
        assert ts_col.n_unique() == len(ts_col)


class TestLogWindows:
    def test_log_runs_without_error(self):
        df = _make_df(500)
        windows = [WalkForwardWindow(0, 300, 300, 400)]
        log_windows(windows, df, "timestamp")

    def test_log_missing_column(self):
        df = _make_df(500)
        windows = [WalkForwardWindow(0, 300, 300, 400)]
        # Should log warning but not crash
        log_windows(windows, df, "nonexistent")
