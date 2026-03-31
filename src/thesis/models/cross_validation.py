"""Time-series cross-validation with purge and embargo for financial data.

This module implements Walk-Forward Cross-Validation suitable for time series
where temporal order must be preserved to prevent data leakage.
"""

import logging
from typing import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import polars as pl

logger = logging.getLogger("thesis.models")


@dataclass
class WalkForwardWindow:
    """A single walk-forward window configuration."""

    train_start: datetime
    train_end: datetime
    val_start: datetime
    val_end: datetime
    window_name: str


class SlidingWindowCV:
    """Sliding window cross-validation for time series.

    Uses fixed-size training windows that slide forward in time,
    validating on the subsequent period.

    Example (2-year train, 1-year val, 1-year step):
        Window 1: Train 2018-2019 | Val 2020
        Window 2: Train 2019-2020 | Val 2021
        Window 3: Train 2020-2021 | Val 2022
        Window 4: Train 2021-2022 | Val 2023
    """

    def __init__(
        self,
        train_years: int = 2,
        val_years: int = 1,
        step_years: int = 1,
        purge_bars: int = 15,
        embargo_bars: int = 10,
        bar_frequency: str = "1h",
    ):
        """Initialize sliding window CV.

        Args:
            train_years: Size of training window in years
            val_years: Size of validation window in years
            step_years: Step size between windows in years
            purge_bars: Bars to remove between train and val to prevent leakage
            embargo_bars: Bars to embargo after validation
            bar_frequency: Frequency of bars ("1h", "1d", etc.)
        """
        self.train_years = train_years
        self.val_years = val_years
        self.step_years = step_years
        self.purge_bars = purge_bars
        self.embargo_bars = embargo_bars
        self.bar_frequency = bar_frequency

    def generate_windows(
        self, overall_start: datetime, overall_end: datetime
    ) -> list[WalkForwardWindow]:
        """Generate sliding windows covering the date range.

        Args:
            overall_start: Start of entire dataset
            overall_end: End of entire dataset

        Returns:
            List of WalkForwardWindow configurations
        """
        windows = []

        # Calculate purge/embargo offsets
        purge_offset = self._bars_to_timedelta(self.purge_bars)
        embargo_offset = self._bars_to_timedelta(self.embargo_bars)

        # First training window starts at overall_start
        current_train_start = overall_start
        window_num = 1

        while True:
            # Calculate window boundaries
            train_end = current_train_start + timedelta(days=365 * self.train_years)

            # Apply purge after training
            val_start = train_end + purge_offset

            # Validation period
            val_end = val_start + timedelta(days=365 * self.val_years)

            # Check if this window fits within overall range
            # We need: val_end + embargo <= overall_end
            effective_val_end = val_end + embargo_offset

            if effective_val_end > overall_end:
                break

            # Create window (training excludes purge period at end)
            window = WalkForwardWindow(
                train_start=current_train_start,
                train_end=train_end,  # Train up to start of purge
                val_start=val_start,
                val_end=val_end,
                window_name=f"Fold_{window_num}",
            )
            windows.append(window)

            logger.info(
                f"Window {window_num}: Train {current_train_start.strftime('%Y-%m-%d')} "
                f"to {train_end.strftime('%Y-%m-%d')} | "
                f"Val {val_start.strftime('%Y-%m-%d')} to {val_end.strftime('%Y-%m-%d')}"
            )

            # Slide forward
            current_train_start = current_train_start + timedelta(
                days=365 * self.step_years
            )
            window_num += 1

        return windows

    def _bars_to_timedelta(self, bars: int) -> timedelta:
        """Convert bar count to timedelta based on frequency."""
        if self.bar_frequency == "1h":
            return timedelta(hours=bars)
        elif self.bar_frequency == "1d":
            return timedelta(days=bars)
        else:
            # Default to hours
            return timedelta(hours=bars)

    def split(
        self,
        df: pl.DataFrame,
        timestamp_col: str = "timestamp",
        overall_start: datetime = None,
        overall_end: datetime = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, str]]:
        """Generate train/val indices for each window.

        Args:
            df: DataFrame with timestamp column
            timestamp_col: Name of timestamp column
            overall_start: Optional override for start date
            overall_end: Optional override for end date

        Yields:
            Tuple of (train_indices, val_indices, window_name)
        """
        # Get date range from data if not provided
        if overall_start is None:
            overall_start = df[timestamp_col].min()
        if overall_end is None:
            overall_end = df[timestamp_col].max()

        # Ensure we have datetime objects
        if isinstance(overall_start, pl.Expr):
            overall_start = overall_start.item()
        if isinstance(overall_end, pl.Expr):
            overall_end = overall_end.item()

        # Convert polars datetime to python datetime if needed
        if hasattr(overall_start, "to_numpy"):
            overall_start = overall_start.to_numpy().item()
        if hasattr(overall_end, "to_numpy"):
            overall_end = overall_end.to_numpy().item()

        # Convert numpy datetime64 to python datetime if needed
        if hasattr(overall_start, "astype"):
            overall_start = overall_start.astype("datetime64[s]").item()
        if hasattr(overall_end, "astype"):
            overall_end = overall_end.astype("datetime64[s]").item()

        logger.info(f"Generating sliding windows from {overall_start} to {overall_end}")

        windows = self.generate_windows(overall_start, overall_end)

        timestamps = df[timestamp_col].to_numpy()

        # Handle timezone-aware timestamps by converting to naive
        if hasattr(timestamps[0], "tzinfo") and timestamps[0].tzinfo is not None:
            # Convert timezone-aware timestamps to naive (UTC)
            timestamps = np.array([t.replace(tzinfo=None) for t in timestamps])

        for window in windows:
            # Find indices for this window
            # Convert window boundaries to naive datetime if they have timezone info
            train_start = (
                window.train_start.replace(tzinfo=None)
                if window.train_start.tzinfo
                else window.train_start
            )
            train_end = (
                window.train_end.replace(tzinfo=None)
                if window.train_end.tzinfo
                else window.train_end
            )
            val_start = (
                window.val_start.replace(tzinfo=None)
                if window.val_start.tzinfo
                else window.val_start
            )
            val_end = (
                window.val_end.replace(tzinfo=None)
                if window.val_end.tzinfo
                else window.val_end
            )

            train_mask = (timestamps >= train_start) & (timestamps < train_end)
            val_mask = (timestamps >= val_start) & (timestamps < val_end)

            train_indices = np.where(train_mask)[0]
            val_indices = np.where(val_mask)[0]

            if len(train_indices) == 0 or len(val_indices) == 0:
                logger.warning(
                    f"Window {window.window_name} has empty train ({len(train_indices)}) "
                    f"or val ({len(val_indices)}) - skipping"
                )
                continue

            logger.info(
                f"Window {window.window_name}: {len(train_indices)} train, "
                f"{len(val_indices)} val samples"
            )

            yield train_indices, val_indices, window.window_name

    def get_n_splits(self, overall_start: datetime, overall_end: datetime) -> int:
        """Get number of splits that will be generated."""
        windows = self.generate_windows(overall_start, overall_end)
        return len(windows)


class ExpandingWindowCV:
    """Expanding window cross-validation for time series.

    Training window starts small and grows over time,
    validating on subsequent periods.

    Example (min 1-year train, expanding, 1-year val, 1-year step):
        Window 1: Train 2018 | Val 2019
        Window 2: Train 2018-2019 | Val 2020
        Window 3: Train 2018-2020 | Val 2021
    """

    def __init__(
        self,
        min_train_years: int = 1,
        val_years: int = 1,
        step_years: int = 1,
        purge_bars: int = 15,
        embargo_bars: int = 10,
        bar_frequency: str = "1h",
    ):
        """Initialize expanding window CV.

        Args:
            min_train_years: Minimum training period in years
            val_years: Validation period in years
            step_years: Step between validation windows
            purge_bars: Bars to purge
            embargo_bars: Bars to embargo
            bar_frequency: Bar frequency
        """
        self.min_train_years = min_train_years
        self.val_years = val_years
        self.step_years = step_years
        self.purge_bars = purge_bars
        self.embargo_bars = embargo_bars
        self.bar_frequency = bar_frequency

    def generate_windows(
        self, overall_start: datetime, overall_end: datetime
    ) -> list[WalkForwardWindow]:
        """Generate expanding windows."""
        windows = []

        purge_offset = self._bars_to_timedelta(self.purge_bars)
        embargo_offset = self._bars_to_timedelta(self.embargo_bars)

        # First validation starts after minimum training + purge
        current_train_start = overall_start
        min_train_end = current_train_start + timedelta(days=365 * self.min_train_years)

        # First validation window
        current_val_start = min_train_end + purge_offset
        window_num = 1

        while True:
            val_end = current_val_start + timedelta(days=365 * self.val_years)
            effective_val_end = val_end + embargo_offset

            if effective_val_end > overall_end:
                break

            window = WalkForwardWindow(
                train_start=current_train_start,
                train_end=current_val_start - purge_offset,  # Train up to purge
                val_start=current_val_start,
                val_end=val_end,
                window_name=f"Fold_{window_num}",
            )
            windows.append(window)

            logger.info(
                f"Expanding Window {window_num}: Train {current_train_start.strftime('%Y-%m-%d')} "
                f"to {(current_val_start - purge_offset).strftime('%Y-%m-%d')} | "
                f"Val {current_val_start.strftime('%Y-%m-%d')} to {val_end.strftime('%Y-%m-%d')}"
            )

            # Next validation window
            current_val_start = current_val_start + timedelta(
                days=365 * self.step_years
            )
            window_num += 1

        return windows

    def _bars_to_timedelta(self, bars: int) -> timedelta:
        """Convert bar count to timedelta."""
        if self.bar_frequency == "1h":
            return timedelta(hours=bars)
        elif self.bar_frequency == "1d":
            return timedelta(days=bars)
        else:
            return timedelta(hours=bars)

    def split(
        self,
        df: pl.DataFrame,
        timestamp_col: str = "timestamp",
        overall_start: datetime = None,
        overall_end: datetime = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, str]]:
        """Generate train/val indices for expanding windows."""
        if overall_start is None:
            overall_start = df[timestamp_col].min()
        if overall_end is None:
            overall_end = df[timestamp_col].max()

        # Convert to python datetime
        if hasattr(overall_start, "to_numpy"):
            overall_start = overall_start.to_numpy().item()
        if hasattr(overall_end, "to_numpy"):
            overall_end = overall_end.to_numpy().item()

        if hasattr(overall_start, "astype"):
            overall_start = overall_start.astype("datetime64[s]").item()
        if hasattr(overall_end, "astype"):
            overall_end = overall_end.astype("datetime64[s]").item()

        logger.info(
            f"Generating expanding windows from {overall_start} to {overall_end}"
        )

        windows = self.generate_windows(overall_start, overall_end)
        timestamps = df[timestamp_col].to_numpy()

        for window in windows:
            train_mask = (timestamps >= window.train_start) & (
                timestamps < window.train_end
            )
            val_mask = (timestamps >= window.val_start) & (timestamps < window.val_end)

            train_indices = np.where(train_mask)[0]
            val_indices = np.where(val_mask)[0]

            if len(train_indices) == 0 or len(val_indices) == 0:
                continue

            yield train_indices, val_indices, window.window_name


class PurgedKFold:
    """K-Fold cross-validation with purge and embargo for time series.

    Unlike standard sklearn KFold, this maintains temporal ordering
    within each fold and applies purge/embargo to prevent leakage.
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_bars: int = 15,
        embargo_bars: int = 10,
        bar_frequency: str = "1h",
    ):
        """Initialize purged K-Fold.

        Args:
            n_splits: Number of folds
            purge_bars: Bars to remove between train and test
            embargo_bars: Bars to embargo after test
            bar_frequency: Bar frequency
        """
        self.n_splits = n_splits
        self.purge_bars = purge_bars
        self.embargo_bars = embargo_bars
        self.bar_frequency = bar_frequency

    def split(
        self, df: pl.DataFrame, timestamp_col: str = "timestamp"
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate purged K-Fold splits.

        Args:
            df: DataFrame with data
            timestamp_col: Timestamp column name

        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(df)
        timestamps = df[timestamp_col].to_numpy()

        # Sort by time
        sorted_indices = np.argsort(timestamps)

        # Calculate fold sizes
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits)
        fold_sizes[: n_samples % self.n_splits] += 1

        current = 0
        for fold_idx in range(self.n_splits):
            start = current
            end = current + fold_sizes[fold_idx]

            # Test indices for this fold (temporally contiguous)
            test_indices = sorted_indices[start:end]

            # Train is everything outside test + purge + embargo
            purge_offset = self.purge_bars
            embargo_offset = self.embargo_bars

            # Remove purge before and embargo after test
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[start:end] = False

            # Remove purge region before test
            purge_start = max(0, start - purge_offset)
            train_mask[purge_start:start] = False

            # Remove embargo region after test
            embargo_end = min(n_samples, end + embargo_offset)
            train_mask[end:embargo_end] = False

            train_indices = np.where(train_mask)[0]
            train_indices = sorted_indices[train_indices]

            yield train_indices, test_indices
            current = end


def create_cv_splitter(config):
    """Factory function to create appropriate CV splitter based on config.

    Args:
        config: Configuration object with CV settings

    Returns:
        CV splitter instance
    """
    # Walk-Forward CV settings are in splitting config
    splitting_cfg = config.splitting
    tree_cfg = config.models["tree"]

    if getattr(splitting_cfg, "use_walk_forward_cv", False):
        window_type = getattr(splitting_cfg, "walk_forward_window_type", "sliding")

        if window_type == "sliding":
            return SlidingWindowCV(
                train_years=getattr(splitting_cfg, "walk_forward_train_years", 2),
                val_years=getattr(splitting_cfg, "walk_forward_val_years", 1),
                step_years=getattr(splitting_cfg, "walk_forward_step_years", 1),
                purge_bars=splitting_cfg.purge_bars,
                embargo_bars=splitting_cfg.embargo_bars,
                bar_frequency="1h",
            )
        elif window_type == "expanding":
            return ExpandingWindowCV(
                min_train_years=getattr(splitting_cfg, "walk_forward_train_years", 2),
                val_years=getattr(splitting_cfg, "walk_forward_val_years", 1),
                step_years=getattr(splitting_cfg, "walk_forward_step_years", 1),
                purge_bars=splitting_cfg.purge_bars,
                embargo_bars=splitting_cfg.embargo_bars,
                bar_frequency="1h",
            )

    # Fallback to purged K-Fold
    return PurgedKFold(
        n_splits=tree_cfg.cv_folds,
        purge_bars=tree_cfg.cv_purge,
        embargo_bars=tree_cfg.cv_embargo,
    )
