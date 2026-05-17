"""Walk-forward training: windows, feature pipeline, targets, loop, dispatch."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import logging
import time
from typing import Any

from feature_engine.selection import DropCorrelatedFeatures, DropDuplicateFeatures
import numpy as np
import polars as pl
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from thesis.shared.config import Config
from thesis.shared.constants import CENSORED_LABEL, REGIME_FEATURES
from thesis.shared.utils import console

logger = logging.getLogger("thesis")


# ---------------------------------------------------------------------------
# Walk-forward windows (validation.py)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Feature pipeline (feature_pipeline.py)
# ---------------------------------------------------------------------------


def select_static_cols(
    config: Config,
    df: pl.DataFrame,
    candidates: list[str],
) -> list[str]:
    """Choose static feature columns.

    Prefer config list. Add regime features when enabled.
    """
    available = [c for c in config.features.static_feature_cols if c in df.columns]
    if not available:
        available = [c for c in candidates if c in df.columns]

    if getattr(config.features, "enable_regime_features", False):
        for c in REGIME_FEATURES:
            if c in df.columns and c not in available:
                available.append(c)

    return available


def _add_label_prior_features(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Add leakage-safe label priors."""
    if "label" not in df.columns:
        return df

    h = config.labels.horizon_bars
    shift_n = h + 1

    is_long = (pl.col("label") == 1).cast(pl.Float64)
    is_short = (pl.col("label") == -1).cast(pl.Float64)

    return df.with_columns(
        [
            is_long.shift(shift_n)
            .rolling_mean(100)
            .fill_null(0.0)
            .alias("label_prior_long_lag1"),
            is_short.shift(shift_n)
            .rolling_mean(100)
            .fill_null(0.0)
            .alias("label_prior_short_lag1"),
        ]
    )


def fit_static_feature_pipeline(
    config: Config,
    train_df: pl.DataFrame,
    cols: list[str],
    y_train: np.ndarray,
) -> tuple[Pipeline, list[str]]:
    """Fit feature filter pipeline.

    Steps:
        1. DropDuplicateFeatures
        2. DropCorrelatedFeatures
        3. RobustScaler
        4. SelectKBest

    Fallback keeps scaled raw columns.
    """
    X = train_df.select(cols).to_pandas()
    X.columns = cols

    k_best = min(max(5, len(cols) // 2), len(cols))
    logger.info("  Feature pipeline: %d cols → k_best=%d", len(cols), k_best)

    pipe = Pipeline(
        [
            ("dedup", DropDuplicateFeatures(missing_values="ignore")),
            (
                "decorr",
                DropCorrelatedFeatures(
                    threshold=config.features.correlation_threshold,
                    method="pearson",
                ),
            ),
            ("scaler", RobustScaler()),
            ("select", SelectKBest(score_func=f_classif, k=k_best)),
        ]
    )

    try:
        pipe.fit(X, y_train)

        pre_select = pipe[:-1].get_feature_names_out()
        pre_cols = [str(c) for c in pre_select]
        mask = pipe.named_steps["select"].get_support()
        selected = [c for c, m in zip(pre_cols, mask, strict=False) if m]

        if not selected:
            selected = list(pre_cols[: min(5, len(pre_cols))])

        logger.info(
            "  Selected %d/%d features: %s", len(selected), len(pre_cols), selected
        )
        return pipe, selected

    except ValueError as exc:
        logger.warning("  Feature selection fallback (all cols scaled): %s", exc)
        fallback = Pipeline([("scaler", RobustScaler())])
        fallback.fit(X[cols], y_train)
        return fallback, list(cols)


# ---------------------------------------------------------------------------
# Targets (targets.py)
# ---------------------------------------------------------------------------


def compute_regression_target(
    df: pl.DataFrame, config: Config
) -> tuple[pl.DataFrame, bool]:
    """Add forward-return target for regression.

    Tail rows cannot see horizon; mark censored then drop target-null rows.
    """
    is_regression = config.model.objective == "regression"
    if not is_regression:
        return df, False

    if "close" not in df.columns:
        raise ValueError("Regression objective requires 'close' column in labeled data")

    h = config.labels.horizon_bars
    close = df["close"].to_numpy()
    n = len(close)

    reg = np.full(n, np.nan, dtype=np.float64)
    reg[: n - h] = (close[h:] - close[: n - h]) / close[: n - h]

    label_arr = df["label"].to_numpy().copy()
    tail_start = max(0, n - h)
    label_arr[tail_start:] = CENSORED_LABEL

    df = df.with_columns(
        [
            pl.Series("regression_target", reg),
            pl.Series("label", label_arr),
        ]
    )

    n_before = len(df)
    df = df.filter(pl.col("regression_target").is_not_nan())
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.info(
            "  Dropped %d regression tail rows (insufficient forward horizon)",
            n_dropped,
        )

    logger.info(
        "  Regression target: horizon=%d bars, mean=%.6f, std=%.6f",
        h,
        float(np.nanmean(reg)),
        float(np.nanstd(reg)),
    )
    return df, True


# ---------------------------------------------------------------------------
# Walk-forward loop (loop.py)
# ---------------------------------------------------------------------------


def run_walk_forward(
    config: Config,
    *,
    prepare_fn: Callable[[Config], tuple[Any, list[Any], list[str], dict[str, Any]]],
    window_fn: Callable[..., dict[str, Any] | None],
    save_fn: Callable[[Config, list[dict[str, Any]], list[Any], float], None],
) -> None:
    """Run walk-forward hooks.

    Args:
        config: Pipeline config.
        prepare_fn: Load data, windows, feature columns, extras.
        window_fn: Train one window, return result or skip.
        save_fn: Persist results after loop.
    """
    t0 = time.perf_counter()
    df, windows, feature_cols, extra_data = prepare_fn(config)
    logger.info(
        "Walk-forward: %d windows, %d features", len(windows), len(feature_cols)
    )

    results: list[dict[str, Any]] = []
    for w_idx, window in enumerate(windows):
        wt = time.perf_counter()
        console.rule(f"[bold cyan]Window {w_idx + 1}/{len(windows)}[/]")
        logger.info(
            "  [%d:%d] train | [%d:%d] test",
            window.train_start_idx,
            window.train_end_idx,
            window.test_start_idx,
            window.test_end_idx,
        )
        result = window_fn(config, w_idx, window, df, feature_cols, **extra_data)
        if result is not None:
            results.append(result)
        logger.info("  Window %d done (%.1fs)", w_idx + 1, time.perf_counter() - wt)

    logger.info(
        "Walk-forward: %d/%d windows produced results (%.1fs total)",
        len(results),
        len(windows),
        time.perf_counter() - t0,
    )
    save_fn(config, results, windows, time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# Dispatcher (dispatcher.py) — public entrypoint
# ---------------------------------------------------------------------------


def train_walk_forward(config: Config) -> None:
    """Run Hybrid Stacking walk-forward training."""
    from thesis.models.stacking import train_stacking_walk_forward

    logger.info("Architecture: hybrid_stacking (fixed)")
    train_stacking_walk_forward(config)
