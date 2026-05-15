"""Triple-barrier labeling. +1 long / 0 hold / -1 short / -2 censored."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl

from thesis.shared.config import Config
from thesis.shared.constants import (
    ATR_HIGH_QUANTILE,
    ATR_LOW_QUANTILE,
    CENSORED_LABEL,
    LABEL_PROFITABILITY_WARN_PCT,
    ROUNDTRIP_MULT,
)
from thesis.shared.schemas import LabelsSchema
from thesis.shared.ui import console
from thesis.stage_3_labels._label_numba import (
    compute_average_uniqueness,
    compute_event_end,
)

logger = logging.getLogger("thesis.labels")


def generate_labels(config: Config) -> None:
    """Load features → triple-barrier labels → filter censored → validate → write."""
    df, atr_col = _load_features_and_ohlcv(config)
    _log_atr_stats(df, atr_col, config.labels.min_atr)

    labels, upper, lower, touched, _ambiguous = _compute_triple_barrier(
        close=df["close"].to_numpy(),
        high=df["high"].to_numpy(),
        low=df["low"].to_numpy(),
        atr=df[atr_col].to_numpy(),
        tp_mult=config.labels.atr_tp_multiplier,
        sl_mult=config.labels.atr_sl_multiplier,
        horizon=config.labels.horizon_bars,
        min_atr=config.labels.min_atr,
    )

    logger.info(
        "tp_mult=%.2f sl_mult=%.2f horizon=%d min_atr=%.6f",
        config.labels.atr_tp_multiplier,
        config.labels.atr_sl_multiplier,
        config.labels.horizon_bars,
        config.labels.min_atr,
    )

    event_end = compute_event_end(touched, config.labels.horizon_bars)
    weights = compute_average_uniqueness(event_end)

    df = _attach_label_columns(df, labels, upper, lower, touched, event_end, weights)
    _log_label_profitability(df, config)
    df = _drop_censored_and_nan(df)
    _log_distribution(df)
    _log_weight_stats(df)

    _validate_no_join_artifacts(df)
    LabelsSchema.validate(df, config=config)

    out_path = Path(config.paths.labels)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    logger.info("Labels saved: %s (%d rows)", out_path, len(df))


def _load_features_and_ohlcv(config: Config) -> tuple[pl.DataFrame, str]:
    """Load features parquet. Join OHLCV if OHLC columns missing from features."""
    features_path = Path(config.paths.features)
    ohlcv_path = Path(config.paths.ohlcv)

    if not features_path.exists():
        raise FileNotFoundError(f"Features not found: {features_path}")
    if not ohlcv_path.exists():
        raise FileNotFoundError(f"OHLCV not found: {ohlcv_path}")

    logger.info("Loading features: %s", features_path)
    with console.status(f"[cyan]Loading features[/] {features_path}"):
        df_features = pl.read_parquet(features_path)
    _check_unique_timestamps(df_features, "features")

    ohlc_cols = {"open", "high", "low", "close"}
    if ohlc_cols.issubset(set(df_features.columns)):
        logger.info("Features already contain OHLC — skipping OHLCV join")
        atr_col = f"atr_{config.features.atr_period}"
        if atr_col not in df_features.columns:
            raise ValueError(f"{atr_col} missing. Run feature engineering first.")
        return df_features, atr_col

    logger.info("Loading OHLCV: %s", ohlcv_path)
    with console.status(f"[cyan]Loading OHLCV[/] {ohlcv_path}"):
        df_ohlcv = pl.read_parquet(ohlcv_path).select(
            ["timestamp", "open", "high", "low", "close"]
        )
    _check_unique_timestamps(df_ohlcv, "OHLCV")

    df = df_features.join(df_ohlcv, on="timestamp", how="inner")
    df = _drop_join_artifacts(df)

    atr_col = f"atr_{config.features.atr_period}"
    if atr_col not in df.columns:
        raise ValueError(f"{atr_col} missing. Run feature engineering first.")
    return df, atr_col


def _compute_triple_barrier(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    tp_mult: float,
    sl_mult: float,
    horizon: int,
    min_atr: float,
) -> tuple:
    """Delegate to numba compute_labels. Returns (labels, upper, lower, touched)."""
    from thesis.stage_3_labels._label_numba import compute_labels as _numba_labels

    return _numba_labels(close, high, low, atr, tp_mult, sl_mult, horizon, min_atr)


def _attach_label_columns(
    df: pl.DataFrame,
    labels: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
    touched: np.ndarray,
    event_end: np.ndarray,
    weights: np.ndarray,
) -> pl.DataFrame:
    """Attach label, barrier, touched, event_end, sample_weight columns."""
    return df.with_columns(
        [
            pl.Series("label", labels),
            pl.Series("upper_barrier", upper),
            pl.Series("lower_barrier", lower),
            pl.Series("touched_bar", touched),
            pl.Series("event_end", event_end),
            pl.Series("sample_weight", weights),
        ]
    )


def _drop_join_artifacts(df: pl.DataFrame) -> pl.DataFrame:
    """Drop _right suffix columns from inner join. Verify timestamp uniqueness."""
    right_cols = [c for c in df.columns if c.endswith("_right")]
    if right_cols:
        logger.warning(
            "Dropping %d join-artifact columns: %s",
            len(right_cols),
            right_cols,
        )
        df = df.drop(right_cols)
    _check_unique_timestamps(df, "joined feature/OHLCV")
    return df


def _check_unique_timestamps(df: pl.DataFrame, name: str) -> None:
    """Raise on duplicate timestamps."""
    if "timestamp" not in df.columns:
        return
    dup_count = len(df) - df["timestamp"].n_unique()
    if dup_count > 0:
        raise ValueError(
            f"{name} has {dup_count} duplicate timestamps — deduplicate first."
        )


def _validate_no_join_artifacts(df: pl.DataFrame) -> None:
    """Fail if output has _right suffix columns."""
    right_cols = [c for c in df.columns if c.endswith("_right")]
    if right_cols:
        raise ValueError(f"labels.parquet contains join artifacts: {right_cols}")


def _drop_censored_and_nan(df: pl.DataFrame) -> pl.DataFrame:
    """Drop censored (label=-2) and NaN regression_target rows."""
    n_before = len(df)
    n_censored = int((df["label"] == CENSORED_LABEL).sum())
    if n_censored > 0:
        df = df.filter(pl.col("label") != CENSORED_LABEL)

    n_nan = 0
    if "regression_target" in df.columns:
        n_nan = int(df["regression_target"].is_nan().sum())
        if n_nan > 0:
            df = df.filter(pl.col("regression_target").is_not_nan())

    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.info(
            "Dropped %d rows (censored=%d regression_nan=%d) — insufficient horizon",
            n_dropped,
            n_censored,
            n_nan,
        )
    return df


def _log_atr_stats(df: pl.DataFrame, atr_col: str, min_atr: float) -> None:
    """ATR min/median/p5/p95 + % below min_atr floor."""
    s = df.select(
        pl.col(atr_col).min().alias("min"),
        pl.col(atr_col).median().alias("median"),
        pl.col(atr_col).quantile(ATR_LOW_QUANTILE).alias("p5"),
        pl.col(atr_col).quantile(ATR_HIGH_QUANTILE).alias("p95"),
        (pl.col(atr_col) < min_atr).mean().alias("floor_rate"),
    ).row(0, named=True)
    logger.info(
        "ATR (%s): min=%.6f median=%.6f p5=%.6f p95=%.6f below_min=%.2f%%",
        atr_col,
        s["min"] or 0.0,
        s["median"] or 0.0,
        s["p5"] or 0.0,
        s["p95"] or 0.0,
        (s["floor_rate"] or 0.0) * 100.0,
    )


def _log_distribution(df: pl.DataFrame) -> None:
    """Label class counts + percentages."""
    if "label" not in df.columns:
        return
    total = len(df)
    for label, count in df["label"].value_counts().sort("label").iter_rows():
        logger.info("  Class %s: %d (%.1f%%)", label, count, count / total * 100)


def _log_weight_stats(df: pl.DataFrame) -> None:
    """Sample weight min/median/max/mean."""
    if "sample_weight" not in df.columns:
        return
    s = df.select(
        pl.col("sample_weight").min().alias("min"),
        pl.col("sample_weight").median().alias("median"),
        pl.col("sample_weight").max().alias("max"),
        pl.col("sample_weight").mean().alias("mean"),
    ).row(0, named=True)
    logger.info(
        "Sample weights: min=%.4f median=%.4f max=%.4f mean=%.4f",
        s["min"] or 0.0,
        s["median"] or 0.0,
        s["max"] or 0.0,
        s["mean"] or 0.0,
    )


def _log_label_profitability(df: pl.DataFrame, config: Config) -> None:
    """% long/short labels profitable after trading costs.

    Net return = (close[t+h] - close[t+1]) / close[t+1] * leverage - cost / close[t+1].
    """
    if not {"close", "label", "timestamp"}.issubset(df.columns):
        return

    h = config.labels.horizon_bars
    cost = _roundtrip_cost_price_units(config)
    lev = config.backtest.leverage

    result = (
        df.sort("timestamp")
        .with_columns(
            (
                (pl.col("close").shift(-h) - pl.col("close").shift(-1))
                / pl.col("close").shift(-1)
                * lev
                - cost / pl.col("close").shift(-1)
            ).alias("_net_return")
        )
        .filter(
            (pl.col("label") != CENSORED_LABEL) & pl.col("_net_return").is_not_null()
        )
    )

    if result.is_empty():
        logger.warning("Label profitability: no valid samples.")
        return

    for label_val, name, profit_expr in (
        (1, "Long", pl.col("_net_return") > 0),
        (-1, "Short", pl.col("_net_return") < 0),
    ):
        class_df = result.filter(pl.col("label") == label_val)
        total = class_df.height
        if total == 0:
            logger.info("  Class %d (%s): no samples", label_val, name)
            continue
        profitable = class_df.filter(profit_expr).height
        pct = profitable / total * 100.0
        logger.info(
            "  %s labels profitable after costs: %.1f%% (%d/%d)",
            name,
            pct,
            profitable,
            total,
        )

    hold_total = result.filter(pl.col("label") == 0).height
    if hold_total:
        logger.info("  Class 0 (Hold): %d samples", hold_total)

    # warn if both sides unprofitable after costs
    long_df = result.filter(pl.col("label") == 1)
    short_df = result.filter(pl.col("label") == -1)
    long_pct = (
        long_df.filter(pl.col("_net_return") > 0).height / long_df.height * 100
        if long_df.height > 0
        else 0.0
    )
    short_pct = (
        short_df.filter(pl.col("_net_return") < 0).height / short_df.height * 100
        if short_df.height > 0
        else 0.0
    )
    if (
        long_pct < LABEL_PROFITABILITY_WARN_PCT
        and short_pct < LABEL_PROFITABILITY_WARN_PCT
    ):
        logger.warning(
            "LABEL PROFITABILITY LOW: Long %.1f%% Short %.1f%% — "
            "labels may not be economically useful after trading costs",
            long_pct,
            short_pct,
        )


def _roundtrip_cost_price_units(config: Config) -> float:
    """Spread + slippage + commission → price units per round-trip."""
    return (
        (config.backtest.spread_ticks + config.backtest.slippage_ticks)
        * config.data.tick_size
        + config.backtest.commission_per_lot
        * ROUNDTRIP_MULT
        / config.data.contract_size
    )
