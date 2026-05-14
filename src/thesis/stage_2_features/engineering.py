"""Feature engineering — production pipeline for price-action features."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandera.polars as pa
import polars as pl

from thesis.shared.config import Config
from thesis.shared.constants import timeframe_to_ms
from thesis.shared.feature_registry import (
    build_feature_output_cols,
    get_static_feature_cols,
)
from thesis.shared.schemas import FeaturesSchema, OhlcvSchema
from thesis.shared.ui import console
from thesis.stage_2_features.indicators import (
    add_adx,
    add_atr,
    add_atr_percentile,
    add_atr_ratio,
    add_ema_crossover,
    add_ema_slope,
    add_high_low_range,
    add_log_returns,
    add_macd,
    add_ohlcv_norm,
    add_pivot_position,
    add_price_action,
    add_price_dist_ratio,
    add_regime,
    add_rsi,
    add_session_dummies,
    add_trend_regime,
    add_volatility_regime,
    add_volume_zscore,
    add_vwap,
)

logger = logging.getLogger("thesis.stage_2_features")


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def generate_features(config: Config) -> None:
    """Load OHLCV → add all features → validate → save parquet + feature list."""
    ohlcv_path = Path(config.paths.ohlcv)
    if not ohlcv_path.exists():
        raise FileNotFoundError(f"OHLCV not found: {ohlcv_path}")

    logger.info("Loading OHLCV: %s", ohlcv_path)
    with console.status(f"[cyan]Loading OHLCV[/] {ohlcv_path}"):
        df = pl.read_parquet(ohlcv_path)
    logger.info("Input bars: %d", len(df))
    OhlcvSchema.validate(df)
    _validate_ohlcv_input(df, config)

    # Stage 1: core indicators — ATR must come first (many divide by it)
    df = add_atr(df, config)
    df = add_rsi(df, config)
    df = add_adx(df, config)

    # Stage 2: multi-timeframe ATR ratio + percentile
    df = add_atr_ratio(df, config)
    df = add_atr_percentile(df, config)

    # Stage 3: EMA features — depend on ATR
    df = add_ema_slope(df, config)
    df = add_ema_crossover(df, config)

    # Stage 4: price-action + VWAP/pivot (session-anchored)
    df = add_price_action(df, config)
    df = add_price_dist_ratio(df, config)
    df = add_vwap(df)
    df = add_pivot_position(df)
    df = add_session_dummies(df)

    # Stage 5: volume + returns — rolling, no cross-feature dependencies
    df = add_volume_zscore(df, config)
    df = add_log_returns(df, config)
    df = add_high_low_range(df, config)

    # Stage 6: MACD — uses ATR column
    df = add_macd(df, config)

    # Stage 7: OHLCV z-score normalization
    df = add_ohlcv_norm(df, config)

    # Stage 8: regime features — gated behind config flag
    if config.features.enable_regime_features:
        df = add_volatility_regime(df, config)
        df = add_trend_regime(df, config)
        df = add_regime(df, config)

    # Map return_1h → log_returns for backward compatibility
    if "return_1h" in df.columns and "log_returns" not in df.columns:
        df = df.with_columns(pl.col("return_1h").alias("log_returns"))

    # Keep only columns registered in feature registry
    desired = build_feature_output_cols(config)
    existing = [c for c in desired if c in df.columns]
    df = df.select(existing)
    model_cols = sorted(
        c for c in df.columns if c in set(get_static_feature_cols(config))
    )
    df = _drop_warmup_rows(df, model_cols)
    _validate_feature_quality(df, config)

    FeaturesSchema.validate(df, config)
    out_path = Path(config.paths.features)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)

    _save_feature_list(out_path, model_cols)
    logger.info(
        "Features saved: %s (%d columns, %d rows)", out_path, len(df.columns), len(df)
    )
    logger.info("Feature columns (%d): %s", len(model_cols), model_cols)


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #


def _validate_ohlcv_input(df: pl.DataFrame, config: Config) -> None:
    """Reject empty, unsorted, duplicate, or heavily-gapped OHLCV."""
    if df.is_empty():
        raise ValueError("OHLCV is empty; cannot generate features")

    n_unsorted = int(
        df.select((pl.col("timestamp").diff().dt.total_milliseconds() < 0).sum()).item()
        or 0
    )
    n_dupes = len(df) - df["timestamp"].n_unique()
    if n_unsorted > 0:
        raise ValueError(f"OHLCV timestamps not sorted ({n_unsorted} reversals)")
    if n_dupes > 0:
        raise ValueError(f"OHLCV timestamps not unique ({n_dupes} duplicates)")

    if len(df) < 2:
        return

    # Log gap stats but do not fail — weekends and holidays expected
    expected_ms = timeframe_to_ms(config.data.timeframe)
    deltas = (
        df.select(pl.col("timestamp").diff().dt.total_milliseconds().alias("delta_ms"))
        .drop_nulls()
        .get_column("delta_ms")
    )
    gaps = deltas.filter(deltas > expected_ms)
    largest = int(deltas.max() or 0)
    logger.info(
        "Gap check: expected=%d ms, gap_count=%d, largest=%.2f bars",
        expected_ms,
        len(gaps),
        largest / expected_ms if expected_ms else 0.0,
    )


def _drop_warmup_rows(df: pl.DataFrame, feature_cols: list[str]) -> pl.DataFrame:
    """Drop rows with null or non-finite model-facing feature values."""
    existing = [c for c in feature_cols if c in df.columns]
    n_before = len(df)
    df = df.fill_nan(None).drop_nulls(subset=existing)
    if existing:
        # Ensure no inf/-inf from division-by-zero edge cases
        df = df.filter(pl.all_horizontal(pl.col(c).is_finite() for c in existing))
    dropped = n_before - len(df)
    if dropped > 0:
        logger.info("Dropped %d warm-up rows", dropped)
    if df.is_empty():
        raise ValueError("No feature rows remain after warm-up drop")
    return df


def _validate_feature_quality(df: pl.DataFrame, config: Config) -> None:
    """Pandera schema + timestamp uniqueness + strictly-increasing + no nulls."""
    p = config.features.rsi_period
    checks = {"timestamp": pa.Column(nullable=False)}
    if f"rsi_{p}" in df.columns:
        checks[f"rsi_{p}"] = pa.Column(
            pl.Float64,
            checks=[pa.Check.ge(0), pa.Check.le(100)],
            nullable=True,
            coerce=True,
        )
    pa.DataFrameSchema(checks, strict=False).validate(df, lazy=True)

    ts = df["timestamp"]
    if ts.n_unique() != len(ts):
        raise ValueError("Features validation failed: timestamp must be unique")
    deltas = ts.diff().drop_nulls().dt.total_milliseconds()
    if int((deltas <= 0).sum()) > 0:
        raise ValueError(
            "Features validation failed: timestamp must be strictly increasing"
        )
    if int(df.null_count().sum_horizontal().sum()) > 0:
        raise ValueError("Features validation failed: null values remain after warm-up")


def _save_feature_list(features_path: Path, feature_cols: list[str]) -> None:
    """Write JSON sidecar listing model-facing feature column names."""
    list_path = features_path.with_suffix(".feature_list.json")
    with open(list_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
