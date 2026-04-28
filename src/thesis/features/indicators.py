"""Feature engineering — 11 core technical indicators, Polars-native.

No ta-lib, no pandas. All indicators are Polars expressions applied to an OHLCV DataFrame.

Features produced:
    - rsi_14: Relative Strength Index
    - atr_14: Average True Range
    - macd_hist: MACD histogram
    - atr_ratio: ATR(5) / ATR(20) — volatility regime
    - price_dist_ratio: (Close - EMA89) / ATR14 — normalized trend distance
    - pivot_position: (Close - S1) / (R1 - S1) — bounded [0,1]
    - atr_percentile: rolling rank of ATR14 over 50 bars
    - sess_asia: Asian session (America/New_York timezone, DST-aware)
    - sess_london: London AM session
    - sess_overlap: London-NY overlap
    - sess_ny_pm: NY afternoon session
"""

import json
import logging
from pathlib import Path

import polars as pl

from thesis.config import Config
from thesis.constants import EXCLUDE_COLS as _EXCLUDE_COLS

logger = logging.getLogger("thesis.features")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_features(config: Config) -> None:
    """
    Generate and persist feature-enriched OHLCV bars based on the provided configuration.

    Loads OHLCV data from config.paths.ohlcv, computes technical indicators and normalized/session features, forward-fills remaining nulls (then replaces any remaining nulls with 0.0), writes the enriched bars to config.paths.features, and saves a sidecar JSON file listing the produced feature column names (excluding core/label columns).

    Args:
        config (Config): Application configuration containing input/output paths and feature parameters.

    Raises:
        FileNotFoundError: If the OHLCV parquet file at config.paths.ohlcv does not exist.
    """
    ohlcv_path = Path(config.paths.ohlcv)
    if not ohlcv_path.exists():
        raise FileNotFoundError(f"OHLCV not found: {ohlcv_path}")

    logger.info("Loading OHLCV: %s", ohlcv_path)
    df = pl.read_parquet(ohlcv_path)
    logger.info("Input bars: %d", len(df))

    # --- Core indicators ---
    df = _add_rsi(df, config)
    df = _add_atr(df, config)
    df = _add_macd(df, config)

    # --- New normalized features ---
    df = _add_new_features(df, config)

    # Fill NaN from warm-up periods
    df = df.fill_null(strategy="forward")
    df = df.fill_null(0.0)

    # Persist
    out_path = Path(config.paths.features)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)

    feature_cols = sorted(c for c in df.columns if c not in _EXCLUDE_COLS)
    _save_feature_list(out_path, feature_cols)

    logger.info(
        "Features saved: %s (%d columns, %d rows)", out_path, len(df.columns), len(df)
    )
    logger.info("Feature columns (%d): %s", len(feature_cols), feature_cols)


# ---------------------------------------------------------------------------
# Column sets — imported from thesis.constants (single source of truth)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Indicator helpers (all Polars-native)
# ---------------------------------------------------------------------------


def _add_rsi(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """
    Compute Wilder-style Relative Strength Index (RSI) and append it as a column.

    Args:
        df (pl.DataFrame): Input OHLCV dataframe containing a `close` column.
        config (Config): Configuration object with `features.rsi_period` specifying the RSI period.

    Returns:
        pl.DataFrame: New dataframe with an added column `rsi_{p}` where `{p}` is the configured RSI period.
    """
    p = config.features.rsi_period
    delta = pl.col("close").diff()
    gain = delta.clip(lower_bound=0.0)
    loss = (-delta).clip(lower_bound=0.0)
    avg_gain = gain.ewm_mean(alpha=1.0 / p, adjust=False)
    avg_loss = loss.ewm_mean(alpha=1.0 / p, adjust=False)
    rs = avg_gain / (avg_loss + 1e-10)
    return df.with_columns((100.0 - 100.0 / (1.0 + rs)).alias(f"rsi_{p}"))


def _add_atr(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """
    Add an Average True Range (ATR) column to the DataFrame.

    Returns:
        pl.DataFrame: The input DataFrame with a new column named `atr_{p}` (where `p` is `config.features.atr_period`) containing the ATR values.
    """
    p = config.features.atr_period
    atr = _compute_atr_expr(p)
    return df.with_columns(atr.alias(f"atr_{p}"))


def _add_macd(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """
    Add a `macd_hist` column to the DataFrame using configured MACD spans.

    The MACD histogram is computed as the difference between the MACD line (EMA(fast) - EMA(slow)) and its signal line (EMA of the MACD line) using spans from `config.features.macd_fast`, `macd_slow`, and `macd_signal`.

    Returns:
        pl.DataFrame: The input DataFrame with an added `macd_hist` column.
    """
    fast = config.features.macd_fast
    slow = config.features.macd_slow
    sig = config.features.macd_signal
    ema_fast = pl.col("close").ewm_mean(span=fast, adjust=False)
    ema_slow = pl.col("close").ewm_mean(span=slow, adjust=False)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm_mean(span=sig, adjust=False)
    return df.with_columns(
        (macd_line - signal_line).alias("macd_hist"),
    )


# ---------------------------------------------------------------------------
# ATR expression helper
# ---------------------------------------------------------------------------


def _compute_atr_expr(period: int) -> pl.Expr:
    """
    Compute an expression for the Average True Range (ATR) smoothed using a Wilder-style exponential moving average.

    Args:
        period (int): Lookback period used for ATR smoothing.

    Returns:
        pl.Expr: Polars expression that yields the ATR series for the specified period.
    """
    tr = pl.max_horizontal(
        [
            (pl.col("high") - pl.col("low")),
            (pl.col("high") - pl.col("close").shift(1)).abs(),
            (pl.col("low") - pl.col("close").shift(1)).abs(),
        ]
    )
    return tr.ewm_mean(alpha=1.0 / period, adjust=False)


# ---------------------------------------------------------------------------
# New normalized features
# ---------------------------------------------------------------------------


def _add_new_features(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """
    Add derived, normalized, and regime/session features to the input OHLCV DataFrame.

    The function appends these columns:
    - `atr_ratio`: ATR(5) divided by ATR(20) (volatility regime).
    - `price_dist_ratio`: (close - EMA89) divided by ATR_{p} where `p = config.features.atr_period`.
    - `pivot_position`: bounded position of price between previous-day S1 and R1 in [0.0, 1.0].
    - Session indicator columns: `sess_asia`, `sess_london`, `sess_overlap`, `sess_ny_pm` (America/New_York, DST-aware).
    - `atr_percentile`: rolling rank of ATR_{p} over a 50-bar window scaled to [0, 1].

    Args:
        config (Config): Configuration object; `config.features.atr_period` determines which ATR column (`atr_{p}`) is used.

    Returns:
        pl.DataFrame: The input DataFrame with the new feature columns appended.
    """
    p = config.features.atr_period

    # ATR Ratio: ATR(5) / ATR(20) — volatility regime
    atr_5 = _compute_atr_expr(5)
    atr_20 = _compute_atr_expr(20)
    df = df.with_columns((atr_5 / (atr_20 + 1e-10)).alias("atr_ratio"))

    # Price Distance Ratio: (Close - EMA89) / ATR14 — normalized trend distance
    ema_89 = pl.col("close").ewm_mean(span=89, adjust=False)
    df = df.with_columns(
        ((pl.col("close") - ema_89) / (pl.col(f"atr_{p}") + 1e-10)).alias(
            "price_dist_ratio"
        )
    )

    # Pivot Position: (Close - S1) / (R1 - S1) — bounded [0,1]
    df = _add_pivot_position(df)

    # Session encoding in America/New_York timezone (DST-aware)
    df = _add_ny_session_dummies(df)

    # ATR Percentile: normalized rolling rank of ATR14 over 50 bars → [0, 1]
    df = df.with_columns(
        (
            pl.col(f"atr_{p}").rolling_rank(window_size=50, method="average") / 50.0
        ).alias("atr_percentile")
    )

    return df


def _add_pivot_position(df: pl.DataFrame) -> pl.DataFrame:
    """Compute previous-day pivot levels and add a bounded pivot_position column."""
    trading_day_expr = _to_ny_trading_day(df)
    pivots = _build_pivot_table(df, trading_day_expr)
    df = df.with_columns(trading_day_expr.alias("_trading_day"))
    df = df.join(
        pivots, left_on="_trading_day", right_on="_trading_day", how="left"
    ).drop("_trading_day")
    return _compute_pivot_position(df)


def _to_ny_trading_day(df: pl.DataFrame) -> pl.Expr:
    """Convert timestamp column to NY trading-day expression (7pm NY = start of trading day)."""
    ts = pl.col("timestamp")
    if df["timestamp"].dtype.time_zone is None:
        ts = ts.dt.replace_time_zone("UTC")
    ts_ny = ts.dt.convert_time_zone("America/New_York")
    return (ts_ny + pl.duration(hours=7)).dt.truncate("1d")


def _build_pivot_table(df: pl.DataFrame, trading_day_expr: pl.Expr) -> pl.DataFrame:
    """Build previous-day pivot/R1/S1 lookup table for pivot_position computation."""
    df_with_day = df.with_columns(trading_day_expr.alias("_trading_day"))
    daily = (
        df_with_day.group_by("_trading_day")
        .agg(
            [
                pl.col("high").max().alias("day_high"),
                pl.col("low").min().alias("day_low"),
                pl.col("close").last().alias("day_close"),
            ]
        )
        .sort("_trading_day")
    )
    pivot = (daily["day_high"] + daily["day_low"] + daily["day_close"]) / 3.0
    r1 = 2.0 * pivot - daily["day_low"]
    s1 = 2.0 * pivot - daily["day_high"]
    return (
        daily.with_columns([pivot.alias("pivot"), r1.alias("r1"), s1.alias("s1")])
        .select(["_trading_day", "pivot", "r1", "s1"])
        .with_columns(
            [
                pl.col("pivot").shift(1).alias("prev_pivot"),
                pl.col("r1").shift(1).alias("prev_r1"),
                pl.col("s1").shift(1).alias("prev_s1"),
            ]
        )
        .select(["_trading_day", "prev_pivot", "prev_r1", "prev_s1"])
    )


def _compute_pivot_position(df: pl.DataFrame) -> pl.DataFrame:
    """Compute bounded pivot_position and drop intermediate pivot columns."""
    return df.with_columns(
        (
            (pl.col("close") - pl.col("prev_s1"))
            / (pl.col("prev_r1") - pl.col("prev_s1") + 1e-10)
        )
        .clip(0.0, 1.0)
        .alias("pivot_position")
    ).drop(["prev_pivot", "prev_r1", "prev_s1"])


def _add_ny_session_dummies(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add four New York session indicator columns to the DataFrame using the America/New_York timezone (DST-aware).

    Handles naive timestamps by first assigning UTC before converting to America/New_York. Adds the following Int8 columns based on the local NY hour:
    - sess_asia: 18–23 and 0–1
    - sess_london: 3–7
    - sess_overlap: 8–11
    - sess_ny_pm: 12–17

    Args:
        df (pl.DataFrame): Input OHLCV bars containing a "timestamp" column.

    Returns:
        pl.DataFrame: A new DataFrame with the four session indicator columns appended.
    """
    # Handle naive timestamps by adding UTC first
    ts = pl.col("timestamp")
    if df["timestamp"].dtype.time_zone is None:
        ts = ts.dt.replace_time_zone("UTC")

    ny_hour = ts.dt.convert_time_zone("America/New_York").dt.hour()

    return df.with_columns(
        [
            (ny_hour.is_in(list(range(18, 24))) | ny_hour.is_in(list(range(0, 2))))
            .cast(pl.Int8)
            .alias("sess_asia"),
            ny_hour.is_in(list(range(3, 8))).cast(pl.Int8).alias("sess_london"),
            ny_hour.is_in(list(range(8, 12))).cast(pl.Int8).alias("sess_overlap"),
            ny_hour.is_in(list(range(12, 18))).cast(pl.Int8).alias("sess_ny_pm"),
        ]
    )


# ---------------------------------------------------------------------------
# Feature list sidecar
# ---------------------------------------------------------------------------


def _save_feature_list(features_path: Path, feature_cols: list[str]) -> None:
    """
    Write a JSON sidecar file next to the given features path named "<features_path>.feature_list.json" containing the feature column names.

    Args:
        features_path (Path): Path to the features output file; the sidecar file is created by replacing this path's suffix with ".feature_list.json".
        feature_cols (list[str]): Ordered list of feature column names to write to the sidecar.
    """
    list_path = features_path.with_suffix(".feature_list.json")
    with open(list_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
