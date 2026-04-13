"""Optimized feature engineering for XAU/USD H1 trading signals.

Features (target ~15-18 after correlation filtering):
    - Momentum: RSI-14, Zero-Lag MACD histogram
    - Volatility: ATR-14, ATR percentile
    - Trend: Donchian position, LSQ slope (zero/minimal lag)
    - Volume: volume ratio, volume delta MA-20
    - Microstructure: body-wick ratio, consecutive bull/bear, close in range
    - Levels: EMA 34/89 Fibonacci distances, daily pivot distance
    - Price Action: liquidity sweep, session high/low position
    - Spread: spread percentage
    - Time: session one-hot, day of week
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import talib
from numpy.polynomial.polynomial import polyfit

from thesis.config.loader import Config

logger = logging.getLogger("thesis.features")

# Small epsilon for division-by-zero protection in ratio features.
# Note: this is separate from config.labels.min_atr which is used for barrier calculations.
_EPSILON = 1e-10


def generate_features(config: Config) -> None:
    """Build and persist feature-enriched bars for modeling.

    The function loads OHLCV bars, applies optimized feature families,
    handles missing values, optionally prunes highly correlated columns,
    and saves both the feature table and feature manifest.

    Args:
        config: Loaded application configuration.
    """
    # Load OHLCV
    ohlcv_path = Path(config.data.ohlcv_path)

    if not ohlcv_path.exists():
        raise FileNotFoundError(f"OHLCV not found: {ohlcv_path}")

    logger.info(f"Loading OHLCV: {ohlcv_path}")
    df = pl.read_parquet(ohlcv_path)

    # Feature routines rely on pandas APIs and TA-Lib; convert once upfront.
    df_pd = df.to_pandas()

    # Generate technical features (RSI, Zero-Lag MACD, ATR, EMA distances)
    if config.features.use_technical:
        logger.info("Generating technical indicators...")
        df_pd = _add_technical_indicators(df_pd, config)

    # Generate volume + ATR percentile features
    logger.info("Generating volume and regime features...")
    df_pd = _add_volume_features(df_pd, config)

    # Generate microstructure features (candle analysis, orderflow proxy)
    logger.info("Generating microstructure features...")
    df_pd = _add_microstructure_features(df_pd, config)

    # Generate pivot points (daily S/R)
    if config.features.use_pivots:
        logger.info("Generating pivot points...")
        df_pd = _add_pivot_points(df_pd, config)

    # Generate session features + session high/low
    if config.features.use_session:
        logger.info("Generating session encoding + session high/low...")
        df_pd = _add_session_features(df_pd, config)

    # Generate spread features
    if config.features.use_spread:
        logger.info("Generating spread features...")
        df_pd = _add_spread_features(df_pd, config)

    # Generate liquidity sweep features
    logger.info("Generating liquidity sweep features...")
    df_pd = _add_liquidity_sweep_features(df_pd, config)

    # Generate zero-lag trend features (Donchian, LSQ slope)
    logger.info("Generating zero-lag trend features...")
    df_pd = _add_trend_features(df_pd, config)

    # Convert back to polars
    df = pl.from_pandas(df_pd)

    # Handle missing values (from indicators)
    df = df.fill_null(strategy="forward")
    df = df.fill_null(0)  # Fill remaining with 0

    # Drop high correlation features
    # NOTE: Correlation filtering is now performed in the split stage
    # (splitting.py) to ensure it uses training data only.
    # This prevents feature selection leakage into val/test sets.

    # Save feature list
    feature_cols = [
        c
        for c in df.columns
        if c
        not in [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "avg_spread",
            "tick_count",
        ]
    ]
    feature_list = {
        "features": feature_cols,
        "count": len(feature_cols),
        "technical": config.features.use_technical,
        "session": config.features.use_session,
    }

    feature_list_path = Path(config.features.feature_list_path)
    feature_list_path.parent.mkdir(parents=True, exist_ok=True)
    with open(feature_list_path, "w") as f:
        json.dump(feature_list, f, indent=2)

    # Save features
    output_path = Path(config.features.features_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)

    logger.info(f"Generated {len(feature_cols)} features")
    logger.info(f"Saved features: {output_path}")
    logger.info(f"Feature list: {feature_list_path}")


# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------


def _add_technical_indicators(
    df: pl.DataFrame | pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    """Add core technical indicators: RSI, Zero-Lag MACD, ATR, EMA levels.

    Args:
        df: OHLCV bars as a Polars or pandas dataframe.
        config: Application configuration.

    Returns:
        pandas DataFrame with technical indicators appended.
    """
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    # EMAs (kept as Fibonacci support/resistance levels, not trend direction)
    for period in config.features.ema_periods:
        df[f"ema_{period}"] = talib.EMA(df["close"], timeperiod=period)

    # RSI - momentum oscillator
    df["rsi_14"] = talib.RSI(df["close"], timeperiod=config.features.rsi_period)

    # Zero-Lag MACD (EMA-of-EMA detrending removes lag)
    zl_fast = _zero_lag_ema(df["close"], config.features.macd_fast)
    zl_slow = _zero_lag_ema(df["close"], config.features.macd_slow)
    zl_macd_line = zl_fast - zl_slow
    zl_signal = talib.EMA(zl_macd_line, timeperiod=config.features.macd_signal)
    df["macd_hist_zl"] = zl_macd_line - zl_signal

    # ATR - core volatility measure
    df["atr_14"] = talib.ATR(
        df["high"], df["low"], df["close"], timeperiod=config.features.atr_period
    )

    # EMA distance features (as Fibonacci S/R levels)
    for period in config.features.ema_periods:
        df[f"close_dist_ema_{period}"] = (df["close"] - df[f"ema_{period}"]) / df[
            f"ema_{period}"
        ]

    # Drop raw EMA columns (only keep distance features)
    df = df.drop(columns=[f"ema_{p}" for p in config.features.ema_periods])

    return df


def _zero_lag_ema(series: pd.Series, period: int) -> pd.Series:
    """Compute Zero-Lag EMA using EMA-of-EMA detrending.

    Formula: ZLEMA = 2 * EMA(close, period) - EMA(EMA(close, period), period)

    Args:
        series: Price series.
        period: EMA period.

    Returns:
        Zero-lag EMA series.
    """
    ema1 = talib.EMA(series, timeperiod=period)
    ema2 = talib.EMA(ema1, timeperiod=period)
    return 2 * ema1 - ema2


# ---------------------------------------------------------------------------
# Volume + regime features
# ---------------------------------------------------------------------------


def _add_volume_features(
    df: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    """Add volume ratio and ATR percentile features.

    Args:
        df: OHLCV DataFrame.
        config: Application configuration.

    Returns:
        DataFrame with volume and regime features.
    """
    # Relative volume (current vs 20-bar average)
    vol_ma_20 = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / vol_ma_20

    # ATR percentile (volatility regime: trending vs ranging)
    if "atr_14" in df.columns:
        df["atr_percentile"] = df["atr_14"].rolling(50).rank(pct=True)

    logger.info("  Added volume features: volume_ratio, atr_percentile")

    return df


# ---------------------------------------------------------------------------
# Microstructure features (lean)
# ---------------------------------------------------------------------------


def _add_microstructure_features(
    df: pl.DataFrame | pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    """Add lean microstructure features: body-wick ratio, streaks, orderflow.

    Only keeps features with unique predictive information. All intermediate
    columns (body, range, wicks) are computed then dropped.

    Args:
        df: OHLCV bars.
        config: Application configuration.

    Returns:
        pandas DataFrame with microstructure features.
    """
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    # Intermediate calculations (will be dropped)
    body = df["close"] - df["open"]
    bar_range = df["high"] - df["low"]
    upper_wick = df["high"] - df[["close", "open"]].max(axis=1)
    lower_wick = df[["close", "open"]].min(axis=1) - df["low"]
    is_bullish = (body > 0).astype(int)
    is_bearish = (body < 0).astype(int)

    # Body-to-wick ratio (market conviction)
    df["body_wick_ratio"] = body.abs() / (upper_wick + lower_wick + _EPSILON)

    # Close position in range (0 = at low, 1 = at high)
    df["close_in_range"] = (df["close"] - df["low"]) / (bar_range + _EPSILON)

    # Consecutive bar streaks
    df["consecutive_bull"] = is_bullish.groupby(
        (is_bullish != is_bullish.shift()).cumsum()
    ).cumsum()
    df["consecutive_bear"] = is_bearish.groupby(
        (is_bearish != is_bearish.shift()).cumsum()
    ).cumsum()

    # Volume delta (orderflow proxy) - smoothed
    volume_delta = np.where(
        df["close"] > df["open"],
        df["volume"],
        -df["volume"],
    )
    df["volume_delta_ma_20"] = pd.Series(volume_delta).rolling(20).mean()

    return df


# ---------------------------------------------------------------------------
# Pivot points (lean)
# ---------------------------------------------------------------------------


def _add_pivot_points(
    df: pl.DataFrame | pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    """Add pivot point features (lean version).

    Only keeps: dist_pivot, inside_prev_range.
    Removes: dist_r1, dist_s1, breakout_high, breakout_low (redundant).

    Args:
        df: OHLCV bars.
        config: Application configuration.

    Returns:
        pandas DataFrame with pivot features.
    """
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    # Ensure timestamp is datetime with timezone
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(
            "America/New_York", ambiguous="infer"
        )

    # Group by day
    df["date"] = df["timestamp"].dt.date

    # Get daily HLC
    daily = (
        df.groupby("date")
        .agg(
            {
                "high": "max",
                "low": "min",
                "close": "last",
            }
        )
        .reset_index()
    )

    # Calculate pivot
    daily["pivot"] = (daily["high"] + daily["low"] + daily["close"]) / 3

    # Shift to get previous day's values
    daily["pivot_prev"] = daily["pivot"].shift(1)
    daily["high_prev"] = daily["high"].shift(1)
    daily["low_prev"] = daily["low"].shift(1)

    # Merge back
    df = df.merge(
        daily[["date", "pivot_prev", "high_prev", "low_prev"]],
        on="date",
        how="left",
    )

    # Distance to pivot (normalized)
    df["dist_pivot"] = (df["close"] - df["pivot_prev"]) / df["pivot_prev"]

    # Inside previous day's range (contraction signal)
    df["inside_prev_range"] = (
        (df["high"] <= df["high_prev"]) & (df["low"] >= df["low_prev"])
    ).astype(int)

    df = df.drop(columns=["date"])

    return df


# ---------------------------------------------------------------------------
# Session features + session high/low
# ---------------------------------------------------------------------------


def _add_session_features(
    df: pl.DataFrame | pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    """Add session encoding and intraday session high/low features.

    Features:
        - Session one-hot: Asia, London, NY PM
        - Day of week
        - Session high/low distance (position within trading day range)

    Args:
        df: OHLCV bars.
        config: Application configuration.

    Returns:
        pandas DataFrame with session features.
    """
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    # Ensure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Extract hour in market timezone
    if df["timestamp"].dt.tz is None:
        df["hour"] = df["timestamp"].dt.hour
    else:
        df["hour"] = df["timestamp"].dt.tz_convert(config.data.market_tz).dt.hour

    # Session one-hot encoding
    df["session_asia"] = ((df["hour"] >= 0) & (df["hour"] < 8)).astype(int)
    df["session_london"] = ((df["hour"] >= 8) & (df["hour"] < 17)).astype(int)
    df["session_ny_pm"] = ((df["hour"] >= 17) & (df["hour"] < 21)).astype(int)

    # Day of week
    df["day_of_week"] = pd.to_datetime(df["timestamp"]).dt.dayofweek

    # Session high/low features (cumulative within trading day — NO LOOKAHEAD)
    # cummax/cummin only accumulate data UP TO the current row within the group,
    # so the 10:00 value uses only data from 00:00-10:00, not future bars.
    df["date"] = df["timestamp"].dt.date
    df["session_high"] = df.groupby("date")["high"].cummax()  # Running max
    df["session_low"] = df.groupby("date")["low"].cummin()  # Running min

    # Distance to session extremes (normalized by close)
    df["dist_session_high"] = (df["session_high"] - df["close"]) / df["close"]
    df["dist_session_low"] = (df["close"] - df["session_low"]) / df["close"]

    # Position within session range (0=at low, 1=at high)
    session_range = df["session_high"] - df["session_low"]
    df["session_position"] = (df["close"] - df["session_low"]) / (
        session_range + _EPSILON
    )

    df = df.drop(columns=["hour", "date", "session_high", "session_low"])

    return df


# ---------------------------------------------------------------------------
# Spread features (lean)
# ---------------------------------------------------------------------------


def _add_spread_features(
    df: pl.DataFrame | pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    """Add spread percentage feature.

    Args:
        df: OHLCV bars.
        config: Application configuration.

    Returns:
        pandas DataFrame with spread feature.
    """
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    # Avg spread relative to price
    df["spread_pct"] = df["avg_spread"] / df["close"] * 100

    return df


# ---------------------------------------------------------------------------
# Liquidity sweep features (new)
# ---------------------------------------------------------------------------


def _add_liquidity_sweep_features(
    df: pl.DataFrame | pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    """Detect liquidity sweep patterns (stop hunts).

    A liquidity sweep occurs when price breaks above/below a recent N-bar
    high/low then closes back inside the range — indicating stop hunting
    and potential reversal. Very relevant for XAU/USD.

    Args:
        df: OHLCV bars.
        config: Application configuration.

    Returns:
        pandas DataFrame with liquidity sweep features.
    """
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    for lookback in [5, 20]:
        # Previous N-bar high/low (excluding current bar)
        prev_high = df["high"].shift(1).rolling(lookback).max()
        prev_low = df["low"].shift(1).rolling(lookback).min()

        # Break above/below previous range
        broke_high = df["high"] > prev_high
        broke_low = df["low"] < prev_low

        # Close back inside (reversal) = sweep confirmed
        swept_high = broke_high & (df["close"] < prev_high)
        swept_low = broke_low & (df["close"] > prev_low)

        df[f"liq_sweep_high_{lookback}"] = swept_high.astype(int)
        df[f"liq_sweep_low_{lookback}"] = swept_low.astype(int)

    # Combined sweep signal (any lookback)
    df["liq_sweep_any"] = (
        df["liq_sweep_high_5"]
        | df["liq_sweep_low_5"]
        | df["liq_sweep_high_20"]
        | df["liq_sweep_low_20"]
    ).astype(int)

    logger.info(
        "  Added liquidity sweep features: liq_sweep_high/low_5/20, liq_sweep_any"
    )

    return df


# ---------------------------------------------------------------------------
# Zero-lag trend features (new)
# ---------------------------------------------------------------------------


def _add_trend_features(
    df: pl.DataFrame | pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    """Add zero-lag trend features: Donchian position and LSQ slope.

    These replace EMA-based trend direction indicators (which lag)
    while keeping EMA 34/89 as support/resistance level distances.

    Args:
        df: OHLCV bars.
        config: Application configuration.

    Returns:
        pandas DataFrame with trend features.
    """
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    # Donchian Channel Position (zero lag)
    # (close - 20bar_low) / (20bar_high - 20bar_low) → 0 to 1
    donchian_high = df["high"].rolling(20).max()
    donchian_low = df["low"].rolling(20).min()
    df["donchian_position"] = (df["close"] - donchian_low) / (
        donchian_high - donchian_low + _EPSILON
    )

    # Linear Regression Slope (minimal lag)
    # 20-bar LSQ slope normalized by price level
    df["lsq_slope_20"] = _rolling_lsq_slope(df["close"], window=20)

    logger.info("  Added trend features: donchian_position, lsq_slope_20")

    return df


def _rolling_lsq_slope(series: pd.Series, window: int = 20) -> pd.Series:
    """Compute rolling linear regression slope.

    Args:
        series: Price series.
        window: Lookback window.

    Returns:
        Slope series normalized by price level.
    """
    slopes = np.full(len(series), np.nan)
    x = np.arange(window, dtype=float)

    values = series.values
    for i in range(window - 1, len(series)):
        y = values[i - window + 1 : i + 1].astype(float)
        if not np.any(np.isnan(y)):
            # polyfit returns [intercept, slope] for degree 1
            coeffs = polyfit(x, y, 1)
            slope = coeffs[1]
            # Normalize by price level for cross-period comparability
            slopes[i] = slope / (y.mean() + _EPSILON)

    return pd.Series(slopes, index=series.index)


# ---------------------------------------------------------------------------
# Correlation filtering
# ---------------------------------------------------------------------------


def _drop_high_correlation(
    df: pl.DataFrame | pd.DataFrame,
    threshold: float,
) -> pl.DataFrame | pd.DataFrame:
    """Drop features with correlation above threshold.

    Args:
        df: Feature dataframe in Polars or pandas format.
        threshold: Absolute correlation threshold for feature removal.

    Returns:
        Dataframe with highly correlated numeric features removed.
    """
    # Convert to pandas for correlation
    df_pd = df.to_pandas() if isinstance(df, pl.DataFrame) else df

    # Select numeric columns only
    numeric_cols = df_pd.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ["timestamp"]]

    if len(numeric_cols) < 2:
        return df

    # Calculate correlation matrix
    corr = df_pd[numeric_cols].corr().abs()

    # Upper triangle
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    # Find features to drop
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    if to_drop:
        logger.info(
            f"Dropping {len(to_drop)} highly correlated features: {to_drop[:5]}..."
        )
        df_pd = df_pd.drop(columns=to_drop)

    return pl.from_pandas(df_pd) if isinstance(df, pl.DataFrame) else df_pd
