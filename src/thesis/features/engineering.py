"""Feature engineering for XAU/USD technical analysis.

Features:
    - Technical indicators: EMA, RSI, MACD, ATR
    - Pivot points (previous day H, L, C)
    - Session encoding (Asia, London, NY AM, NY PM)
    - Spread features
    - Lag features (for tree models)
"""

import json
import logging
from pathlib import Path

import numpy as np
import polars as pl
import talib

from thesis.config.loader import Config

logger = logging.getLogger("thesis.features")


def generate_features(config: Config) -> None:
    """Generate features from OHLCV data.
    
    Args:
        config: Configuration object.
    """
    # Load OHLCV
    ohlcv_path = Path(config.data.ohlcv_path)
    
    if not ohlcv_path.exists():
        raise FileNotFoundError(f"OHLCV not found: {ohlcv_path}")
    
    logger.info(f"Loading OHLCV: {ohlcv_path}")
    df = pl.read_parquet(ohlcv_path)
    
    # Convert to pandas for TA-Lib
    df_pd = df.to_pandas()
    
    # Generate technical features
    if config.features.use_technical:
        logger.info("Generating technical indicators...")
        df_pd = _add_technical_indicators(df_pd, config)
    
    # Generate microstructure features (candlestick patterns, volume delta)
    logger.info("Generating microstructure features...")
    df_pd = _add_microstructure_features(df_pd, config)
    
    # Generate pivot points
    if config.features.use_pivots:
        logger.info("Generating pivot points...")
        df_pd = _add_pivot_points(df_pd, config)
    
    # Generate session features
    if config.features.use_session:
        logger.info("Generating session encoding...")
        df_pd = _add_session_features(df_pd, config)
    
    # Generate lag features (for tree models)
    logger.info("Generating lag features...")
    df_pd = _add_lag_features(df_pd, config)
    
    # Generate spread features
    if config.features.use_spread:
        logger.info("Generating spread features...")
        df_pd = _add_spread_features(df_pd, config)
    
    # Convert back to polars
    df = pl.from_pandas(df_pd)
    
    # Handle missing values (from indicators)
    df = df.fill_null(strategy="forward")
    df = df.fill_null(0)  # Fill remaining with 0
    
    # Drop high correlation features
    if config.features.drop_high_corr:
        logger.info("Removing highly correlated features...")
        df = _drop_high_correlation(df, config.features.correlation_threshold)
    
    # Save feature list
    feature_cols = [c for c in df.columns if c not in ["timestamp", "open", "high", "low", "close", "volume", "avg_spread", "tick_count"]]
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


def _add_technical_indicators(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Add technical indicators using TA-Lib.
    
    Args:
        df: OHLCV DataFrame (pandas).
        config: Configuration object.
        
    Returns:
        DataFrame with technical indicators.
    """
    
    # Ensure pandas
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()
    
    # EMAs
    for period in config.features.ema_periods:
        df[f"ema_{period}"] = talib.EMA(df["close"], timeperiod=period)
    
    # RSI
    df["rsi_14"] = talib.RSI(df["close"], timeperiod=config.features.rsi_period)
    
    # MACD
    macd, macd_signal, macd_hist = talib.MACD(
        df["close"],
        fastperiod=config.features.macd_fast,
        slowperiod=config.features.macd_slow,
        signalperiod=config.features.macd_signal,
    )
    df["macd"] = macd
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist
    
    # ATR
    df["atr_14"] = talib.ATR(df["high"], df["low"], df["close"], timeperiod=config.features.atr_period)
    
    # Volatility
    df["returns"] = df["close"].pct_change()
    df["volatility_20"] = df["returns"].rolling(20).std() * np.sqrt(252)
    
    # Price distance from EMAs
    for period in config.features.ema_periods:
        df[f"close_dist_ema_{period}"] = (df["close"] - df[f"ema_{period}"]) / df[f"ema_{period}"]
    
    return df


def _add_pivot_points(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Add pivot point features.
    
    P = (H + L + C) / 3
    R1 = 2P - L
    S1 = 2P - H
    
    Args:
        df: DataFrame.
        config: Configuration object.
        
    Returns:
        DataFrame with pivot features.
    """
    import pandas as pd
    
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()
    
    # Ensure timestamp is datetime with timezone
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("America/New_York", ambiguous="infer")
    
    # Group by day
    df["date"] = df["timestamp"].dt.date
    
    # Get daily HLC
    daily = df.groupby("date").agg({
        "high": "max",
        "low": "min",
        "close": "last",
    }).reset_index()
    
    # Calculate pivots
    daily["pivot"] = (daily["high"] + daily["low"] + daily["close"]) / 3
    daily["r1"] = 2 * daily["pivot"] - daily["low"]
    daily["s1"] = 2 * daily["pivot"] - daily["high"]
    
    # Shift to get previous day's pivots
    daily["pivot_prev"] = daily["pivot"].shift(1)
    daily["r1_prev"] = daily["r1"].shift(1)
    daily["s1_prev"] = daily["s1"].shift(1)
    daily["high_prev"] = daily["high"].shift(1)
    daily["low_prev"] = daily["low"].shift(1)
    
    # Merge back
    df = df.merge(daily[["date", "pivot_prev", "r1_prev", "s1_prev", "high_prev", "low_prev"]], on="date", how="left")
    
    # Distance to pivots
    df["dist_pivot"] = (df["close"] - df["pivot_prev"]) / df["pivot_prev"]
    df["dist_r1"] = (df["close"] - df["r1_prev"]) / df["r1_prev"]
    df["dist_s1"] = (df["close"] - df["s1_prev"]) / df["s1_prev"]
    
    # Inside previous day's range?
    df["inside_prev_range"] = ((df["high"] <= df["high_prev"]) & (df["low"] >= df["low_prev"])).astype(int)
    
    # Breakout of previous day's range?
    df["breakout_high"] = (df["high"] > df["high_prev"]).astype(int)
    df["breakout_low"] = (df["low"] < df["low_prev"]).astype(int)
    
    df = df.drop(columns=["date"])
    
    return df


def _add_session_features(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Add session encoding features.
    
    Sessions:
        - Asia: 00:00-08:00 NY
        - London: 08:00-17:00 NY (includes NY AM session)
        - NY PM: 17:00-21:00 NY
    
    Args:
        df: DataFrame.
        config: Configuration object.
        
    Returns:
        DataFrame with session features.
    """
    import pandas as pd
    
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()
    
    # Ensure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Extract hour in market timezone
    if df["timestamp"].dt.tz is None:
        df["hour"] = df["timestamp"].dt.hour
    else:
        # Convert to market timezone
        df["hour"] = df["timestamp"].dt.tz_convert(config.data.market_tz).dt.hour
    
    # Session one-hot encoding
    # Asia: 0-8
    df["session_asia"] = ((df["hour"] >= 0) & (df["hour"] < 8)).astype(int)
    # London: 8-17
    df["session_london"] = ((df["hour"] >= 8) & (df["hour"] < 17)).astype(int)
    # NY PM: 17-21
    df["session_ny_pm"] = ((df["hour"] >= 17) & (df["hour"] < 21)).astype(int)
    
    # Day of week
    df["day_of_week"] = pd.to_datetime(df["timestamp"]).dt.dayofweek
    df["is_monday"] = (df["day_of_week"] == 0).astype(int)
    df["is_friday"] = (df["day_of_week"] == 4).astype(int)
    
    df = df.drop(columns=["hour"])
    
    return df


def _add_lag_features(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Add lag features for tree models.
    
    Args:
        df: DataFrame.
        config: Configuration object.
        
    Returns:
        DataFrame with lag features.
    """
    
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()
    
    # Price lags
    for lag in config.features.lag_periods:
        df[f"close_lag_{lag}"] = df["close"].shift(lag)
        df[f"returns_lag_{lag}"] = df["close"].pct_change(lag)
        df[f"high_lag_{lag}"] = df["high"].shift(lag)
        df[f"low_lag_{lag}"] = df["low"].shift(lag)
    
    # Volume lags
    for lag in config.features.lag_periods[:3]:  # Only first 3 lags for volume
        df[f"volume_lag_{lag}"] = df["volume"].shift(lag)
    
    return df


def _add_spread_features(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Add spread-related features.
    
    Args:
        df: DataFrame.
        config: Configuration object.
        
    Returns:
        DataFrame with spread features.
    """
    
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()
    
    # Avg spread relative to price
    df["spread_pct"] = df["avg_spread"] / df["close"] * 100
    
    # Spread MA
    df["spread_ma_20"] = df["avg_spread"].rolling(20).mean()
    df["spread_ratio"] = df["avg_spread"] / df["spread_ma_20"]
    
    return df


def _add_microstructure_features(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Add candlestick patterns, volume delta, and orderflow proxies.
    
    Features for directional trading signal prediction:
        - Candlestick patterns (engulfing, doji, hammer, etc.)
        - Volume delta (buying/selling pressure)
        - Consecutive bar analysis
        - Body-to-wick ratios (market conviction)
    
    Args:
        df: DataFrame.
        config: Configuration object.
        
    Returns:
        DataFrame with microstructure features.
    """
    
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()
    
    # 1. Candlestick Patterns
    df["body"] = df["close"] - df["open"]
    df["body_prev"] = df["body"].shift(1)
    df["is_bullish"] = (df["body"] > 0).astype(int)
    df["is_bearish"] = (df["body"] < 0).astype(int)
    
    # Bullish Engulfing
    df["bullish_engulfing"] = (
        (df["is_bullish"] == 1) & 
        (df["body_prev"] < 0) & 
        (df["body"].abs() > df["body_prev"].abs())
    ).astype(int)
    
    # Bearish Engulfing
    df["bearish_engulfing"] = (
        (df["is_bearish"] == 1) & 
        (df["body_prev"] > 0) & 
        (df["body"].abs() > df["body_prev"].abs())
    ).astype(int)
    
    # Doji
    df["range"] = df["high"] - df["low"]
    df["doji"] = (df["body"].abs() / (df["range"] + 1e-10) < 0.1).astype(int)
    
    # Hammer
    df["upper_wick"] = df["high"] - df[["close", "open"]].max(axis=1)
    df["lower_wick"] = df[["close", "open"]].min(axis=1) - df["low"]
    df["hammer"] = (
        (df["lower_wick"] > 2 * df["body"].abs()) & 
        (df["upper_wick"] < df["body"].abs()) &
        (df["is_bullish"] == 1)
    ).astype(int)
    
    # Shooting Star
    df["shooting_star"] = (
        (df["upper_wick"] > 2 * df["body"].abs()) & 
        (df["lower_wick"] < df["body"].abs()) &
        (df["is_bearish"] == 1)
    ).astype(int)
    
    # Marubozu (no wicks - strong directional conviction)
    df["marubozu"] = (
        (df["upper_wick"] < 0.1 * df["range"]) & 
        (df["lower_wick"] < 0.1 * df["range"])
    ).astype(int)
    
    # 2. Volume Delta (Orderflow proxy)
    df["volume_delta"] = np.where(
        df["close"] > df["open"],
        df["volume"],   # Buying volume
        -df["volume"]   # Selling volume
    )
    
    # Volume delta moving averages
    df["volume_delta_ma_10"] = df["volume_delta"].rolling(10).mean()
    df["volume_delta_ma_20"] = df["volume_delta"].rolling(20).mean()
    
    # 3. Body-to-Wick Ratio (Market conviction)
    df["body_wick_ratio"] = df["body"].abs() / (df["upper_wick"] + df["lower_wick"] + 1e-10)
    
    # 4. Consecutive bars analysis
    df["consecutive_bull"] = df["is_bullish"].groupby(
        (df["is_bullish"] != df["is_bullish"].shift()).cumsum()
    ).cumsum()
    
    df["consecutive_bear"] = df["is_bearish"].groupby(
        (df["is_bearish"] != df["is_bearish"].shift()).cumsum()
    ).cumsum()
    
    # 5. Tick intensity (relative volume)
    df["tick_intensity"] = df["volume"] / df["volume"].rolling(20).mean()
    
    # 6. Close position in range (bullish/bearish pressure)
    df["close_in_range"] = (df["close"] - df["low"]) / (df["range"] + 1e-10)
    
    # 7. Large body detection (strong momentum)
    body_ma = df["body"].abs().rolling(20).mean()
    df["large_body"] = (df["body"].abs() > 1.5 * body_ma).astype(int)
    
    # Cleanup intermediate columns
    df = df.drop(columns=["body_prev"])
    
    return df


def _drop_high_correlation(df: pl.DataFrame, threshold: float) -> pl.DataFrame:
    """Drop features with correlation above threshold.
    
    Args:
        df: DataFrame.
        threshold: Correlation threshold.
        
    Returns:
        DataFrame with highly correlated features removed.
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
        logger.info(f"Dropping {len(to_drop)} highly correlated features: {to_drop[:5]}...")
        df_pd = df_pd.drop(columns=to_drop)
    
    return pl.from_pandas(df_pd) if isinstance(df, pl.DataFrame) else df_pd
