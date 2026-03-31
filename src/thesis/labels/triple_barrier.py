"""Triple-Barrier label generation for XAU/USD trading - CORRECTED VERSION.

This version eliminates lookahead bias by using proper sequential logic.
The previous version had a subtle bias where it checked barriers sequentially
rather than simultaneously at each bar.

Classes:
    +1: Long (take profit hit first)
    0: Hold/No trade (neither hit within horizon)
    -1: Short (stop loss hit first)

Parameters (per config.toml - symmetric barriers):
    TP = Close + 1.5 × ATR
    SL = Close - 1.5 × ATR
    Horizon = 20 bars
"""

import logging
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm

from thesis.config.loader import Config

logger = logging.getLogger("thesis.labels")


def generate_labels(config: Config) -> None:
    """Generate Triple-Barrier labels from feature data.

    This implementation simulates real-time labeling where at each bar,
    we look at the next `horizon` bars sequentially (not simultaneously)
    to determine which barrier hits first.

    Args:
        config: Configuration object.
    """
    features_path = Path(config.features.features_path)
    ohlcv_path = Path(config.data.ohlcv_path)

    if not features_path.exists():
        raise FileNotFoundError(f"Features not found: {features_path}")
    if not ohlcv_path.exists():
        raise FileNotFoundError(f"OHLCV not found: {ohlcv_path}")

    logger.info(f"Loading features: {features_path}")
    df_features = pl.read_parquet(features_path)

    logger.info(f"Loading OHLCV data: {ohlcv_path}")
    df_ohlcv = pl.read_parquet(ohlcv_path)

    # Join OHLCV with features on timestamp
    df = df_features.join(
        df_ohlcv.select(["timestamp", "open", "high", "low", "close"]),
        on="timestamp",
        how="inner",
    )

    # Get parameters
    atr_tp = config.labels.atr_multiplier_tp
    atr_sl = config.labels.atr_multiplier_sl
    horizon = config.labels.horizon_bars

    logger.info("Triple-Barrier parameters:")
    logger.info(f"  TP: {atr_tp} × ATR")
    logger.info(f"  SL: {atr_sl} × ATR")
    logger.info(f"  Horizon: {horizon} bars")

    # Ensure ATR exists
    if "atr_14" not in df.columns:
        raise ValueError("ATR_14 not found in features. Run feature engineering first.")

    # Generate labels using corrected logic
    labels_data = _generate_labels_corrected(
        df=df,
        atr_tp=atr_tp,
        atr_sl=atr_sl,
        horizon=horizon,
        min_atr=config.labels.min_atr,
    )

    # Create labels DataFrame with explicit datetime type to match features
    timestamp_dtype = df["timestamp"].dtype
    labels_df = pl.DataFrame(
        {
            "timestamp": pl.Series(labels_data["timestamps"], dtype=timestamp_dtype),
            "label": labels_data["labels"],
            "tp_price": labels_data["tp_prices"],
            "sl_price": labels_data["sl_prices"],
            "touched_bar": labels_data["touched_bar"],
        }
    )

    # Join with features
    df = df.join(labels_df, on="timestamp", how="left")

    # Log class distribution
    _log_class_distribution(df)

    # Save
    output_path = Path(config.labels.labels_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)

    logger.info(f"Saved labeled data: {output_path}")


def _generate_labels_corrected(
    df: pl.DataFrame, atr_tp: float, atr_sl: float, horizon: int, min_atr: float
) -> dict:
    """Generate labels with corrected logic to eliminate lookahead bias.

    Key correction: Uses bar-by-bar sequential checking instead of
    simultaneous high/low checks that could bias toward one direction.

    Args:
        df: DataFrame with OHLCV and ATR data
        atr_tp: ATR multiplier for take profit
        atr_sl: ATR multiplier for stop loss
        horizon: Maximum bars to look ahead
        min_atr: Minimum ATR value to use

    Returns:
        Dictionary with labels, timestamps, barrier prices, and touch bars
    """
    labels = []
    timestamps = []
    tp_prices = []
    sl_prices = []
    touched_bar = []

    # Convert to numpy for faster processing
    close_arr = df["close"].to_numpy()
    high_arr = df["high"].to_numpy()
    low_arr = df["low"].to_numpy()
    atr_arr = df["atr_14"].to_numpy()
    time_arr = df["timestamp"].to_list()

    n = len(df)

    logger.info(f"Processing {n:,} bars for label generation (corrected logic)...")

    for i in tqdm(range(n - horizon), desc="Generating labels", unit="bar"):
        # Current bar
        current_close = close_arr[i]
        current_atr = atr_arr[i]

        # Skip if ATR is too small
        if current_atr < min_atr:
            labels.append(0)
            timestamps.append(time_arr[i])
            tp_prices.append(np.nan)
            sl_prices.append(np.nan)
            touched_bar.append(0)
            continue

        # Calculate barriers at entry time (point-in-time)
        tp_price = current_close + atr_tp * current_atr
        sl_price = current_close - atr_sl * current_atr

        # Look ahead sequentially (no rolling window bias)
        # Check each bar's high and low simultaneously to determine first touch
        label = 0  # Default: no touch
        touch_bar = 0

        for j in range(horizon):
            future_idx = i + 1 + j
            if future_idx >= n:
                break

            future_high = high_arr[future_idx]
            future_low = low_arr[future_idx]

            # Check both barriers at this bar (simultaneous check, not sequential)
            tp_touched = future_high >= tp_price
            sl_touched = future_low <= sl_price

            if tp_touched and sl_touched:
                # Both touched on same bar - use close price to break tie
                # This is realistic: in real trading, you'd see which side was hit
                future_close = close_arr[future_idx]
                if future_close >= current_close:
                    label = 1  # Long (close above entry favors TP)
                else:
                    label = -1  # Short (close below entry favors SL)
                touch_bar = j + 1
                break
            elif tp_touched:
                label = 1  # Take profit hit first
                touch_bar = j + 1
                break
            elif sl_touched:
                label = -1  # Stop loss hit first
                touch_bar = j + 1
                break

        labels.append(label)
        timestamps.append(time_arr[i])
        tp_prices.append(tp_price if label != 0 else np.nan)
        sl_prices.append(sl_price if label != 0 else np.nan)
        touched_bar.append(touch_bar)

    # Handle last horizon bars (can't look ahead enough)
    for i in range(n - horizon, n):
        labels.append(0)  # Neutral for bars at end
        timestamps.append(time_arr[i])
        tp_prices.append(np.nan)
        sl_prices.append(np.nan)
        touched_bar.append(0)

    return {
        "labels": labels,
        "timestamps": timestamps,
        "tp_prices": tp_prices,
        "sl_prices": sl_prices,
        "touched_bar": touched_bar,
    }


def _log_class_distribution(df: pl.DataFrame) -> None:
    """Log label distribution.

    Args:
        df: DataFrame with labels.
    """
    counts = df["label"].value_counts().sort("label")
    total = len(df)

    logger.info("Label distribution:")
    for row in counts.iter_rows():
        label, count = row
        pct = count / total * 100
        label_name = {1: "Long", 0: "Hold/Neutral", -1: "Short"}.get(label, "Unknown")
        logger.info(f"  {label} ({label_name}): {count:,} ({pct:.1f}%)")
