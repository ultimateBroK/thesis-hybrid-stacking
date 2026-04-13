"""Triple-Barrier label generation for XAU/USD trading - CORRECTED VERSION.

This version eliminates lookahead bias by using proper sequential logic.
The previous version had a subtle bias where it checked barriers sequentially
rather than simultaneously at each bar.

Classes:
    +1: Long (take profit hit first)
    0: Hold/No trade (neither hit within horizon)
    -1: Short (stop loss hit first)

Parameters (per config.toml):
    Fixed:    TP = Close + 1.5 × ATR, SL = Close - 1.5 × ATR
    Session:  TP/SL multipliers vary by market session (dead/active/overlap)
    Horizon = 20 bars
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm

from thesis.config.loader import Config, SessionATRConfig

logger = logging.getLogger("thesis.labels")


def _is_dst_ny(timestamp: datetime) -> bool:
    """Determine whether a UTC timestamp falls in US Daylight Saving Time.

    US DST rules: begins 2nd Sunday of March, ends 1st Sunday of November.
    Both transitions occur at 02:00 local time (07:00 / 06:00 UTC).

    Handles both timezone-aware and naive timestamps by stripping tz info
    before comparison with the calculated naive boundary datetimes.

    Args:
        timestamp: UTC datetime (aware or naive).

    Returns:
        ``True`` if DST is active (summer time), ``False`` otherwise.
    """
    # Strip timezone to allow comparison with naive boundary datetimes
    if timestamp.tzinfo is not None:
        timestamp = timestamp.replace(tzinfo=None)

    year = timestamp.year

    # 2nd Sunday of March
    march_first_weekday = datetime(year, 3, 1).weekday()  # Mon=0, Sun=6
    second_sunday = 8 + (6 - march_first_weekday) % 7
    dst_start = datetime(year, 3, second_sunday, 7, 0)  # 02:00 ET = 07:00 UTC

    # 1st Sunday of November
    nov_first_weekday = datetime(year, 11, 1).weekday()
    first_sunday = 1 + (6 - nov_first_weekday) % 7
    dst_end = datetime(year, 11, first_sunday, 6, 0)  # 02:00 ET = 06:00 UTC

    return dst_start <= timestamp < dst_end


def _get_session_atr_multiplier(
    utc_hour: int,
    session_config: SessionATRConfig,
    is_dst: bool,
) -> tuple[float, float, bool]:
    """Look up ATR multipliers for a given UTC hour and DST state.

    Args:
        utc_hour: Hour of day in UTC (0-23).
        session_config: Session ATR configuration with summer/winter definitions.
        is_dst: Whether DST is currently active.

    Returns:
        Tuple of ``(tp_mult, sl_mult, is_dead_hour)``.
    """
    sessions = session_config.summer if is_dst else session_config.winter

    for s in sessions:
        if s.start_utc <= s.end_utc:
            # Normal range (e.g. 6-12)
            if s.start_utc <= utc_hour < s.end_utc:
                return s.tp_mult, s.sl_mult, s.tp_mult == 0.0
        else:
            # Wrap-around midnight (e.g. 20-0 means 20-24 and 0)
            if utc_hour >= s.start_utc or utc_hour < s.end_utc:
                return s.tp_mult, s.sl_mult, s.tp_mult == 0.0

    # Fallback: use fixed multipliers (should not reach here if config is complete)
    return 1.5, 1.5, False


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
    session_atr = config.labels.session_atr

    logger.info("Triple-Barrier parameters:")
    if session_atr.enabled:
        logger.info("  Mode: session-aware ATR multipliers")
        logger.info(f"  Summer sessions: {len(session_atr.summer)}")
        logger.info(f"  Winter sessions: {len(session_atr.winter)}")
    else:
        logger.info(f"  TP: {atr_tp} × ATR (fixed)")
        logger.info(f"  SL: {atr_sl} × ATR (fixed)")
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
        session_atr=session_atr,
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
            "dead_hour": labels_data["dead_hours"],
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
    df: pl.DataFrame,
    atr_tp: float,
    atr_sl: float,
    horizon: int,
    min_atr: float,
    session_atr: SessionATRConfig | None = None,
) -> dict:
    """Generate labels with corrected logic to eliminate lookahead bias.

    Key correction: Uses bar-by-bar sequential checking instead of
    simultaneous high/low checks that could bias toward one direction.

    When ``session_atr`` is provided and enabled, ATR multipliers are
    looked up per-bar based on the UTC hour and US DST state.  Bars in
    dead hours (``tp_mult == 0.0``) are forced to label ``0`` with
    ``dead_hour = True``.

    Args:
        df: DataFrame with OHLCV and ATR data
        atr_tp: Fallback ATR multiplier for take profit
        atr_sl: Fallback ATR multiplier for stop loss
        horizon: Maximum bars to look ahead
        min_atr: Minimum ATR value to use
        session_atr: Optional session-aware ATR config

    Returns:
        Dictionary with labels, timestamps, barrier prices, touch bars,
        and dead_hour flags.
    """
    use_session = session_atr is not None and session_atr.enabled

    labels = []
    timestamps = []
    tp_prices = []
    sl_prices = []
    touched_bar = []
    dead_hours = []

    # Convert to numpy for faster processing
    close_arr = df["close"].to_numpy()
    high_arr = df["high"].to_numpy()
    low_arr = df["low"].to_numpy()
    atr_arr = df["atr_14"].to_numpy()
    time_arr = df["timestamp"].to_list()

    n = len(df)

    # Pre-compute DST and session lookups per-bar for performance
    session_tp = np.full(n, atr_tp)
    session_sl = np.full(n, atr_sl)
    is_dead = np.zeros(n, dtype=bool)

    if use_session:
        for i in range(n):
            ts = time_arr[i]
            # Ensure we have a datetime for DST check
            if hasattr(ts, "hour"):
                utc_hour = ts.hour
            else:
                utc_hour = 0
            is_dst = _is_dst_ny(ts) if hasattr(ts, "year") else False
            tp_m, sl_m, dead = _get_session_atr_multiplier(
                utc_hour, session_atr, is_dst
            )
            session_tp[i] = tp_m
            session_sl[i] = sl_m
            is_dead[i] = dead

        n_dead = int(is_dead.sum())
        logger.info(f"  Dead-hour bars: {n_dead:,} / {n:,} ({n_dead / n * 100:.1f}%)")

    logger.info(f"Processing {n:,} bars for label generation (corrected logic)...")

    for i in tqdm(range(n - horizon), desc="Generating labels", unit="bar"):
        # Current bar
        current_close = close_arr[i]
        current_atr = atr_arr[i]

        # Dead hour: force Hold label
        if use_session and is_dead[i]:
            labels.append(0)
            timestamps.append(time_arr[i])
            tp_prices.append(np.nan)
            sl_prices.append(np.nan)
            touched_bar.append(0)
            dead_hours.append(True)
            continue

        # Skip if ATR is too small
        if current_atr < min_atr:
            labels.append(0)
            timestamps.append(time_arr[i])
            tp_prices.append(np.nan)
            sl_prices.append(np.nan)
            touched_bar.append(0)
            dead_hours.append(False)
            continue

        # Select multipliers (session-aware or fixed fallback)
        tp_mult = session_tp[i] if use_session else atr_tp
        sl_mult = session_sl[i] if use_session else atr_sl

        # Calculate barriers at entry time (point-in-time)
        tp_price = current_close + tp_mult * current_atr
        sl_price = current_close - sl_mult * current_atr

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
                # Both touched on same bar - intra-bar path is unknown.
                # Assign Hold (0) to avoid inflating performance with
                # unknowable tie-breaking heuristics.
                label = 0
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
        dead_hours.append(False)

    # Handle last horizon bars (can't look ahead enough)
    for i in range(n - horizon, n):
        labels.append(0)  # Neutral for bars at end
        timestamps.append(time_arr[i])
        tp_prices.append(np.nan)
        sl_prices.append(np.nan)
        touched_bar.append(0)
        dead_hours.append(is_dead[i] if use_session else False)

    return {
        "labels": labels,
        "timestamps": timestamps,
        "tp_prices": tp_prices,
        "sl_prices": sl_prices,
        "touched_bar": touched_bar,
        "dead_hours": dead_hours,
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
