"""Tick data to OHLCV H1 converter.

Converts Dukascopy tick data to H1 candles with proper timezone handling.
NY market close at 17:00 DST-aware (America/New_York).
"""

import logging
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm

from thesis.config.loader import Config

logger = logging.getLogger("thesis.data")


def validate_ohlcv_quality(df: pl.DataFrame) -> dict:
    """Validate OHLCV data quality and return diagnostic report.

    Checks for:
    - Missing bars (gaps in time series)
    - Zero volume hours
    - Price gaps (suspicious jumps)
    - Duplicate timestamps
    - OHLC logic violations (high < low, close outside range, etc.)
    - Weekend data (should be minimal for forex)

    Args:
        df: OHLCV DataFrame with timestamp column.

    Returns:
        Dictionary with quality metrics and flags.
    """
    logger.info("Running OHLCV data quality validation...")

    issues = []
    warnings = []

    # Check 1: Missing bars (gaps)
    timestamps = df["timestamp"].to_list()
    expected_interval = 3600  # 1 hour in seconds

    gaps = []
    for i in range(1, len(timestamps)):
        gap_seconds = (timestamps[i] - timestamps[i - 1]).total_seconds()
        if gap_seconds > expected_interval * 1.5:  # Allow 50% tolerance
            gaps.append(
                {
                    "from": timestamps[i - 1],
                    "to": timestamps[i],
                    "hours_missing": gap_seconds / 3600,
                }
            )

    if gaps:
        issues.append(f"Found {len(gaps)} gaps in data (missing bars)")
        logger.warning(f"  Data gaps detected: {len(gaps)} instances")
        for gap in gaps[:3]:  # Show first 3
            logger.warning(
                f"    Gap: {gap['from']} → {gap['to']} ({gap['hours_missing']:.1f}h)"
            )

    # Check 2: Zero volume bars
    zero_volume = (df["volume"] == 0).sum()
    zero_volume_pct = zero_volume / len(df) * 100
    if zero_volume > 0:
        warnings.append(f"{zero_volume} bars with zero volume ({zero_volume_pct:.1f}%)")
        logger.warning(f"  Zero volume bars: {zero_volume} ({zero_volume_pct:.1f}%)")

    # Check 3: Price gaps (suspicious jumps > 5% in one hour)
    closes = df["close"].to_numpy()
    price_changes = np.abs(np.diff(closes) / closes[:-1])
    large_gaps = price_changes > 0.05  # 5% threshold

    if large_gaps.sum() > 0:
        issues.append(f"{large_gaps.sum()} suspicious price jumps > 5% detected")
        logger.warning(f"  Large price gaps: {large_gaps.sum()} instances")

    # Check 4: Duplicate timestamps
    unique_ts = df["timestamp"].n_unique()
    total_ts = len(df)
    if unique_ts < total_ts:
        duplicates = total_ts - unique_ts
        issues.append(f"{duplicates} duplicate timestamps found")
        logger.error(f"  Duplicate timestamps: {duplicates}")

    # Check 5: OHLC logic violations
    ohlc_issues = 0

    # High should be >= Low
    hl_violations = (df["high"] < df["low"]).sum()
    if hl_violations > 0:
        ohlc_issues += hl_violations
        issues.append(f"{hl_violations} bars where high < low")

    # Close should be between high and low
    close_violations = ((df["close"] > df["high"]) | (df["close"] < df["low"])).sum()
    if close_violations > 0:
        ohlc_issues += close_violations
        issues.append(f"{close_violations} bars where close outside high-low range")

    # Open should be between high and low
    open_violations = ((df["open"] > df["high"]) | (df["open"] < df["low"])).sum()
    if open_violations > 0:
        ohlc_issues += open_violations
        issues.append(f"{open_violations} bars where open outside high-low range")

    if ohlc_issues > 0:
        logger.error(f"  OHLC logic violations: {ohlc_issues}")

    # Check 6: Weekend data (Saturday/Sunday)
    df_with_dow = df.with_columns(pl.col("timestamp").dt.weekday().alias("day_of_week"))
    weekend_bars = (df_with_dow["day_of_week"] > 5).sum()
    weekend_pct = weekend_bars / len(df) * 100
    if weekend_bars > 0:
        warnings.append(
            f"{weekend_bars} weekend bars ({weekend_pct:.1f}%) - expected minimal for forex"
        )
        logger.info(f"  Weekend bars: {weekend_bars} ({weekend_pct:.1f}%)")

    # Summary
    quality_score = 100
    quality_score -= len(issues) * 10  # -10 for each issue
    quality_score -= len(warnings) * 5  # -5 for each warning
    quality_score = max(0, quality_score)

    result = {
        "total_bars": len(df),
        "date_range": (df["timestamp"].min(), df["timestamp"].max()),
        "gaps_detected": len(gaps),
        "zero_volume_bars": int(zero_volume),
        "zero_volume_pct": zero_volume_pct,
        "large_price_gaps": int(large_gaps.sum()),
        "duplicate_timestamps": total_ts - unique_ts,
        "ohlc_violations": ohlc_issues,
        "weekend_bars": int(weekend_bars),
        "issues": issues,
        "warnings": warnings,
        "quality_score": quality_score,
        "is_valid": len(issues) == 0 and quality_score >= 80,
    }

    logger.info(f"Quality validation complete. Score: {quality_score}/100")
    if result["is_valid"]:
        logger.info("  Data quality: GOOD")
    else:
        logger.warning("  Data quality: NEEDS REVIEW")

    return result


def process_all_tick_files(config: Config) -> None:
    """Process all raw tick files and save to single OHLCV parquet.

    Args:
        config: Configuration object.
    """
    raw_path = Path(config.data.raw_data_path)
    output_path = Path(config.data.ohlcv_path)

    # Find all parquet files in raw directory
    tick_files = sorted(raw_path.glob("*.parquet"))

    if not tick_files:
        raise FileNotFoundError(f"No parquet files found in {raw_path}")

    logger.info(f"Found {len(tick_files)} tick files to process")

    # Process all files
    ohlcv_data = []

    for tick_file in tqdm(tick_files, desc="Processing tick files", unit="file"):
        try:
            ohlcv_chunk = _process_tick_file(tick_file, config)
            if ohlcv_chunk is not None and len(ohlcv_chunk) > 0:
                ohlcv_data.append(ohlcv_chunk)
        except Exception as e:
            logger.warning(f"Failed to process {tick_file.name}: {e}")
            continue

    if not ohlcv_data:
        raise ValueError("No valid data processed from tick files")

    # Concatenate all chunks
    logger.info("Combining all OHLCV data...")
    full_ohlcv = pl.concat(ohlcv_data, how="vertical")

    # Sort by timestamp
    full_ohlcv = full_ohlcv.sort("timestamp")

    # Remove duplicates if any
    full_ohlcv = full_ohlcv.unique(subset=["timestamp"], keep="first")

    # NEW: Validate data quality
    quality_report = validate_ohlcv_quality(full_ohlcv)

    if not quality_report["is_valid"]:
        logger.warning("⚠️ Data quality issues detected - review recommended")
        for issue in quality_report["issues"]:
            logger.warning(f"  Issue: {issue}")

    # Save to parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    full_ohlcv.write_parquet(output_path)

    logger.info(f"Saved OHLCV data: {len(full_ohlcv):,} candles")
    logger.info(
        f"Date range: {full_ohlcv['timestamp'].min()} → {full_ohlcv['timestamp'].max()}"
    )
    logger.info(f"Quality score: {quality_report['quality_score']}/100")


def _process_tick_file(file_path: Path, config: Config) -> pl.DataFrame | None:
    """Process a single tick file to OHLCV H1.

    Args:
        file_path: Path to tick parquet file.
        config: Configuration object.

    Returns:
        OHLCV DataFrame or None if error.
    """
    # Read tick data
    ticks = pl.read_parquet(file_path)

    # Calculate mid price
    ticks = ticks.with_columns(
        [
            ((pl.col("ask") + pl.col("bid")) / 2).alias("mid"),
            (pl.col("ask") - pl.col("bid")).alias("spread"),
        ]
    )

    # Convert timestamp to datetime (UTC-like, no tz info from Dukascopy)
    ticks = ticks.with_columns(pl.col("timestamp").cast(pl.Datetime("ms")))

    # Convert to market timezone for proper day roll
    ticks = ticks.with_columns(
        pl.col("timestamp")
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone(config.data.market_tz)
        .alias("market_time")
    )

    # Aggregate to H1
    ohlcv = _aggregate_to_h1(ticks, config)

    return ohlcv


def _aggregate_to_h1(ticks: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Aggregate tick data to H1 candles.

    Args:
        ticks: Tick DataFrame with market_time column.
        config: Configuration object.

    Returns:
        H1 OHLCV DataFrame.
    """
    # Floor to hour
    ticks = ticks.with_columns(pl.col("market_time").dt.truncate("1h").alias("hour"))

    # Aggregate
    ohlcv = ticks.group_by("hour").agg(
        [
            pl.col("mid").first().alias("open"),
            pl.col("mid").max().alias("high"),
            pl.col("mid").min().alias("low"),
            pl.col("mid").last().alias("close"),
            pl.col("ask_volume").sum().alias("ask_volume"),
            pl.col("bid_volume").sum().alias("bid_volume"),
            pl.col("spread").mean().alias("avg_spread"),
            pl.len().alias("tick_count"),
        ]
    )

    # Sort by hour
    ohlcv = ohlcv.sort("hour")

    # Rename hour to timestamp and convert back to UTC for consistency
    ohlcv = ohlcv.rename({"hour": "timestamp"})

    # Add total volume
    ohlcv = ohlcv.with_columns(
        [(pl.col("ask_volume") + pl.col("bid_volume")).alias("volume")]
    )

    # Drop separate ask/bid volume (keep total)
    ohlcv = ohlcv.drop(["ask_volume", "bid_volume"])

    # Ensure proper column order
    ohlcv = ohlcv.select(
        [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "avg_spread",
            "tick_count",
        ]
    )

    return ohlcv


def load_ohlcv(config: Config) -> pl.DataFrame:
    """Load cached OHLCV data.

    Args:
        config: Configuration object.

    Returns:
        OHLCV DataFrame.
    """
    ohlcv_path = Path(config.data.ohlcv_path)

    if not ohlcv_path.exists():
        raise FileNotFoundError(
            f"OHLCV data not found: {ohlcv_path}. Run data pipeline first."
        )

    return pl.read_parquet(ohlcv_path)
