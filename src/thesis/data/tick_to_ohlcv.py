"""Tick data to OHLCV H1 converter.

Converts Dukascopy tick data to H1 candles with proper timezone handling.
NY market close at 17:00 DST-aware (America/New_York).
"""

import logging
from pathlib import Path

import polars as pl
from tqdm import tqdm

from thesis.config.loader import Config

logger = logging.getLogger("thesis.data")


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
    
    # Save to parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    full_ohlcv.write_parquet(output_path)
    
    logger.info(f"Saved OHLCV data: {len(full_ohlcv):,} candles")
    logger.info(f"Date range: {full_ohlcv['timestamp'].min()} → {full_ohlcv['timestamp'].max()}")


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
    ticks = ticks.with_columns([
        ((pl.col("ask") + pl.col("bid")) / 2).alias("mid"),
        (pl.col("ask") - pl.col("bid")).alias("spread"),
    ])
    
    # Convert timestamp to datetime (UTC-like, no tz info from Dukascopy)
    ticks = ticks.with_columns(
        pl.col("timestamp").cast(pl.Datetime("ms"))
    )
    
    # Convert to market timezone for proper day roll
    ticks = ticks.with_columns(
        pl.col("timestamp").dt.replace_time_zone("UTC")
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
    ticks = ticks.with_columns(
        pl.col("market_time").dt.truncate("1h").alias("hour")
    )
    
    # Aggregate
    ohlcv = ticks.group_by("hour").agg([
        pl.col("mid").first().alias("open"),
        pl.col("mid").max().alias("high"),
        pl.col("mid").min().alias("low"),
        pl.col("mid").last().alias("close"),
        pl.col("ask_volume").sum().alias("ask_volume"),
        pl.col("bid_volume").sum().alias("bid_volume"),
        pl.col("spread").mean().alias("avg_spread"),
        pl.len().alias("tick_count"),
    ])
    
    # Sort by hour
    ohlcv = ohlcv.sort("hour")
    
    # Rename hour to timestamp and convert back to UTC for consistency
    ohlcv = ohlcv.rename({"hour": "timestamp"})
    
    # Add total volume
    ohlcv = ohlcv.with_columns([
        (pl.col("ask_volume") + pl.col("bid_volume")).alias("volume")
    ])
    
    # Drop separate ask/bid volume (keep total)
    ohlcv = ohlcv.drop(["ask_volume", "bid_volume"])
    
    # Ensure proper column order
    ohlcv = ohlcv.select([
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "avg_spread",
        "tick_count",
    ])
    
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
            f"OHLCV data not found: {ohlcv_path}. "
            "Run data pipeline first."
        )
    
    return pl.read_parquet(ohlcv_path)
