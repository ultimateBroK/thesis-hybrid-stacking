"""Data preparation — aggregate raw tick data to OHLCV bars.

Reads monthly tick parquet files from data/raw/XAUUSD/, computes mid-price
OHLCV bars at the configured timeframe, and saves to data/processed/ohlcv.parquet.

Memory-efficient: aggregates each monthly file independently, then concats
only the small OHLCV results (~56K rows for 8 years of 1H bars).
"""

import logging
from pathlib import Path

import polars as pl

from thesis.config import Config

logger = logging.getLogger("thesis.prepare")


def _aggregate_file(file_path: Path, group_ms: int) -> pl.DataFrame:
    """
    Aggregate a monthly tick Parquet file into OHLCV bars aligned to the specified millisecond bar size.

    Parameters:
        file_path (Path): Path to a monthly tick Parquet file containing at least the columns `timestamp`, `bid`, `ask`, `ask_volume`, and `bid_volume`.
        group_ms (int): Bar size in milliseconds used to floor tick timestamps to bar boundaries.

    Returns:
        pl.DataFrame: DataFrame with columns `timestamp`, `open`, `high`, `low`, `close`, `volume`, `tick_count`, and `avg_spread`. Rows correspond to bars whose timestamps are the bar boundary datetimes; ticks with years outside 2000–2100 are excluded.
    """
    ticks = pl.read_parquet(
        file_path,
        columns=["timestamp", "bid", "ask", "ask_volume", "bid_volume"],
    )

    # Compute mid-price and total volume
    ticks = ticks.with_columns(
        [
            (
                (pl.col("ask") * pl.col("bid_volume") + pl.col("bid") * pl.col("ask_volume"))
                / (pl.col("ask_volume") + pl.col("bid_volume") + 1e-10)
            ).alias("mid"),
            (pl.col("ask_volume") + pl.col("bid_volume")).alias("volume"),
        ]
    )

    # Filter out corrupted timestamps (year must be 2000-2100)
    ticks = ticks.filter(
        (pl.col("timestamp").dt.year() >= 2000)
        & (pl.col("timestamp").dt.year() <= 2100)
    )

    # Floor timestamps to bar boundaries
    ts_ms = ticks["timestamp"].dt.timestamp("ms")
    bar_group = (ts_ms // group_ms) * group_ms

    ticks = ticks.with_columns(
        [
            bar_group.cast(pl.Datetime("ms")).alias("bar_time"),
        ]
    )

    # Sort ticks by bar_time then timestamp before aggregation
    # so first()/last() within each bar give deterministic open/close
    ticks = ticks.sort(["bar_time", "timestamp"])

    # Aggregate to OHLCV
    ohlcv = (
        ticks.group_by("bar_time", maintain_order=True)
        .agg(
            [
                pl.col("mid").first().alias("open"),
                pl.col("mid").max().alias("high"),
                pl.col("mid").min().alias("low"),
                pl.col("mid").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
                pl.col("mid").count().alias("tick_count"),
                ((pl.col("ask") - pl.col("bid")).mean()).alias("avg_spread"),
            ]
        )
        .rename({"bar_time": "timestamp"})
    )

    return ohlcv


def prepare_data(config: Config) -> None:
    """
    Prepare OHLCV bars from raw monthly tick parquet files and write the resulting parquet to the configured output path.

    Reads all parquet files under config.paths.data_raw, aggregates ticks into OHLCV bars using the timeframe in config.data.timeframe, concatenates and deduplicates monthly results, filters bars with years outside 2000–2100, and writes the final dataset to config.paths.ohlcv.

    Parameters:
        config (Config): Application configuration. `config.data.timeframe` controls bar size and accepts hourly (e.g., "1H", "4H"), minute (e.g., "1MIN", "5M"), or daily ("D" or "1D") formats.

    Raises:
        FileNotFoundError: If the raw data directory is missing or contains no parquet files (unless the output already exists, in which case the function returns).
        ValueError: If `config.data.timeframe` uses an unsupported format.
    """
    raw_dir = Path(config.paths.data_raw)
    ohlcv_path = Path(config.paths.ohlcv)

    if not raw_dir.exists():
        if ohlcv_path.exists():
            logger.warning(
                "Raw data dir missing (%s) but OHLCV exists (%s) — skipping prepare.",
                raw_dir,
                ohlcv_path,
            )
            return
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    parquet_files = sorted(raw_dir.glob("*.parquet"))
    if not parquet_files:
        if ohlcv_path.exists():
            logger.warning(
                "No parquet files found (%s) but OHLCV exists (%s) — skipping prepare.",
                raw_dir,
                ohlcv_path,
            )
            return
        raise FileNotFoundError(f"No parquet files in {raw_dir}")

    logger.info("Found %d tick files in %s", len(parquet_files), raw_dir)

    # Determine timeframe in milliseconds for grouping
    tf = config.data.timeframe.upper()
    if tf.endswith("H"):
        hours = int(tf[:-1])
        if hours <= 0:
            raise ValueError(f"Invalid timeframe '{tf}': hours must be > 0")
        group_ms = hours * 3_600_000
    elif tf.endswith("MIN") or tf.endswith("M"):
        minutes = int(tf.replace("MIN", "").replace("M", ""))
        if minutes <= 0:
            raise ValueError(f"Invalid timeframe '{tf}': minutes must be > 0")
        group_ms = minutes * 60_000
    elif tf in ("D", "1D"):
        group_ms = 86_400_000
    else:
        raise ValueError(f"Unsupported timeframe: {tf}")

    # Aggregate each monthly file separately — memory-efficient
    monthly_bars: list[pl.DataFrame] = []
    for f in parquet_files:
        logger.info("  Processing %s", f.name)
        bars = _aggregate_file(f, group_ms)
        monthly_bars.append(bars)

    # Concat small OHLCV DataFrames (tiny compared to ticks)
    ohlcv = pl.concat(monthly_bars, how="vertical").sort("timestamp")

    # Remove duplicate bar timestamps (boundary overlap between months)
    ohlcv = ohlcv.unique(subset=["timestamp"], keep="first").sort("timestamp")

    # Filter out bars with corrupted timestamps (year outside 2000-2100)
    n_before = len(ohlcv)
    ohlcv = ohlcv.filter(
        (pl.col("timestamp").dt.year() >= 2000)
        & (pl.col("timestamp").dt.year() <= 2100)
    )
    n_after = len(ohlcv)
    if n_before != n_after:
        logger.warning("Dropped %d bars with corrupted timestamps", n_before - n_after)

    logger.info("OHLCV bars: %d (timeframe=%s)", len(ohlcv), config.data.timeframe)
    logger.info(
        "Date range: %s to %s",
        ohlcv["timestamp"].min(),
        ohlcv["timestamp"].max(),
    )

    # Save
    ohlcv_path.parent.mkdir(parents=True, exist_ok=True)
    ohlcv.write_parquet(ohlcv_path)
    logger.info("Saved OHLCV: %s", ohlcv_path)
