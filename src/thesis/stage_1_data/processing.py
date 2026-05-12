"""Data preparation stage — aggregate raw tick data to OHLCV bars.

Reads monthly tick parquet files from ``data/raw/XAUUSD/``, computes quote
microprice OHLCV bars at the configured timeframe, and saves to
``data/processed/ohlcv.parquet`` alongside a data-quality JSON sidecar.

Memory-efficient: aggregates each monthly file independently, then concats
only the small OHLCV results (~56K rows for 8 years of 1H bars).

Args:
    config: Runtime configuration containing data source paths, timeframe,
        and date-range bounds.

Returns:
    None. Writes ``ohlcv.parquet`` and ``data_quality.json`` to disk.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import polars as pl

from thesis.shared.config import Config
from thesis.shared.constants import FEATURE_EPS
from thesis.shared.constants import timeframe_to_ms as _timeframe_to_ms
from thesis.shared.data_quality import (
    check_candle_quality,
    check_gap_report,
    classify_calendar_gaps,
)

logger = logging.getLogger("thesis.prepare")


def _parse_datetime_bound(
    value: str, name: str, dtype: pl.DataType, tz: str
) -> pl.Expr:
    """Parse a config datetime string into a timezone-aware Polars expression.

    Args:
        value: Datetime string from config (e.g. "2021-01-01" or
            "2021-01-01T23:59:59").
        name: Config key name used in error messages (e.g. "start_date").
        dtype: Target Polars dtype for the resulting expression.
        tz: Timezone string (e.g. "America/New_York") to localise the value.

    Returns:
        A Polars expression that filters timestamps inclusively.

    Note:
        If ``value`` contains only a date with no time component, the time
        is set to ``T23:59:59`` so that ``<=`` filtering covers the entire day.
    """
    if not value:
        raise ValueError(f"config.data.{name} must not be empty")
    normalized = value.strip()
    if "T" not in normalized and " " not in normalized and ":" not in normalized:
        normalized = normalized + "T23:59:59"
    return pl.lit(normalized).str.to_datetime(time_unit="us", time_zone=tz).cast(dtype)


def _aggregate_file(file_path: Path, group_every: str) -> pl.DataFrame:
    """Aggregate one monthly tick parquet file into OHLCV bars.

    Args:
        file_path: Path to a single monthly ``.parquet`` tick file.
        group_every: Polars group-by-dynamic interval string (e.g. "1h").


    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume,
        tick_count, avg_spread. Bars with all-null microprice are dropped.
    """
    ticks = pl.read_parquet(
        file_path,
        columns=["timestamp", "bid", "ask", "ask_volume", "bid_volume"],
    )

    n_before = len(ticks)
    ticks = ticks.filter(
        (pl.col("bid") > 0)
        & (pl.col("ask") > 0)
        & (pl.col("ask") >= pl.col("bid"))
        & (pl.col("ask_volume") >= 0)
        & (pl.col("bid_volume") >= 0)
    )
    dropped_quotes = n_before - len(ticks)
    if dropped_quotes > 0:
        logger.warning(
            "%s: dropped %d invalid quote ticks (bid/ask/spread/volume)",
            file_path.name,
            dropped_quotes,
        )

    # Microprice: volume-weighted by opposing-side quote sizes (not midpoint).
    ticks = ticks.with_columns(
        (
            (
                pl.col("ask") * pl.col("bid_volume")
                + pl.col("bid") * pl.col("ask_volume")
            )
            / (pl.col("ask_volume") + pl.col("bid_volume") + FEATURE_EPS)
        ).alias("microprice"),
        (pl.col("ask_volume") + pl.col("bid_volume")).alias("volume"),
    )

    ticks = ticks.sort("timestamp")

    agg_df = ticks.group_by_dynamic(
        "timestamp",
        every=group_every,
        period=group_every,
        closed="left",
        label="left",
        start_by="window",
    ).agg(
        pl.col("microprice").first().alias("open"),
        pl.col("microprice").max().alias("high"),
        pl.col("microprice").min().alias("low"),
        pl.col("microprice").last().alias("close"),
        pl.col("volume").sum().alias("volume"),
        pl.col("microprice").count().alias("tick_count"),
        (pl.col("ask") - pl.col("bid")).mean().alias("avg_spread"),
    )
    agg_df = agg_df.drop_nulls()

    # Clip to nominal month boundaries — last-day-of-prev-month ticks in each file
    # produce bars that belong to the next month; drop them to avoid duplicates.
    stem = file_path.stem  # e.g. "2021-01"
    year, month = int(stem[:4]), int(stem[5:7])
    month_start = pl.datetime(year, month, 1)
    if month == 12:
        month_end = pl.datetime(year + 1, 1, 1) - pl.duration(seconds=1)
    else:
        month_end = pl.datetime(year, month + 1, 1) - pl.duration(seconds=1)
    agg_df = agg_df.filter(
        (pl.col("timestamp") >= month_start) & (pl.col("timestamp") <= month_end)
    )
    return agg_df


def _deduplicate_and_filter(ohlcv: pl.DataFrame) -> tuple[pl.DataFrame, int, int]:
    """Deduplicate OHLCV bars and filter rows with corrupted timestamps.

    Args:
        ohlcv: DataFrame with a ``timestamp`` column.

    Returns:
        A 3-tuple ``(filtered_df, dropped_count, duplicate_count)`` where:

        - ``filtered_df``: deduplicated DataFrame with year 2000–2100 filter applied
        - ``dropped_count``: number of rows removed by the year filter
        - ``duplicate_count``: number of duplicate timestamps found before dedup
    """
    duplicate_count = len(ohlcv) - ohlcv.get_column("timestamp").n_unique()
    if duplicate_count > 0:
        logger.warning(
            "Found %d duplicate OHLCV bar timestamps before dedup; keeping first. "
            "Likely overlapping raw files; downstream stages now validate uniqueness.",
            duplicate_count,
        )
    ohlcv = ohlcv.unique(subset=["timestamp"], keep="first").sort("timestamp")
    n_before = len(ohlcv)
    ohlcv = ohlcv.filter(
        (pl.col("timestamp").dt.year() >= 2000)
        & (pl.col("timestamp").dt.year() <= 2100)
    )
    dropped = n_before - len(ohlcv)
    if dropped > 0:
        logger.warning("Dropped %d bars with corrupted timestamps", dropped)
    return ohlcv, dropped, duplicate_count


def _filter_date_range(ohlcv: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Apply the configured inclusive date bounds to OHLCV bars.

    Args:
        ohlcv: DataFrame with a ``timestamp`` column.
        config: Runtime configuration providing start/end dates and timezone.

    Returns:
        DataFrame containing only bars within the configured date range.


    Raises:
        ValueError: If no bars remain after applying the date filter.
    """
    n_before = len(ohlcv)
    ts_dtype = ohlcv["timestamp"].dtype
    market_tz = config.data.market_tz
    start = _parse_datetime_bound(
        config.data_range.start, "start_date", ts_dtype, market_tz
    )
    end = _parse_datetime_bound(        config.data_range.end, "end_date", ts_dtype, market_tz)
    ohlcv = ohlcv.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") <= end))
    dropped = n_before - len(ohlcv)
    if dropped > 0:
        logger.info(
            "Dropped %d bars outside configured range [%s, %s]",
            dropped,
            config.data_range.start,
            config.data_range.end,
        )
    if ohlcv.is_empty():
        raise ValueError(
            "No OHLCV bars remain after applying configured data date range "
            f"[{config.data.start_date}, {config.data.end_date}]"
        )
    return ohlcv


def _log_gap_report(ohlcv: pl.DataFrame, group_ms: int) -> None:
    """Log timestamp continuity diagnostics via shared data-quality checks.

    Args:
        ohlcv: DataFrame with a ``timestamp`` column.
        group_ms: Expected bar duration in milliseconds (from timeframe).
    """
    if len(ohlcv) < 2:
        logger.warning("OHLCV gap report skipped: fewer than 2 bars")
        return

    result = check_gap_report(ohlcv, group_ms)

    diffs = (
        ohlcv.select(
            (pl.col("timestamp").diff().dt.total_milliseconds()).alias("delta_ms")
        )
        .drop_nulls()
        .get_column("delta_ms")
    )
    non_increasing = int((diffs <= 0).sum())

    logger.info(
        "OHLCV calendar gap report: expected_delta=%d ms, calendar_gap_count=%d, "
        "estimated_missing_bars=%d, largest_gap=%.2f bars, non_increasing_deltas=%d",
        group_ms,
        result["gap_count"],
        result["estimated_missing_bars"],
        result["largest_gap_bars"],
        non_increasing,
    )


def _log_candle_quality_report(ohlcv: pl.DataFrame) -> None:
    """Log OHLCV candle integrity and spread statistics.

    Args:
        ohlcv: DataFrame with columns open, high, low, close, avg_spread,
            tick_count.
    """
    if ohlcv.is_empty():
        return

    result = check_candle_quality(ohlcv)
    if result["invalid_count"] > 0:
        logger.warning(
            "OHLCV quality: %d invalid candles detected",
            result["invalid_count"],
        )

    stats = ohlcv.select(
        (pl.col("high") - pl.col("low")).median().alias("median_range"),
        (pl.col("high") - pl.col("low")).quantile(0.99).alias("p99_range"),
        pl.col("avg_spread").median().alias("median_spread"),
        pl.col("avg_spread").quantile(0.99).alias("p99_spread"),
        pl.col("tick_count").quantile(0.01).alias("p01_tick_count"),
    ).row(0, named=True)
    logger.info(
        "OHLCV quality: median_range=%.6f, p99_range=%.6f, "
        "median_spread=%.6f, p99_spread=%.6f, p01_tick_count=%.1f",
        stats["median_range"] or 0.0,
        stats["p99_range"] or 0.0,
        stats["median_spread"] or 0.0,
        stats["p99_spread"] or 0.0,
        stats["p01_tick_count"] or 0.0,
    )


def _compute_data_quality_stats(
    ohlcv: pl.DataFrame,
    group_ms: int,
    deduped_timestamps: int,
) -> dict:
    """Compute data-quality summary statistics for the OHLCV output.

    Args:
        ohlcv: DataFrame with a ``timestamp`` column.
        group_ms: Expected bar duration in milliseconds (from timeframe).
        deduped_timestamps: Number of duplicate timestamps removed before this step.

    Returns:
        Dictionary with keys: total_bars, deduped_timestamps, start_date,
        end_date, calendar_gaps, weekend_gaps, real_gaps,
        estimated_missing_bars, largest_gap_bars, calendar_warnings.
    """
    total_bars = len(ohlcv)
    start_date = str(ohlcv["timestamp"].min())
    end_date = str(ohlcv["timestamp"].max())

    base = {
        "total_bars": total_bars,
        "deduped_timestamps": deduped_timestamps,
        "start_date": start_date,
        "end_date": end_date,
        "calendar_gaps": 0,
        "weekend_gaps": 0,
        "real_gaps": 0,
        "estimated_missing_bars": 0,
        "largest_gap_bars": 0,
    }

    if total_bars < 2:
        return base

    gap_summary = classify_calendar_gaps(ohlcv, group_ms)
    base.update(
        calendar_gaps=gap_summary.calendar_gap_count + gap_summary.real_gap_count,
        weekend_gaps=gap_summary.calendar_gap_count,
        real_gaps=gap_summary.real_gap_count,
        estimated_missing_bars=gap_summary.estimated_missing_bars,
        largest_gap_bars=gap_summary.largest_gap_bars,
        calendar_warnings=gap_summary.warnings,
    )
    return base


def _save_data_quality_json(stats: dict, config: Config) -> None:
    """Write data-quality statistics to the JSON sidecar file.

    Args:
        stats: Data quality dictionary from ``_compute_data_quality_stats``.
        config: Runtime configuration providing the output file path.
    """
    dq_path = Path(config.paths.data_quality_json)
    dq_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dq_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info(
        "Data quality JSON saved: %s (total_bars=%d)",
        dq_path,
        stats["total_bars"],
    )


def generate_data(config: Config) -> None:
    """Build OHLCV bars from raw monthly tick files and persist parquet + JSON stats.

    Pipeline overview:

    1. Validate raw data directory and discover ``.parquet`` files.
    2. Aggregate each monthly tick file into OHLCV bars using microprice.
    3. Concatenate monthly results, deduplicate, and filter by date range.
    4. Log gap and candle-quality diagnostics.
    5. Persist ``ohlcv.parquet`` and ``data_quality.json``.


    Args:
        config: Runtime configuration. Must contain ``data.raw_dir``,
            ``data.ohlcv_path``, ``data.timeframe``, ``data.start_date``,
            ``data.end_date``, ``data.market_tz``, and ``workflow.force_rerun``.

    Raises:
        FileNotFoundError: If the raw data directory does not exist and no
            cached OHLCV file is present.
    """
    raw_dir = Path(config.paths.data_raw)
    ohlcv_path = Path(config.paths.ohlcv)

    if config.workflow.force_rerun and ohlcv_path.exists():
        ohlcv_path.unlink()
        logger.info("force_rerun=True — removed existing OHLCV, will rebuild.")

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

    group_ms = _timeframe_to_ms(config.data.timeframe)
    group_every = config.data.timeframe.lower()

    monthly_bars: list[pl.DataFrame] = []
    total_files = len(parquet_files)
    for idx, file_path in enumerate(parquet_files, start=1):
        logger.info(
            "Aggregating monthly file %d/%d: %s",
            idx,
            total_files,
            file_path.name,
        )
        monthly_bars.append(_aggregate_file(file_path, group_every))

    ohlcv = pl.concat(monthly_bars, how="vertical").sort("timestamp")

    ohlcv, _, deduped_count = _deduplicate_and_filter(ohlcv)
    ohlcv = _filter_date_range(ohlcv, config)
    _log_gap_report(ohlcv, group_ms)
    _log_candle_quality_report(ohlcv)

    dq_stats = _compute_data_quality_stats(ohlcv, group_ms, deduped_count)
    _save_data_quality_json(dq_stats, config)

    logger.info("OHLCV bars: %d (timeframe=%s)", len(ohlcv), config.data.timeframe)
    logger.info(
        "Date range: %s to %s",
        ohlcv["timestamp"].min(),
        ohlcv["timestamp"].max(),
    )

    ohlcv_path.parent.mkdir(parents=True, exist_ok=True)
    ohlcv.write_parquet(ohlcv_path)
    logger.info("Saved OHLCV: %s", ohlcv_path)


prepare_data = generate_data
