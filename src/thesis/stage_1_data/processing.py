"""Data prep — raw ticks → OHLCV bars."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import polars as pl

from thesis.shared.config import Config
from thesis.shared.constants import FEATURE_EPS
from thesis.shared.constants import timeframe_to_ms as _timeframe_to_ms
from thesis.shared.data_quality import (
    check_gap_report,
    classify_calendar_gaps,
    validate_ohlcv,
)

logger = logging.getLogger("thesis.prepare")

# DISCOVER


def _discover_files(raw_dir: Path, ohlcv_path: Path) -> list[Path]:
    """Find monthly parquet files. Skip if OHLCV cached."""
    if not raw_dir.exists():
        if ohlcv_path.exists():
            logger.warning("Raw dir missing but OHLCV cached — skip.")
            return []
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")
    files = sorted(raw_dir.glob("*.parquet"))
    if not files:
        if ohlcv_path.exists():
            logger.warning("No parquet files but OHLCV cached — skip.")
            return []
        raise FileNotFoundError(f"No parquet files in {raw_dir}")
    return files


# AGGREGATE


def _parse_dt(value: str, tz: str) -> pl.Expr:
    """Parse config datetime string → Polars timezone-aware expr."""
    if not value:
        raise ValueError("datetime bound must not be empty")
    value = value.strip()
    if "T" not in value and " " not in value and ":" not in value:
        value = value + "T23:59:59"
    return pl.lit(value).str.to_datetime(time_unit="us", time_zone=tz)


def _microprice(ticks: pl.DataFrame) -> pl.DataFrame:
    """Add microprice + volume columns. Weights bid/ask by opposing side size."""
    return ticks.with_columns(
        (
            (
                pl.col("ask") * pl.col("bid_volume")
                + pl.col("bid") * pl.col("ask_volume")
            )
            / (pl.col("ask_volume") + pl.col("bid_volume") + FEATURE_EPS)
        ).alias("microprice"),
        (pl.col("ask_volume") + pl.col("bid_volume")).alias("volume"),
    )


def _clip_to_month(df: pl.DataFrame, stem: str) -> pl.DataFrame:
    """Drop bars outside nominal month. Edge ticks from boundary go to wrong month."""
    year, month = int(stem[:4]), int(stem[5:7])
    start = pl.datetime(year, month, 1)
    end = (
        pl.datetime(year + 1, 1, 1) - pl.duration(seconds=1)
        if month == 12
        else pl.datetime(year, month + 1, 1) - pl.duration(seconds=1)
    )
    return df.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") <= end))


def _aggregate_file(file_path: Path, group_every: str) -> pl.DataFrame:
    """One file → OHLCV. Validate quotes, compute microprice, group by time."""
    ticks = pl.read_parquet(
        file_path,
        columns=["timestamp", "bid", "ask", "ask_volume", "bid_volume"],
    )
    n_raw = len(ticks)
    # Filter bad quotes: zero/negative bid/ask, negative volume, crossed spread
    ticks = ticks.filter(
        (pl.col("bid") > 0)
        & (pl.col("ask") > 0)
        & (pl.col("ask") >= pl.col("bid"))
        & (pl.col("ask_volume") >= 0)
        & (pl.col("bid_volume") >= 0)
    )
    if (dropped := n_raw - len(ticks)) > 0:
        logger.warning("%s: dropped %d invalid quotes", file_path.name, dropped)

    ticks = _microprice(ticks).sort("timestamp")

    bars = (
        ticks.group_by_dynamic(
            "timestamp",
            every=group_every,
            period=group_every,
            closed="left",
            label="left",
            start_by="window",
        )
        .agg(
            pl.col("microprice").first().alias("open"),
            pl.col("microprice").max().alias("high"),
            pl.col("microprice").min().alias("low"),
            pl.col("microprice").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
            pl.col("microprice").count().alias("tick_count"),
            (pl.col("ask") - pl.col("bid")).mean().alias("avg_spread"),
        )
        .drop_nulls()
    )
    return _clip_to_month(bars, file_path.stem)


def _aggregate_all(files: list[Path], group_every: str) -> pl.DataFrame:
    """All monthly files → one sorted OHLCV DataFrame."""
    bars = []
    for i, f in enumerate(files, 1):
        logger.info("Aggregating %d/%d: %s", i, len(files), f.name)
        bars.append(_aggregate_file(f, group_every))
    return pl.concat(bars, how="vertical").sort("timestamp")


# CLEAN


def _dedupe_and_filter(ohlcv: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """Drop duplicate timestamps (keep first). Drop bars with year < 2000 or > 2100."""
    n = len(ohlcv)
    dupes = n - ohlcv.get_column("timestamp").n_unique()
    if dupes > 0:
        logger.warning("Found %d duplicate timestamps — keeping first", dupes)
    ohlcv = ohlcv.unique(subset=["timestamp"], keep="first").sort("timestamp")
    before = len(ohlcv)
    ohlcv = ohlcv.filter(
        (pl.col("timestamp").dt.year() >= 2000)
        & (pl.col("timestamp").dt.year() <= 2100)
    )
    if (dropped := before - len(ohlcv)) > 0:
        logger.warning("Dropped %d bars with corrupted timestamps", dropped)
    return ohlcv, dupes


def _filter_range(ohlcv: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Apply date range from config. Raise if no bars remain."""
    tz = config.data.market_tz
    start = _parse_dt(config.data_range.start, tz)
    end = _parse_dt(config.data_range.end, tz)
    before = len(ohlcv)
    ohlcv = ohlcv.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") <= end))
    if (dropped := before - len(ohlcv)) > 0:
        logger.info(
            "Dropped %d bars outside range [%s, %s]",
            dropped,
            config.data_range.start,
            config.data_range.end,
        )
    if ohlcv.is_empty():
        raise ValueError(
            f"No OHLCV bars after date filter "
            f"[{config.data_range.start}, {config.data_range.end}]"
        )
    return ohlcv


# PERSIST


def _log_gap(ohlcv: pl.DataFrame, group_ms: int) -> None:
    """Log timestamp continuity: calendar gaps, missing bars, non-increasing deltas."""
    if len(ohlcv) < 2:
        logger.warning("Gap report skipped: < 2 bars")
        return
    result = check_gap_report(ohlcv, group_ms)
    diffs = (
        ohlcv.select((pl.col("timestamp").diff().dt.total_milliseconds()).alias("d"))
        .drop_nulls()
        .get_column("d")
    )
    non_inc = int((diffs <= 0).sum())
    logger.info(
        "Gap report: expected=%dms gaps=%d missing=%d largest=%.2f bars non_inc=%d",
        group_ms,
        result["gap_count"],
        result["estimated_missing_bars"],
        result["largest_gap_bars"],
        non_inc,
    )


def _log_quality(ohlcv: pl.DataFrame) -> None:
    """Log candle integrity: invalid candles, range/spread/tick stats."""
    if ohlcv.is_empty():
        return
    v = validate_ohlcv(ohlcv)
    if v["invalid_count"] > 0:
        logger.warning("%d invalid candles", v["invalid_count"])
    s = ohlcv.select(
        (pl.col("high") - pl.col("low")).median().alias("med_range"),
        (pl.col("high") - pl.col("low")).quantile(0.99).alias("p99_range"),
        pl.col("avg_spread").median().alias("med_spread"),
        pl.col("avg_spread").quantile(0.99).alias("p99_spread"),
        pl.col("tick_count").quantile(0.01).alias("p01_ticks"),
    ).row(0, named=True)
    logger.info(
        "Quality: med_range=%.6f p99_range=%.6f "
        "med_spread=%.6f p99_spread=%.6f p01_ticks=%.1f",
        s["med_range"] or 0,
        s["p99_range"] or 0,
        s["med_spread"] or 0,
        s["p99_spread"] or 0,
        s["p01_ticks"] or 0,
    )


def _save_json(ohlcv: pl.DataFrame, config: Config, group_ms: int, dupes: int) -> None:
    """Compute + write data quality stats to JSON sidecar."""
    total = len(ohlcv)
    stats = {
        "total_bars": total,
        "deduped_timestamps": dupes,
        "start_date": str(ohlcv["timestamp"].min()),
        "end_date": str(ohlcv["timestamp"].max()),
        "calendar_gaps": 0,
        "weekend_gaps": 0,
        "real_gaps": 0,
        "estimated_missing_bars": 0,
        "largest_gap_bars": 0,
    }
    if total >= 2:
        g = classify_calendar_gaps(ohlcv, group_ms)
        stats.update(
            calendar_gaps=g.calendar_gap_count + g.real_gap_count,
            weekend_gaps=g.calendar_gap_count,
            real_gaps=g.real_gap_count,
            estimated_missing_bars=g.estimated_missing_bars,
            largest_gap_bars=g.largest_gap_bars,
            calendar_warnings=g.warnings,
        )
    path = Path(config.paths.data_quality_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info("Saved quality JSON: %s (bars=%d)", path, total)


def _persist(ohlcv: pl.DataFrame, config: Config, dupes: int) -> None:
    """Log diagnostics, save quality JSON, write OHLCV parquet."""
    group_ms = _timeframe_to_ms(config.data.timeframe)
    _log_gap(ohlcv, group_ms)
    _log_quality(ohlcv)
    _save_json(ohlcv, config, group_ms, dupes)
    logger.info(
        "OHLCV: %d bars | %s to %s",
        len(ohlcv),
        ohlcv["timestamp"].min(),
        ohlcv["timestamp"].max(),
    )
    out = Path(config.paths.ohlcv)
    out.parent.mkdir(parents=True, exist_ok=True)
    ohlcv.write_parquet(out)
    logger.info("Saved: %s", out)


# ──────────────────────────────────────────────────────────────────────────────
# ORCHESTRATE
# ──────────────────────────────────────────────────────────────────────────────


def generate_data(config: Config) -> None:
    """Build OHLCV from raw ticks → parquet + quality JSON."""
    raw_dir = Path(config.paths.data_raw)
    ohlcv_path = Path(config.paths.ohlcv)

    if config.workflow.force_rerun and ohlcv_path.exists():
        ohlcv_path.unlink()
        logger.info("force_rerun — removed old OHLCV")

    files = _discover_files(raw_dir, ohlcv_path)
    if not files:
        return
    logger.info("Found %d tick files", len(files))

    ohlcv = _aggregate_all(files, config.data.timeframe.lower())
    ohlcv, dupes = _dedupe_and_filter(ohlcv)
    ohlcv = _filter_range(ohlcv, config)
    _persist(ohlcv, config, dupes)
