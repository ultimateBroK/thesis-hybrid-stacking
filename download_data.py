"""Download implementation for the Dukascopy ingestion pipeline."""

from __future__ import annotations

import asyncio
import logging
import calendar
from dataclasses import dataclass
from datetime import date, datetime, timezone
import json
import lzma
from pathlib import Path
import random
import struct
import aiohttp
import polars as pl

from thesis.config.loader import load_config, Config

logger = logging.getLogger(__name__)
BASE_URL = "https://datafeed.dukascopy.com/datafeed"


def _get_symbol_paths(config: Config, symbol: str) -> tuple[Path, Path]:
    """Get raw data directory and state file paths for a symbol."""
    raw_dir = Path(config.paths.data_raw)
    state_file = raw_dir / f".{symbol}_download_state.json"
    return raw_dir, state_file


@dataclass
class DownloadRuntimeConfig:
    """Runtime config for Dukascopy tick downloader."""

    symbol: str
    start_year: int
    start_month: int
    end_year: int | None
    end_month: int | None
    asset_class: str
    concurrency: int
    force: bool
    skip_current_month: bool
    output_dir: Path
    state_file: Path


def build_download_config(
    symbol: str,
    start_year: int,
    start_month: int,
    asset_class: str,
    concurrency: int,
    force: bool,
    *,
    end_year: int | None = None,
    end_month: int | None = None,
    skip_current_month: bool = False,
    config: Config | None = None,
) -> DownloadRuntimeConfig:
    """Build an immutable runtime config for the downloader."""
    if config is None:
        config = load_config()
    output_dir, state_file = _get_symbol_paths(config, symbol)
    return DownloadRuntimeConfig(
        symbol=symbol,
        start_year=start_year,
        start_month=start_month,
        end_year=end_year,
        end_month=end_month,
        asset_class=asset_class,
        concurrency=concurrency,
        force=force,
        skip_current_month=skip_current_month,
        output_dir=output_dir,
        state_file=state_file,
    )


def list_available_raw_months(
    symbol: str,
    *,
    config: Config | None = None,
) -> list[tuple[int, int]]:
    """Return list of (year, month) from YYYY-MM.parquet files in raw dir."""
    if config is None:
        config = load_config()
    raw_dir, _ = _get_symbol_paths(config, symbol)
    if not raw_dir.exists():
        return []
    result: list[tuple[int, int]] = []
    for p in raw_dir.glob("????-??.parquet"):
        stem = p.stem
        if len(stem) == 7 and stem[4] == "-":
            try:
                y, m = int(stem[:4]), int(stem[5:7])
                if 1 <= m <= 12:
                    result.append((y, m))
            except ValueError:
                continue
    return sorted(result)


def load_state(state_file: Path) -> dict[str, dict[str, int]]:
    """Read the downloaded-month tracking state from disk."""
    if state_file.exists():
        with state_file.open() as handle:
            return json.load(handle)
    return {}


def save_state(state_file: Path, state: dict[str, dict[str, int]]) -> None:
    """Persist the month-tracking state dictionary as JSON."""
    state_file.parent.mkdir(parents=True, exist_ok=True)
    with state_file.open("w") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)


def parse_hour(
    raw: bytes, year: int, month: int, day: int, hour: int
) -> pl.DataFrame | None:
    """Decode raw Dukascopy bi5 bytes into a tick dataframe."""
    if not raw:
        return None

    base_ms = int(
        datetime(year, month, day, hour, tzinfo=timezone.utc).timestamp() * 1000
    )
    chunk = 20
    records = [
        (
            base_ms + struct.unpack_from(">I", raw, i)[0],
            struct.unpack_from(">I", raw, i + 4)[0] / 1000.0,
            struct.unpack_from(">I", raw, i + 8)[0] / 1000.0,
            struct.unpack_from(">f", raw, i + 12)[0],
            struct.unpack_from(">f", raw, i + 16)[0],
        )
        for i in range(0, len(raw) - chunk + 1, chunk)
    ]
    if not records:
        return None
    return pl.DataFrame(
        records,
        schema=["timestamp_ms", "ask", "bid", "ask_volume", "bid_volume"],
        orient="row",
    )


async def _fetch_one(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    config: DownloadRuntimeConfig,
    year: int,
    month_idx: int,
    day: int,
    hour: int,
    month: int,
) -> pl.DataFrame | None | str:
    """Fetch, decompress, and parse one hourly file asynchronously."""
    url = f"{BASE_URL}/{config.symbol}/{year:04d}/{month_idx:02d}/{day:02d}/{hour:02d}h_ticks.bi5"
    async with semaphore:
        for attempt in range(4):
            try:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 404:
                        return None
                    if response.status in (429, 503):
                        await asyncio.sleep(min(2**attempt, 16) + random.random())
                        continue
                    compressed = await response.read()
                try:
                    raw = lzma.decompress(compressed)
                except lzma.LZMAError:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                return parse_hour(raw, year, month, day, hour)
            except (asyncio.TimeoutError, aiohttp.ClientError):
                if attempt < 3:
                    await asyncio.sleep(min(2**attempt, 8) * 0.5 + random.random())
    return "TIMEOUT"


async def _fetch_hours_async(
    config: DownloadRuntimeConfig,
    slots: list[tuple[int, int, int, int]],
    month: int,
) -> tuple[list[pl.DataFrame], int]:
    """Execute asynchronous downloads over multiple hourly slots."""
    semaphore = asyncio.Semaphore(config.concurrency)
    connector = aiohttp.TCPConnector(limit=config.concurrency, ttl_dns_cache=300)
    headers = {"User-Agent": "Mozilla/5.0"}

    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        results = await asyncio.gather(
            *[
                _fetch_one(
                    session, semaphore, config, year, month_idx, day, hour, month
                )
                for year, month_idx, day, hour in slots
            ]
        )

    frames: list[pl.DataFrame] = []
    timed_out = 0
    for result in results:
        if result is None:
            continue
        if isinstance(result, str):
            timed_out += 1
            continue
        frames.append(result)
    return frames, timed_out


def fetch_hours(
    config: DownloadRuntimeConfig,
    slots: list[tuple[int, int, int, int]],
    month: int,
) -> tuple[list[pl.DataFrame], int]:
    """Public sync interface that runs the async downloader under the hood."""
    if not slots:
        return [], 0
    frames, timed_out = asyncio.run(_fetch_hours_async(config, slots, month))
    if timed_out:
        logger.warning("%d hours timed out (will retry on next run)", timed_out)
    return frames, timed_out


def _get_slots(
    config: DownloadRuntimeConfig,
    year: int,
    month: int,
    *,
    exclude_weekends: bool = False,
) -> list[tuple[int, int, int, int]]:
    """Generate hour slots for a month.

    Args:
        exclude_weekends: If True, exclude all weekend hours (for repair checks).
                         If False, allow Sunday evening 21:00-24:00 (for full download).
    """
    month_idx = month - 1
    days_in_month = calendar.monthrange(year, month)[1]

    if config.asset_class == "crypto":
        return [
            (year, month_idx, day, hour)
            for day in range(1, days_in_month + 1)
            for hour in range(24)
        ]

    slots = []
    for day in range(1, days_in_month + 1):
        weekday = date(year, month, day).weekday()
        if exclude_weekends:
            # Weekdays only (Mon-Fri)
            if weekday < 5:
                for hour in range(24):
                    slots.append((year, month_idx, day, hour))
        else:
            # All except Saturday, and Sunday before 21:00
            if weekday != 5:  # Not Saturday
                start_hour = 21 if weekday == 6 else 0
                for hour in range(start_hour, 24):
                    slots.append((year, month_idx, day, hour))
    return slots


all_slots = lambda config, year, month: _get_slots(
    config, year, month, exclude_weekends=False
)
weekday_slots = lambda config, year, month: _get_slots(
    config, year, month, exclude_weekends=True
)


def _get_covered_hours(df: pl.DataFrame) -> set[tuple[int, int]]:
    """Extract set of (day, hour) tuples covered in the dataframe."""
    return set(
        df.with_columns(
            [
                pl.col("timestamp").dt.day().alias("_day"),
                pl.col("timestamp").dt.hour().alias("_hour"),
            ]
        )
        .select(["_day", "_hour"])
        .unique()
        .rows()
    )


def repair_month(
    config: DownloadRuntimeConfig,
    year: int,
    month: int,
    file_path: Path,
) -> tuple[int, int]:
    """Detect and patch missing weekday-hour slots in an existing parquet file."""
    df = pl.read_parquet(file_path)
    covered = _get_covered_hours(df)

    missing = [
        (year, month_idx, day, hour)
        for year, month_idx, day, hour in weekday_slots(config, year, month)
        if (day, hour) not in covered
    ]
    if not missing:
        return len(df), 0

    logger.info("-> %d weekday-hour slots missing, fetching...", len(missing))
    new_frames, _ = fetch_hours(config, missing, month)
    if new_frames:
        added = sum(len(frame) for frame in new_frames)
        new_dfs = [
            frame.with_columns(
                pl.from_epoch("timestamp_ms", time_unit="ms").alias("timestamp")
            ).select(["timestamp", "ask", "bid", "ask_volume", "bid_volume"])
            for frame in new_frames
        ]
        df = (
            pl.concat([df] + new_dfs)
            .unique(subset=["timestamp"], keep="first")
            .sort("timestamp")
        )
        df.write_parquet(file_path)
        logger.info("-> Patched +%s rows", f"{added:,}")

    covered_after = _get_covered_hours(df)
    still_missing = sum(
        1
        for _, _, day, hour in weekday_slots(config, year, month)
        if (day, hour) not in covered_after
    )
    return len(df), still_missing


def run_download_job(
    symbol: str,
    asset_class: str,
    start_year: int,
    start_month: int,
    concurrency: int,
    force: bool,
    *,
    end_year: int | None = None,
    end_month: int | None = None,
    skip_current_month: bool = False,
    config: Config | None = None,
) -> bool:
    """Download, validate, and repair monthly tick parquet files for one symbol."""
    download_config = build_download_config(
        symbol,
        start_year,
        start_month,
        asset_class,
        concurrency,
        force,
        end_year=end_year,
        end_month=end_month,
        skip_current_month=skip_current_month,
        config=config,
    )
    download_config.output_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    effective_end_year = (
        download_config.end_year if download_config.end_year is not None else now.year
    )
    effective_end_month = (
        download_config.end_month
        if download_config.end_month is not None
        else now.month
    )

    state = load_state(download_config.state_file)

    for year in range(download_config.start_year, effective_end_year + 1):
        month_start = (
            download_config.start_month if year == download_config.start_year else 1
        )
        month_end = effective_end_month if year == effective_end_year else 12
        for month in range(month_start, month_end + 1):
            key = f"{year}-{month:02d}"
            file_path = download_config.output_dir / f"{key}.parquet"
            is_past = not (year == now.year and month == now.month)
            is_current_month = year == now.year and month == now.month
            entry = state.get(key)

            if download_config.skip_current_month and is_current_month:
                logger.info(
                    "Skip     %s  (current month, skip_current_month=True)", key
                )
                continue

            if (
                is_past
                and entry
                and entry["missing_hours"] == 0
                and file_path.exists()
                and not download_config.force
            ):
                logger.info(
                    "Skip     %s  rows=%10s  missing=0", key, f"{entry['rows']:,}"
                )
                continue

            if (
                file_path.exists()
                and entry is None
                and is_past
                and not download_config.force
            ):
                try:
                    df = pl.read_parquet(file_path)
                    rows = len(df)
                except Exception:
                    rows = -1
                state[key] = {"rows": rows, "missing_hours": 0}
                save_state(download_config.state_file, state)
                logger.info(
                    "Skip     %s  rows=%10s  (inferred from file, no state)",
                    key,
                    f"{rows:,}",
                )
                continue

            if file_path.exists():
                logger.info("Checking %s ...", key)
                rows, missing = repair_month(download_config, year, month, file_path)
                flag = "full" if missing == 0 else f"{missing} hrs missing"
                logger.info("   %s  rows=%10s  %s", key, f"{rows:,}", flag)
                if is_past:
                    state[key] = {"rows": rows, "missing_hours": missing}
                    save_state(download_config.state_file, state)
                continue

            logger.info("Download %s ...", key)
            frames, timed_out = fetch_hours(
                download_config, all_slots(download_config, year, month), month
            )
            if frames:
                combined = pl.concat(frames).sort("timestamp_ms")
                df = combined.with_columns(
                    pl.from_epoch("timestamp_ms", time_unit="ms").alias("timestamp")
                ).select(["timestamp", "ask", "bid", "ask_volume", "bid_volume"])
                df.write_parquet(file_path)
                logger.info("Saved %s rows.", f"{len(df):,}")
            else:
                df = None
                logger.info("No data found.")

            if is_past:
                rows = len(df) if df is not None else 0
                state[key] = {"rows": rows, "missing_hours": timed_out}
                save_state(download_config.state_file, state)
    return True
