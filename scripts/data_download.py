"""Download implementation for the Dukascopy ingestion pipeline."""

from __future__ import annotations

# Add src directory to path for imports when running as script  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

src_dir = Path(__file__).parent / "src"  # noqa: E402
if src_dir.exists() and str(src_dir) not in sys.path:  # noqa: E402
    sys.path.insert(0, str(src_dir))  # noqa: E402

import argparse  # noqa: E402
import asyncio  # noqa: E402
import logging  # noqa: E402
import calendar  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from datetime import date, datetime, timezone  # noqa: E402
import json  # noqa: E402
import lzma  # noqa: E402
import random  # noqa: E402
import struct  # noqa: E402
import aiohttp  # noqa: E402
import polars as pl  # noqa: E402

logger = logging.getLogger(__name__)
BASE_URL = "https://datafeed.dukascopy.com/datafeed"


# ---------------------------------------------------------------------------
# Lightweight paths helper (self-contained — no dependency on thesis.config)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectPaths:
    """Minimal path resolver for the download pipeline."""

    data_raw_base: Path = Path("data/raw")

    def raw_data_dir(self, symbol: str) -> Path:
        """
        Get the raw-tick data directory path for the given filesystem-safe symbol.
        
        Parameters:
            symbol (str): Filesystem-safe symbol (e.g., with '/' replaced by '_') used as the directory name under the raw data base.
        
        Returns:
            Path: Path to the symbol's raw-tick directory (data_raw_base / symbol).
        """
        return self.data_raw_base / symbol

    def state_file(self, symbol: str) -> Path:
        """
        Get the filesystem path to the download-state JSON file for the given symbol.
        
        Parameters:
            symbol (str): Filesystem-safe symbol (e.g., slashes replaced with underscores).
        
        Returns:
            Path: Path to 'download_state.json' located under the project's raw data directory for the symbol.
        """
        return self.data_raw_base / symbol / "download_state.json"


DEFAULT_PATHS = ProjectPaths()


def _fs_symbol(symbol: str) -> str:
    """Return a filesystem-safe symbol for directory and file names.

    Replaces ``/`` with ``_`` so that composite instrument names
    (e.g. ``BASE/QUOTE``) produce valid directory and file names.
    """
    return symbol.replace("/", "_")


# Display-name → Dukascopy data-feed symbol mappings.
# Add entries here when an instrument's URL symbol differs from its
# display name.  Keys are the canonical name passed via --symbol;
# values are what Dukascopy's datafeed expects in the URL path.
SYMBOL_OVERRIDES: dict[str, str] = {}


def _download_symbol(symbol: str) -> str:
    """
    Select the Dukascopy feed symbol corresponding to a canonical symbol.
    
    If an explicit mapping exists in SYMBOL_OVERRIDES that mapping is returned; otherwise, if the canonical symbol contains a slash (`/`) the substring before the slash is returned; if neither condition applies, the original symbol is returned.
    
    Returns:
        str: The symbol to use in Dukascopy URL paths.
    """
    if symbol in SYMBOL_OVERRIDES:
        return SYMBOL_OVERRIDES[symbol]
    if "/" in symbol:
        return symbol.split("/")[0]
    return symbol


@dataclass(frozen=True)
class DownloadRuntimeConfig:
    """Immutable runtime config for Dukascopy tick downloader."""

    symbol: str
    download_symbol: str  # Symbol for URL construction (may differ from fs name)
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
    paths: ProjectPaths = DEFAULT_PATHS,
) -> DownloadRuntimeConfig:
    """
    Create a DownloadRuntimeConfig with filesystem-safe output/state paths and the Dukascopy download symbol resolved.
    
    Parameters:
        symbol (str): Canonical symbol provided by the user (may contain '/' characters).
        start_year (int): First year to include.
        start_month (int): First month to include (1-12).
        asset_class (str): Asset class identifier (e.g., "fx", "crypto", "index").
        concurrency (int): Maximum concurrent HTTP downloads.
        force (bool): If true, ignore existing files/state and re-download.
        end_year (int | None): Optional final year to include.
        end_month (int | None): Optional final month to include (1-12).
        skip_current_month (bool): If true, omit processing the current month.
        paths (ProjectPaths): Project path helpers used to compute output and state file locations.
    
    Returns:
        DownloadRuntimeConfig: Immutable runtime configuration with `download_symbol`, `output_dir`, and `state_file` populated.
    """
    fs_sym = _fs_symbol(symbol)
    return DownloadRuntimeConfig(
        symbol=symbol,
        download_symbol=_download_symbol(symbol),
        start_year=start_year,
        start_month=start_month,
        end_year=end_year,
        end_month=end_month,
        asset_class=asset_class,
        concurrency=concurrency,
        force=force,
        skip_current_month=skip_current_month,
        output_dir=paths.raw_data_dir(fs_sym),
        state_file=paths.state_file(fs_sym),
    )


def list_available_raw_months(
    symbol: str,
    *,
    paths: ProjectPaths = DEFAULT_PATHS,
) -> list[tuple[int, int]]:
    """
    List available months for which raw parquet files exist for a symbol.
    
    Scans the raw-data directory for files named `YYYY-MM.parquet` and returns the parsed (year, month) tuples sorted in ascending order.
    
    Parameters:
    	symbol (str): Canonical symbol; slashes are converted to a filesystem-safe form before locating the directory.
    	paths (ProjectPaths): Paths helper used to locate the symbol's raw data directory (defaults to DEFAULT_PATHS).
    
    Returns:
    	list[tuple[int, int]]: Sorted list of (year, month) tuples found in the raw directory; empty if the directory does not exist or no matching files are found.
    """
    raw_dir = paths.raw_data_dir(_fs_symbol(symbol))
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
    if state_file.exists() and state_file.stat().st_size > 0:
        try:
            with state_file.open() as handle:
                data = json.load(handle)
            if isinstance(data, list):
                migrated = {key: {"rows": -1, "missing_hours": 0} for key in data}
                save_state(state_file, migrated)
                logger.info(
                    "Migrated state file to new format (%d entries)", len(migrated)
                )
                return migrated
            return data
        except json.JSONDecodeError:
            logger.warning(
                "State file %s is corrupted/empty, starting fresh", state_file
            )
            return {}
    return {}


def save_state(state_file: Path, state: dict[str, dict[str, int]]) -> None:
    """Persist the month-tracking state dictionary as JSON."""
    state_file.parent.mkdir(parents=True, exist_ok=True)
    with state_file.open("w") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)


def migrate_old_markers(
    output_dir: Path,
    state_file: Path,
    state: dict[str, dict[str, int]],
) -> dict[str, dict[str, int]]:
    """Absorb old `*.parquet.complete` files into the JSON state file."""
    if not output_dir.exists():
        return state

    old_markers = list(output_dir.glob("*.parquet.complete"))
    for marker in old_markers:
        key = marker.name.replace(".parquet.complete", "")
        state.setdefault(key, {"rows": -1, "missing_hours": 0})
        marker.unlink()
    if old_markers:
        logger.info("Migrated %d old markers -> %s", len(old_markers), state_file)
    return state


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


def to_datetime_df(df: pl.DataFrame) -> pl.DataFrame:
    """Convert an epoch-based tick dataframe into canonical timestamp columns."""
    return df.with_columns(
        pl.from_epoch("timestamp_ms", time_unit="ms")
        .dt.cast_time_unit("ms")
        .alias("timestamp")
    ).select(["timestamp", "ask", "bid", "ask_volume", "bid_volume"])


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
    """
    Download, decompress, and parse a single Dukascopy hourly bi5 file for a specific datetime slot.
    
    Attempts up to four HTTP fetch/decompress/parse attempts with backoff. Returns parsed tick data when successful, `None` when the remote file is absent (HTTP 404), or the string `'TIMEOUT'` when all attempts fail.
    
    Parameters:
        month_idx (int): Zero-based month component used in the Dukascopy URL path (0 == January).
        month (int): One-based calendar month used for parsing and downstream metadata (1 == January).
    
    Returns:
        pl.DataFrame | None | str: `pl.DataFrame` containing parsed tick records for the hour; `None` if the server returned 404 (no file for that hour); or the string `'TIMEOUT'` if all retries were exhausted without a successful parse.
    """
    url = f"{BASE_URL}/{config.download_symbol}/{year:04d}/{month_idx:02d}/{day:02d}/{hour:02d}h_ticks.bi5"
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


def all_slots(
    config: DownloadRuntimeConfig, year: int, month: int
) -> list[tuple[int, int, int, int]]:
    """Return all tradeable hour slots for a month."""
    month_idx = month - 1
    days_in_month = calendar.monthrange(year, month)[1]
    if config.asset_class == "crypto":
        return [
            (year, month_idx, day, hour)
            for day in range(1, days_in_month + 1)
            for hour in range(24)
        ]
    return [
        (year, month_idx, day, hour)
        for day in range(1, days_in_month + 1)
        for hour in range(24)
        if date(year, month, day).weekday() != 5
        and not (date(year, month, day).weekday() == 6 and hour < 21)
    ]


def weekday_slots(
    config: DownloadRuntimeConfig,
    year: int,
    month: int,
) -> list[tuple[int, int, int, int]]:
    """Return Mon-Fri slots, or 24/7 slots for crypto, for repair checks."""
    month_idx = month - 1
    days_in_month = calendar.monthrange(year, month)[1]
    if config.asset_class == "crypto":
        return [
            (year, month_idx, day, hour)
            for day in range(1, days_in_month + 1)
            for hour in range(24)
        ]
    return [
        (year, month_idx, day, hour)
        for day in range(1, days_in_month + 1)
        if date(year, month, day).weekday() < 5
        for hour in range(24)
    ]


def repair_month(
    config: DownloadRuntimeConfig,
    year: int,
    month: int,
    file_path: Path,
) -> tuple[int, int]:
    """Detect and patch missing weekday-hour slots in an existing parquet file."""
    df = pl.read_parquet(file_path)
    covered = set(
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
        # Ensure consistent datetime precision (milliseconds) for all frames
        df = (
            pl.concat(
                [
                    df.with_columns(pl.col("timestamp").cast(pl.Datetime("ms"))),
                    *[
                        to_datetime_df(frame).with_columns(
                            pl.col("timestamp").cast(pl.Datetime("ms"))
                        )
                        for frame in new_frames
                    ],
                ]
            )
            .unique(subset=["timestamp"], keep="first")
            .sort("timestamp")
        )
        df.write_parquet(file_path)
        logger.info("-> Patched +%s rows", f"{added:,}")

    covered_after = set(
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
    still_missing = sum(
        1
        for _, _, day, hour in weekday_slots(config, year, month)
        if (day, hour) not in covered_after
    )
    return len(df), still_missing


def _infer_state_from_file(file_path: Path) -> tuple[int, int]:
    """Read row count from parquet file. Returns (rows, missing_hours=0)."""
    try:
        df = pl.read_parquet(file_path)
        return len(df), 0
    except Exception:
        return -1, 0


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
    paths: ProjectPaths = DEFAULT_PATHS,
) -> bool:
    """Download, validate, and repair monthly tick parquet files for one symbol."""
    config = build_download_config(
        symbol,
        start_year,
        start_month,
        asset_class,
        concurrency,
        force,
        end_year=end_year,
        end_month=end_month,
        skip_current_month=skip_current_month,
        paths=paths,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    effective_end_year = config.end_year if config.end_year is not None else now.year
    effective_end_month = (
        config.end_month if config.end_month is not None else now.month
    )

    state = migrate_old_markers(
        config.output_dir,
        config.state_file,
        load_state(config.state_file),
    )

    for year in range(config.start_year, effective_end_year + 1):
        month_start = config.start_month if year == config.start_year else 1
        month_end = effective_end_month if year == effective_end_year else 12
        for month in range(month_start, month_end + 1):
            key = f"{year}-{month:02d}"
            file_path = config.output_dir / f"{key}.parquet"
            is_past = not (year == now.year and month == now.month)
            is_current_month = year == now.year and month == now.month
            entry = state.get(key)

            if config.skip_current_month and is_current_month:
                logger.info(
                    "Skip     %s  (current month, skip_current_month=True)", key
                )
                continue

            if (
                is_past
                and entry
                and entry["missing_hours"] == 0
                and file_path.exists()
                and not config.force
            ):
                logger.info(
                    "Skip     %s  rows=%10s  missing=0", key, f"{entry['rows']:,}"
                )
                continue

            if file_path.exists() and entry is None and is_past and not config.force:
                rows, _ = _infer_state_from_file(file_path)
                state[key] = {"rows": rows, "missing_hours": 0}
                save_state(config.state_file, state)
                logger.info(
                    "Skip     %s  rows=%10s  (inferred from file, no state)",
                    key,
                    f"{rows:,}",
                )
                continue

            if file_path.exists():
                logger.info("Checking %s ...", key)
                rows, missing = repair_month(config, year, month, file_path)
                flag = "full" if missing == 0 else f"{missing} hrs missing"
                logger.info("   %s  rows=%10s  %s", key, f"{rows:,}", flag)
                if is_past:
                    state[key] = {"rows": rows, "missing_hours": missing}
                    save_state(config.state_file, state)
                continue

            logger.info("Download %s ...", key)
            frames, timed_out = fetch_hours(
                config, all_slots(config, year, month), month
            )
            if frames:
                df = to_datetime_df(pl.concat(frames).sort("timestamp_ms"))
                df.write_parquet(file_path)
                logger.info("Saved %s rows.", f"{len(df):,}")
            else:
                df = None
                logger.info("No data found.")

            if is_past:
                rows = len(df) if df is not None else 0
                state[key] = {"rows": rows, "missing_hours": timed_out}
                save_state(config.state_file, state)
    return True


def _add_download_parser(subparsers: argparse._SubParsersAction) -> None:
    """
    Register the "download" subcommand on the provided subparsers and configure its CLI arguments using defaults from config.toml.
    
    If loading config.toml fails, sensible fallback defaults are used for start/end dates, concurrency, force, and skip-current-month.
    
    Parameters:
        subparsers (argparse._SubParsersAction): Subparsers collection returned by ArgumentParser.add_subparsers() to which the "download" command will be added.
    """
    from thesis.config import load_config

    # Load config to get defaults
    try:
        cfg = load_config("config.toml")
        data_cfg = cfg.data

        # Parse start_date for year/month defaults
        start_dt = datetime.strptime(data_cfg.start_date, "%Y-%m-%d")
        default_start_year = start_dt.year
        default_start_month = start_dt.month

        # Parse end_date for year/month defaults (optional)
        end_dt = datetime.strptime(data_cfg.end_date, "%Y-%m-%d")
        default_end_year = end_dt.year
        default_end_month = end_dt.month

        # Get download-specific defaults
        default_symbol = data_cfg.symbol
        default_asset_class = data_cfg.asset_class
        default_concurrency = data_cfg.download_concurrency
        default_force = data_cfg.download_force
        default_skip_current = data_cfg.download_skip_current_month
    except Exception:
        # Fallback defaults if config loading fails
        default_symbol = None
        default_asset_class = "fx"
        default_start_year = 2018
        default_start_month = 1
        default_end_year = None  # Current year
        default_end_month = None  # Current month
        default_concurrency = 8
        default_force = False
        default_skip_current = False

    download = subparsers.add_parser(
        "download",
        help="Download raw tick data from Dukascopy",
        description="Download historical tick data for a symbol.",
    )
    download.add_argument(
        "--symbol",
        required=True,
        help="Instrument symbol to download (e.g. XAUUSD, EURUSD)",
    )
    download.add_argument(
        "--asset-class",
        choices=["fx", "crypto", "index"],
        default=default_asset_class,
        help=f"Asset class (default: {default_asset_class})",
    )
    download.add_argument(
        "--start-year",
        type=int,
        default=default_start_year,
        help=f"Start year (default: {default_start_year})",
    )
    download.add_argument(
        "--start-month",
        type=int,
        default=default_start_month,
        help=f"Start month 1-12 (default: {default_start_month})",
    )
    download.add_argument(
        "--end-year",
        type=int,
        default=default_end_year,
        help="End year (default: current year)"
        if default_end_year is None
        else f"End year (default: {default_end_year})",
    )
    download.add_argument(
        "--end-month",
        type=int,
        default=default_end_month,
        help="End month (default: current month)"
        if default_end_month is None
        else f"End month (default: {default_end_month})",
    )
    download.add_argument(
        "--concurrency",
        type=int,
        default=default_concurrency,
        help=f"Parallel downloads (default: {default_concurrency})",
    )
    download.add_argument(
        "--force",
        action=argparse.BooleanOptionalAction,
        default=default_force,
        help=f"Force re-verify existing months (default: {default_force})",
    )
    download.add_argument(
        "--skip-current-month",
        action=argparse.BooleanOptionalAction,
        default=default_skip_current,
        help=f"Skip checking/repairing current month (default: {default_skip_current})",
    )


def _get_download_defaults_from_config():
    """
    Load download-related defaults from config.toml.
    
    Returns a mapping of download defaults extracted from the file with the following keys:
    - "symbol": canonical symbol string.
    - "asset_class": asset class (e.g., "fx", "crypto", "index").
    - "start_year": start year as an integer.
    - "start_month": start month as an integer (1-12).
    - "end_year": end year as an integer.
    - "end_month": end month as an integer (1-12).
    - "concurrency": download concurrency as an integer.
    - "force": boolean indicating whether to force re-downloads.
    - "skip_current_month": boolean indicating whether to skip the current month.
    
    Returns:
        dict[str, object]: The defaults mapping, or `None` if the configuration could not be loaded or parsed.
    """
    from thesis.config import load_config
    from datetime import datetime

    try:
        cfg = load_config("config.toml")
        data_cfg = cfg.data

        start_dt = datetime.strptime(data_cfg.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(data_cfg.end_date, "%Y-%m-%d")

        return {
            "symbol": data_cfg.symbol,
            "asset_class": data_cfg.asset_class,
            "start_year": start_dt.year,
            "start_month": start_dt.month,
            "end_year": end_dt.year,
            "end_month": end_dt.month,
            "concurrency": data_cfg.download_concurrency,
            "force": data_cfg.download_force,
            "skip_current_month": data_cfg.download_skip_current_month,
        }
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to load config defaults: {e}")
        return None


_SYMBOL_HELP = """\
supported symbols (verified against Dukascopy data feed):

  FX majors & minors:
    EURUSD  GBPUSD  USDJPY  USDCHF  AUDUSD  NZDUSD  USDCAD

  FX crosses:
    EURGBP  EURJPY  EURCHF  EURAUD  EURNZD  EURCAD
    GBPJPY  GBPCHF  GBPAUD  GBPNZD  GBPCAD
    AUDJPY  AUDCHF  AUDNZD  AUDCAD
    NZDJPY  NZDCHF  NZDCAD  CADJPY  CADCHF  CHFJPY

  FX Scandi & exotic:
    USDSEK  USDNOK  USDDKK  USDPLN  USDCZK  USDHUF
    EURSEK  EURNOK  EURPLN
    USDTRY  USDMXN  USDZAR  USDSGD  USDHKD  USDCNH
    EURTRY

  Metals (--asset-class fx):
    XAUUSD  XAGUSD  XAUEUR  XAGEUR

  Indices (--asset-class index):
    DOLLARIDXUSD  DEUIDXEUR  ESPIDXEUR

  Crypto (--asset-class crypto):
    BTCUSD  ETHUSD  XRPUSD  LTCUSD  ADAUSD  BTCEUR  ETHEUR

Pass any of these to --symbol.  The list is not exhaustive — other
Dukascopy instruments may work if you know the exact feed name.
"""


def main():
    """
    CLI entry point that parses command-line arguments and runs the data download job.
    
    Parses download-related CLI options, configures logging, invokes `run_download_job` with the parsed values, and returns an appropriate process exit code.
    
    Returns:
        int: `0` on success, `1` on failure.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="data_download.py",
        description="Download historical tick data from Dukascopy.",
        epilog=_SYMBOL_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--symbol",
        required=True,
        help="Instrument symbol to download (see list below)",
    )
    parser.add_argument(
        "--asset-class",
        choices=["fx", "crypto", "index"],
        default="fx",
        help="Asset class — controls trading-hour assumptions (default: fx)",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        required=True,
        help="Start year",
    )
    parser.add_argument(
        "--start-month",
        type=int,
        required=True,
        help="Start month 1-12",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="End year (default: current year)",
    )
    parser.add_argument(
        "--end-month",
        type=int,
        default=None,
        help="End month (default: current month)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Parallel downloads (default: 8)",
    )
    parser.add_argument(
        "--force",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force re-verify existing months (default: False)",
    )
    parser.add_argument(
        "--skip-current-month",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip checking/repairing current month (default: False)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    # Run download
    success = run_download_job(
        symbol=args.symbol,
        asset_class=args.asset_class,
        start_year=args.start_year,
        start_month=args.start_month,
        concurrency=args.concurrency,
        force=args.force,
        end_year=args.end_year,
        end_month=args.end_month,
        skip_current_month=args.skip_current_month,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
