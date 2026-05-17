#!/usr/bin/env python3
"""Main entry point — thesis pipeline (4-stage contract).

Stages:
    1 — Data Preparation + Feature Engineering
    2 — Dataset Assembly
    3 — Model Training
    4 — Reporting

Execution contract:
    --stage N  means "start at Stage N and continue through Stage 4".
    No --stage means the same as --stage 1 (i.e., run all stages).

Usage:
    python main.py [--config CONFIG] [--session SESSION] [--stage N] [--force]

Options:
    --session SESSION   Continue from an existing session directory name
                        (e.g., XAUUSD_H1_20260418_143052)
    --stage N           Start at Stage N and continue through Stage 4 (1–4).
                        Default: --stage 1 (all).
    --force             Force re-run all stages
"""

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import re
import shutil
import sys
import time

import structlog

PROJECT_ROOT = Path(__file__).resolve().parent
if (PROJECT_ROOT / "src").exists():
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from thesis.shared.config import load_config  # noqa: E402
from thesis.pipeline import run_pipeline  # noqa: E402
from thesis.shared.utils import console, stage_header, stage_skip  # noqa: E402


_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


class _StripAnsiFormatter(logging.Formatter):
    """Formatter that strips ANSI escape codes — for file handlers."""

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a logging.LogRecord into a string and remove ANSI escape codes.

        This returns the formatted log message with any ANSI escape sequences stripped so it is safe for plain-text file output.

        Parameters:
            record (logging.LogRecord): The log record to format.

        Returns:
            str: The formatted log message with ANSI escape codes removed.
        """
        return _ANSI_RE.sub("", super().format(record))


def _find_session(session_name: str) -> Path | None:
    """
    Find an existing session directory by name.

    Parameters:
        session_name: The session directory name (e.g., XAUUSD_H1_20260418_143052)

    Returns:
        Path to the session directory if found, None otherwise.
    """
    results = Path("results")
    if not results.exists():
        return None

    session_path = results / session_name
    if session_path.exists() and (session_path / "config").exists():
        return session_path

    return None


def _load_session_config(session_dir: Path) -> object:
    """Load configuration from an existing session directory (snapshot + paths)."""
    snapshot = session_dir / "config" / "config_snapshot.toml"
    if snapshot.exists():
        config = load_config(snapshot)
    else:
        config = load_config("config.toml")
    config.paths.session_dir = str(session_dir)
    config.paths.model = str(session_dir / "models" / "lightgbm_model.pkl")
    config.paths.predictions = str(session_dir / "predictions" / "final_predictions.csv")
    config.paths.report = str(session_dir / "reports" / "thesis_report.md")
    config.paths.backtest_results = str(session_dir / "backtest" / "backtest_results.json")
    return config


def _apply_force_flag(config: object, force: bool) -> object:
    """Apply CLI force flag after any config load path."""
    if force:
        config.workflow.force_rerun = True
    return config


def _apply_stage_flags(config: object, stage: int | None) -> object:
    """Apply CLI stage skip contract — 4-stage model."""
    if stage is None:
        return config
    if stage >= 2:
        config.workflow.run_data = False
    if stage >= 3:
        config.workflow.run_dataset = False
    if stage >= 4:
        config.workflow.run_models = False
    return config


def main() -> None:
    """
    Command-line entry point that runs the thesis ML pipeline and records a session.

    Parses command-line options, loads and snapshots the configuration, creates a
    timestamped session directory (updating config paths and creating subdirectories),
    sets up console and file logging, executes the pipeline (and optionally an
    ablation study), and writes a session manifest JSON with metadata and timing.
    """
    parser = argparse.ArgumentParser(description="Thesis ML Pipeline")
    parser.add_argument("--config", default="config.toml", help="Path to config.toml")
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Continue from existing session (e.g., XAUUSD_H1_20260418_143052)",
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 3, 4],
        default=None,
        help=(
            "Start at Stage N and continue through Stage 4. "
            "Stage 3 means skip only Stages 1–2. "
            "No --stage means the same as --stage 1 (run all)."
        ),
    )
    parser.add_argument("--force", action="store_true", help="Force re-run all stages")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Handle force flag for new-session config.
    config = _apply_force_flag(config, args.force)

    # Determine session directory and setup
    if args.session:
        # Continue from existing session
        session_dir = _find_session(args.session)
        if session_dir is None:
            print(f"Error: Session '{args.session}' not found in results/")
            sys.exit(1)

        # Load existing session config
        config = _load_session_config(session_dir)
        config = _apply_force_flag(config, args.force)
        session_ts = (
            session_dir.name.split("_")[-2] + "_" + session_dir.name.split("_")[-1]
        )
        config.workflow.session_timestamp = session_ts

        log_mode = "a"  # Append to existing log
    else:
        # Create new session directory
        session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_name = f"{config.data.symbol}_{config.data.timeframe}_{session_ts}"
        session_dir = Path("results") / session_name

        # Update config paths to point to session directory
        config.workflow.session_timestamp = session_ts
        config.paths.session_dir = str(session_dir)
        config.paths.model = str(session_dir / "models" / "lightgbm_model.pkl")
        config.paths.predictions = str(session_dir / "predictions" / "final_predictions.csv")
        config.paths.report = str(session_dir / "reports" / "thesis_report.md")
        config.paths.backtest_results = str(session_dir / "backtest" / "backtest_results.json")

        # Create session subdirectories
        for subdir in [
            "config",
            "models",
            "predictions",
            "reports",
            "logs",
            "backtest",
        ]:
            (session_dir / subdir).mkdir(parents=True, exist_ok=True)

        # Save config snapshot
        shutil.copy2(args.config, session_dir / "config" / "config_snapshot.toml")

        log_mode = "w"  # New log file

    # Apply --stage after the final config source is known. Session config
    # snapshots otherwise reset workflow flags loaded from the CLI config.
    config = _apply_stage_flags(config, args.stage)

    _log_fmt = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    plain_file_handler = logging.FileHandler(
        session_dir / "logs" / "pipeline.log", mode=log_mode
    )
    plain_file_handler.setFormatter(_StripAnsiFormatter(_log_fmt))

    logging.basicConfig(
        level=logging.INFO,
        format=_log_fmt,
        datefmt="[%X]",
        handlers=[plain_file_handler, logging.StreamHandler(sys.stdout)],
    )
    structlog.configure(
        processors=[
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.KeyValueRenderer(
                key_order=["timestamp", "level", "logger", "event"]
            ),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    logger = structlog.get_logger("thesis")

    logger.info("config_loaded", config=args.config)
    logger.info("market", symbol=config.data.symbol, timeframe=config.data.timeframe)
    logger.info("session_directory", path=str(session_dir))
    if args.session:
        logger.info("resume_session", session=args.session)
        if args.stage is not None:
            logger.info("start_stage", stage=args.stage)

    # Track pipeline timing
    t_start = time.monotonic()
    pipeline_ok = False

    # Run pipeline and ablation with error handling
    try:
        run_pipeline(config)
        pipeline_ok = True
    except Exception as e:
        logger.exception("pipeline_failed", error=str(e))
        pipeline_ok = False
    finally:
        elapsed = round(time.monotonic() - t_start, 2)

    # Save session_info.json manifest (only for new sessions)
    if not args.session:
        # Compute full config hash so each run is tied to its exact config
        config_raw = json.dumps(asdict(config), sort_keys=True, default=str)
        config_hash = hashlib.sha256(config_raw.encode()).hexdigest()

        session_info = {
            "config_hash": config_hash,
            "symbol": config.data.symbol,
            "timeframe": config.data.timeframe,
            "session_timestamp": session_ts,
            "pipeline_duration_seconds": elapsed,
            "pipeline_ok": pipeline_ok,
            "log_files": {
                "plain": "logs/pipeline.log",
            },
            "validation": {
                "method": config.validation.method,
                "train_window_bars": config.validation.train_window_bars,
                "test_window_bars": config.validation.test_window_bars,
                "purge_bars": config.validation.purge_bars,
                "embargo_bars": config.validation.embargo_bars,
            },
            "data_range": [
                str(config.data_range.start),
                str(config.data_range.end),
            ],
            "force_rerun": config.workflow.force_rerun,
            "random_seed": config.workflow.random_seed,
        }
        session_info_path = session_dir / "config" / "session_info.json"
        with open(session_info_path, "w") as f:
            json.dump(session_info, f, indent=2)
        logger.info("session_info_saved", path=str(session_info_path))

    logger.info(
        "pipeline_done",
        results=str(session_dir),
        elapsed_seconds=elapsed,
        status="OK" if pipeline_ok else "FAILED",
    )
    logger.info("log_file", path=str(plain_file_handler.baseFilename))
    sys.exit(0 if pipeline_ok else 1)


if __name__ == "__main__":
    main()
