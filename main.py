#!/usr/bin/env python3
"""Main entry point — simplified thesis pipeline.

Usage:
    python main.py [--config CONFIG] [--force]
"""

import argparse
import json
import logging
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from thesis.config import load_config
from thesis.pipeline import run_pipeline
from thesis.ablation import run_ablation


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
    parser.add_argument("--force", action="store_true", help="Force re-run all stages")
    parser.add_argument(
        "--ablation", action="store_true", help="Run ablation study after pipeline"
    )
    args = parser.parse_args()

    # Load config first (before logging setup, so we know the session dir)
    config = load_config(args.config)
    if args.force:
        config.workflow.force_rerun = True

    # Create session directory and update paths
    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_name = f"{config.data.symbol}_{config.data.timeframe}_{session_ts}"
    session_dir = Path("results") / session_name

    # Update config paths to point to session directory
    config.workflow.session_timestamp = session_ts
    config.paths.session_dir = str(session_dir)
    config.paths.model = str(session_dir / "models" / "lightgbm_model.pkl")
    config.paths.predictions = str(
        session_dir / "predictions" / "final_predictions.parquet"
    )
    config.paths.backtest_results = str(
        session_dir / "backtest" / "backtest_results.json"
    )
    config.paths.report = str(session_dir / "reports" / "thesis_report.md")

    # Create session subdirectories
    for subdir in ["config", "models", "predictions", "reports", "backtest", "logs"]:
        (session_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Save config snapshot
    shutil.copy2(args.config, session_dir / "config" / "config_snapshot.toml")

    # Logging setup — Rich for console, plain for file
    from rich.logging import RichHandler

    from thesis.ui import console as _console

    _log_fmt = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    file_handler = logging.FileHandler(session_dir / "logs" / "pipeline.log", mode="w")
    file_handler.setFormatter(_StripAnsiFormatter(_log_fmt))

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=_console,
                rich_tracebacks=True,
                show_path=False,
                show_time=True,
                omit_repeated_times=False,
                log_time_format="[%H:%M:%S]",
                markup=True,
            ),
            file_handler,
        ],
    )
    logger = logging.getLogger("thesis")

    logger.info("Config loaded: %s", args.config)
    logger.info("Symbol: %s, Timeframe: %s", config.data.symbol, config.data.timeframe)
    logger.info("Session directory: %s", session_dir)

    # Track pipeline timing
    t_start = time.monotonic()

    # Run pipeline
    run_pipeline(config)

    # Run ablation study if requested
    if args.ablation:
        logger.info("Running ablation study...")
        run_ablation(config)

    elapsed = round(time.monotonic() - t_start, 2)

    # Save session_info.json manifest
    session_info = {
        "symbol": config.data.symbol,
        "timeframe": config.data.timeframe,
        "session_timestamp": session_ts,
        "pipeline_duration_seconds": elapsed,
        "data_range": {
            "train": [
                str(config.splitting.train_start),
                str(config.splitting.train_end),
            ],
            "val": [str(config.splitting.val_start), str(config.splitting.val_end)],
            "test": [str(config.splitting.test_start), str(config.splitting.test_end)],
        },
        "force_rerun": config.workflow.force_rerun,
        "random_seed": config.workflow.random_seed,
    }
    session_info_path = session_dir / "config" / "session_info.json"
    with open(session_info_path, "w") as f:
        json.dump(session_info, f, indent=2)
    logger.info("Session info saved: %s", session_info_path)

    logger.info("Done. Results saved to: %s (%.1fs)", session_dir, elapsed)


if __name__ == "__main__":
    main()
