#!/usr/bin/env python3
"""Main entry point — simplified thesis pipeline.

Usage:
    python main.py [--config CONFIG] [--force]
"""

import argparse
import logging
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from thesis.config import load_config
from thesis.pipeline import run_pipeline
from thesis.ablation import run_ablation


_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


class _StripAnsiFormatter(logging.Formatter):
    """Formatter that strips ANSI escape codes — for file handlers."""

    def format(self, record: logging.LogRecord) -> str:
        return _ANSI_RE.sub("", super().format(record))


def main() -> None:
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
    session_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
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

    # Logging setup (now that we know the session dir)
    _log_fmt = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    file_handler = logging.FileHandler(session_dir / "logs" / "pipeline.log", mode="w")
    file_handler.setFormatter(_StripAnsiFormatter(_log_fmt))

    logging.basicConfig(
        level=logging.INFO,
        format=_log_fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            file_handler,
        ],
    )
    logger = logging.getLogger("thesis")

    logger.info("Config loaded: %s", args.config)
    logger.info("Symbol: %s, Timeframe: %s", config.data.symbol, config.data.timeframe)
    logger.info("Session directory: %s", session_dir)

    # Run pipeline
    run_pipeline(config)

    # Run ablation study if requested
    if args.ablation:
        logger.info("Running ablation study...")
        run_ablation(config)

    logger.info("Done. Results saved to: %s", session_dir)


if __name__ == "__main__":
    main()
