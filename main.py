#!/usr/bin/env python3
"""
Main entry point for Hybrid Stacking thesis pipeline.

Bachelor's Thesis: Hybrid Stacking (LSTM + LightGBM) for XAU/USD H1 Trading Signals
Student: Nguyen Duc Hieu - 2151061192
Advisor: Hoang Quoc Dung
Thuy Loi University

Usage:
    python main.py [--stage STAGE] [--force] [--config CONFIG]

Options:
    --stage STAGE         Run specific stage only (data, features, labels, split,
                          lightgbm, lstm, stacking, backtest, report)
    --force               Force re-run (ignore cache)
    --config CONFIG       Path to config file (default: config.toml)
    --session-id NAME     Use custom session ID (default: auto-generated)
    --list-sessions       List all existing sessions and exit

Session Management:
    Each pipeline run creates a session folder: results/SYMBOL_TIMEFRAME_YYYYMMDD_HHMMSS/
    The 'results/latest' symlink always points to the most recent session.

Environment Variables:
    THESIS_WORKFLOW__FORCE_RERUN=true    Force pipeline re-run
    THESIS_WORKFLOW__N_JOBS=8            Set parallel workers
    THESIS_DATA__TIMEFRAME=30m           Override timeframe
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from thesis.config.loader import load_config
from thesis.pipeline.runner import run_thesis_workflow
from thesis.pipeline.session import SessionManager


import re


_ORIGINAL_STDOUT = sys.stdout
_ORIGINAL_STDERR = sys.stderr
_PIPELINE_LOG_STREAM = None

# ANSI escape sequence regex for stripping colors
_ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


class TeeWriter:
    """Tee output to multiple streams, with ANSI stripping for file output."""

    def __init__(self, *streams, strip_ansi_for=None):
        self._streams = streams
        self._strip_ansi_for = strip_ansi_for or []

    def write(self, data: str) -> int:
        for i, stream in enumerate(self._streams):
            # Strip ANSI codes for designated streams (log file)
            if i in self._strip_ansi_for:
                clean_data = _ANSI_ESCAPE.sub("", data)
                stream.write(clean_data)
            else:
                stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        # Only return True if the first stream (console) is a tty
        # This prevents tqdm from showing progress bars in log files
        first_stream = self._streams[0]
        return getattr(first_stream, "isatty", lambda: False)()


def setup_logging(log_path: str | Path) -> logging.Logger:
    """Configure logging for the pipeline."""
    global _PIPELINE_LOG_STREAM

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if _PIPELINE_LOG_STREAM is not None and not _PIPELINE_LOG_STREAM.closed:
        _PIPELINE_LOG_STREAM.close()

    _PIPELINE_LOG_STREAM = log_path.open("a", encoding="utf-8", buffering=1)
    # Use strip_ansi_for=[1] to strip ANSI codes when writing to file stream (index 1)
    sys.stdout = TeeWriter(_ORIGINAL_STDOUT, _PIPELINE_LOG_STREAM, strip_ansi_for=[1])
    sys.stderr = TeeWriter(_ORIGINAL_STDERR, _PIPELINE_LOG_STREAM, strip_ansi_for=[1])

    # Configure optuna logging to disable colors
    import optuna.logging

    optuna.logging.disable_default_handler()
    optuna.logging.enable_propagation()  # Let our handlers handle optuna logs

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(_ORIGINAL_STDOUT),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )
    return logging.getLogger("thesis")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Hybrid Stacking Thesis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full pipeline
    python main.py
    
    # Run specific stage only
    python main.py --stage data
    
    # Force re-run with custom config
    python main.py --force --config my_config.toml
        """,
    )

    parser.add_argument(
        "--stage",
        type=str,
        choices=[
            "data",
            "features",
            "labels",
            "split",
            "lightgbm",
            "lstm",
            "stacking",
            "backtest",
            "report",
            "all",
        ],
        default="all",
        help="Run specific stage only (default: all)",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run (ignore cache)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.toml",
        help="Path to configuration file (default: config.toml)",
    )

    parser.add_argument(
        "--jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs (default: -1 = all cores)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    # Session management arguments
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Custom session ID (default: auto-generated from timestamp)",
    )

    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="List all existing sessions and exit",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Handle --list-sessions before anything else
    if args.list_sessions:
        from thesis.pipeline.session import list_sessions

        sessions = list_sessions()
        if sessions:
            print(f"Found {len(sessions)} session(s):")
            for session in sessions:
                print(f"  - {session}")
            print(f"\nLatest: results/{sessions[-1]}/")
        else:
            print("No sessions found in results/")
        return 0

    # Load configuration first (before setting up logging)
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: Configuration file not found: {e}", file=sys.stderr)
        return 1

    # Apply command-line overrides
    if args.session_id:
        config.paths.session_id = args.session_id

    if args.force:
        config.workflow.force_rerun = True

    if args.jobs != -1:
        config.workflow.n_jobs = args.jobs

    config.workflow.random_seed = args.seed

    # Initialize session (always creates session folder)
    session_manager = SessionManager(config)
    session_manager.update_config_paths(config)
    session_manager.create_config_snapshot()

    # Setup logging NOW (after session is initialized)
    log_path = session_manager.get_log_path()
    logger = setup_logging(log_path)

    logger.info("=" * 70)
    logger.info("Hybrid Stacking (LSTM + LightGBM) - XAU/USD H1 Trading Signals")
    logger.info("Bachelor's Thesis - Thuy Loi University")
    logger.info("Student: Nguyen Duc Hieu | Advisor: Hoang Quoc Dung")
    logger.info("=" * 70)

    try:
        # Log configuration summary
        logger.info(f"Data range: {config.data.start_date} to {config.data.end_date}")
        logger.info(
            f"Train: {config.splitting.train_start} → {config.splitting.train_end}"
        )
        logger.info(f"Val: {config.splitting.val_start} → {config.splitting.val_end}")
        logger.info(
            f"Test: {config.splitting.test_start} → {config.splitting.test_end}"
        )
        logger.info(f"LSTM sequence length: {config.models['lstm'].sequence_length}")
        logger.info(f"Triple-Barrier horizon: {config.labels.horizon_bars} bars")

        # Run pipeline
        if args.stage == "all":
            logger.info("Running full pipeline...")
            run_thesis_workflow(config, session_manager=session_manager)
        else:
            logger.info(f"Running stage: {args.stage}")
            run_thesis_workflow(config, stage=args.stage, session_manager=session_manager)

        logger.info("=" * 70)
        logger.info("Pipeline completed successfully!")
        logger.info(f"Results: {config.paths.final_report}")
        logger.info("=" * 70)

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
