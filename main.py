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
    --stage STAGE    Run specific stage only (data, features, labels, split, 
                      lightgbm, lstm, stacking, backtest, report)
    --force           Force re-run (ignore cache)
    --config CONFIG   Path to config file (default: config.toml)

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


def setup_logging() -> logging.Logger:
    """Configure logging for the pipeline."""
    # Ensure logs directory exists before creating file handler
    Path("logs").mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/pipeline.log", encoding="utf-8"),
        ],
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
        choices=["data", "features", "labels", "split", 
                "lightgbm", "lstm", "stacking", "backtest", "report", "all"],
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
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Setup logging (creates logs/ dir automatically)
    logger = setup_logging()
    
    logger.info("=" * 70)
    logger.info("Hybrid Stacking (LSTM + LightGBM) - XAU/USD H1 Trading Signals")
    logger.info("Bachelor's Thesis - Thuy Loi University")
    logger.info("Student: Nguyen Duc Hieu | Advisor: Hoang Quoc Dung")
    logger.info("=" * 70)
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        
        # Apply command-line overrides
        if args.force:
            config.workflow.force_rerun = True
            logger.info("Force re-run enabled (ignoring cache)")
        
        if args.jobs != -1:
            config.workflow.n_jobs = args.jobs
        
        config.workflow.random_seed = args.seed
        
        # Log configuration summary
        logger.info(f"Data range: {config.data.start_date} to {config.data.end_date}")
        logger.info(f"Train: {config.splitting.train_start} → {config.splitting.train_end}")
        logger.info(f"Val: {config.splitting.val_start} → {config.splitting.val_end}")
        logger.info(f"Test: {config.splitting.test_start} → {config.splitting.test_end}")
        logger.info(f"LSTM sequence length: {config.models['lstm'].sequence_length}")
        logger.info(f"Triple-Barrier horizon: {config.labels.horizon_bars} bars")
        
        # Run pipeline
        if args.stage == "all":
            logger.info("Running full pipeline...")
            run_thesis_workflow(config)
        else:
            logger.info(f"Running stage: {args.stage}")
            run_thesis_workflow(config, stage=args.stage)
        
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
