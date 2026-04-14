"""Pipeline orchestration — sequential stage runner with cache checking."""

import logging
from pathlib import Path

from thesis.config import Config
from thesis.prepare import prepare_data
from thesis.features import generate_features
from thesis.labels import generate_labels
from thesis.data import split_data
from thesis.model import train_model
from thesis.backtest import run_backtest
from thesis.report import generate_report
from thesis.visualize import generate_all_charts

logger = logging.getLogger("thesis.pipeline")

# ---------------------------------------------------------------------------
# ANSI color codes for stage identification
# ---------------------------------------------------------------------------

_BOLD = "\033[1m"
_RESET = "\033[0m"

_STAGE_COLORS: dict[str, str] = {
    "prepare": "\033[94m",  # Blue
    "features": "\033[92m",  # Green
    "labels": "\033[93m",  # Yellow
    "split": "\033[96m",  # Cyan
    "train": "\033[95m",  # Magenta
    "backtest": "\033[91m",  # Red
    "report": "\033[97m",  # White/Bright
}

_SKIP_COLOR = "\033[90m"  # Dark gray


def _stage_log(stage_key: str, message: str) -> None:
    """Log a stage message with color coding."""
    color = _STAGE_COLORS.get(stage_key, "")
    logger.info("%s%s%s", f"{_BOLD}{color}▶ {message}{_RESET}", "", "")


def _skip_log(message: str) -> None:
    """Log a skip message in gray."""
    logger.info("%s⊘ %s%s", _SKIP_COLOR, message, _RESET)


def run_pipeline(config: Config) -> None:
    """Execute the full thesis pipeline sequentially.

    Stages:
        0. Data preparation (tick to OHLCV)
        1. Feature engineering
        2. Triple-barrier labeling
        3. Data splitting (train/val/test)
        4. Model training (GRU feature extractor + LightGBM)
        5. Backtest
        6. Report generation

    Args:
        config: Loaded application configuration.
    """
    force = config.workflow.force_rerun

    # Stage 0: Prepare OHLCV from raw ticks
    if config.workflow.run_data_pipeline:
        ohlcv_path = Path(config.paths.ohlcv)
        if force or not ohlcv_path.exists():
            _stage_log("prepare", "STAGE 0/6: Data Preparation (Tick → OHLCV)")
            prepare_data(config)
        else:
            _skip_log(f"SKIP: OHLCV already exists ({ohlcv_path})")
    else:
        _skip_log("SKIP: Data preparation disabled")

    # Stage 1: Features
    if config.workflow.run_feature_engineering:
        features_path = Path(config.paths.features)
        if force or not features_path.exists():
            _stage_log("features", "STAGE 1/6: Feature Engineering")
            generate_features(config)
        else:
            _skip_log(f"SKIP: Features already exist ({features_path})")
    else:
        _skip_log("SKIP: Feature engineering disabled")

    # Stage 2: Labels
    if config.workflow.run_label_generation:
        labels_path = Path(config.paths.labels)
        if force or not labels_path.exists():
            _stage_log("labels", "STAGE 2/6: Triple-Barrier Labeling")
            generate_labels(config)
        else:
            _skip_log(f"SKIP: Labels already exist ({labels_path})")
    else:
        _skip_log("SKIP: Label generation disabled")

    # Stage 3: Split
    if config.workflow.run_data_splitting:
        train_path = Path(config.paths.train_data)
        if force or not train_path.exists():
            _stage_log("split", "STAGE 3/6: Data Splitting (Train/Val/Test)")
            split_data(config)
        else:
            _skip_log(f"SKIP: Splits already exist ({train_path})")
    else:
        _skip_log("SKIP: Data splitting disabled")

    # Stage 4: Model training
    if config.workflow.run_model_training:
        model_path = Path(config.paths.model)
        preds_path = Path(config.paths.predictions)
        if force or not model_path.exists() or not preds_path.exists():
            _stage_log("train", "STAGE 4/6: Model Training (GRU + LightGBM)")
            train_model(config)
        else:
            _skip_log(f"SKIP: Model already exists ({model_path})")
    else:
        _skip_log("SKIP: Model training disabled")

    # Stage 5: Backtest
    if config.workflow.run_backtest:
        _stage_log("backtest", "STAGE 5/6: Backtest (CFD Simulation)")
        run_backtest(config)
    else:
        _skip_log("SKIP: Backtest disabled")

    # Stage 6: Report
    if config.workflow.run_reporting:
        _stage_log("report", "STAGE 6/6: Report Generation")
        generate_report(config)
        generate_all_charts(config)
    else:
        _skip_log("SKIP: Reporting disabled")

    logger.info("✓ Pipeline complete.")
