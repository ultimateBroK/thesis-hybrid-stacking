"""Pipeline orchestration — sequential stage runner with cache checking."""

import logging
from pathlib import Path

from thesis.config import Config
from thesis.ui import console, stage_header, stage_skip
from thesis.agg import prepare_data
from thesis.features import generate_features
from thesis.labeling import generate_labels
from thesis.splitting import split_data
from thesis.hybrid import train_model
from thesis.backtest import run_backtest
from thesis.report import generate_report
from thesis.plots import generate_all_charts

logger = logging.getLogger("thesis.pipeline")


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
            stage_header(0)
            prepare_data(config)
        else:
            stage_skip(0, f"OHLCV exists ({ohlcv_path.name})")
    else:
        stage_skip(0, "disabled")

    # Stage 1: Features
    if config.workflow.run_feature_engineering:
        features_path = Path(config.paths.features)
        if force or not features_path.exists():
            stage_header(1)
            generate_features(config)
        else:
            stage_skip(1, f"Features exist ({features_path.name})")
    else:
        stage_skip(1, "disabled")

    # Stage 2: Labels
    if config.workflow.run_label_generation:
        labels_path = Path(config.paths.labels)
        if force or not labels_path.exists():
            stage_header(2)
            generate_labels(config)
        else:
            stage_skip(2, f"Labels exist ({labels_path.name})")
    else:
        stage_skip(2, "disabled")

    # Stage 3: Split
    if config.workflow.run_data_splitting:
        train_path = Path(config.paths.train_data)
        if force or not train_path.exists():
            stage_header(3)
            split_data(config)
        else:
            stage_skip(3, f"Splits exist ({train_path.name})")
    else:
        stage_skip(3, "disabled")

    # Stage 4: Model training
    if config.workflow.run_model_training:
        model_path = Path(config.paths.model)
        preds_path = Path(config.paths.predictions)
        if force or not model_path.exists() or not preds_path.exists():
            stage_header(4)
            train_model(config)
        else:
            stage_skip(4, f"Model exists ({model_path.name})")
    else:
        stage_skip(4, "disabled")

    # Stage 5: Backtest
    if config.workflow.run_backtest:
        stage_header(5)
        run_backtest(config)
    else:
        stage_skip(5, "disabled")

    # Stage 6: Report
    if config.workflow.run_reporting:
        stage_header(6)
        generate_report(config)
        generate_all_charts(config)
    else:
        stage_skip(6, "disabled")

    console.print()
    console.rule("[bold green]✓ Pipeline Complete[/]")
    console.print()
