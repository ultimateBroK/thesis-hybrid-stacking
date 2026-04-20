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


def _run_stage(
    stage_num: int,
    config: Config,
    flag_name: str,
    cache_path: str | Path | None,
    work_fn: callable,
) -> None:
    """Execute a pipeline stage with cache checking.

    Args:
        stage_num: Stage number for display.
        config: Application configuration.
        flag_name: Name of the workflow flag in config (e.g., 'run_data_pipeline').
        cache_path: Path to check for cached output; if None, always runs.
        work_fn: Function to call if stage should run.
    """
    flag = getattr(config.workflow, flag_name, False)
    if not flag:
        stage_skip(stage_num, "disabled")
        return

    if cache_path is not None:
        cache_path = Path(cache_path)
        if not config.workflow.force_rerun and cache_path.exists():
            stage_skip(stage_num, f"cached ({cache_path.name})")
            return

    stage_header(stage_num)
    work_fn(config)


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
    # Stage 0: Prepare OHLCV from raw ticks
    _run_stage(0, config, "run_data_pipeline", config.paths.ohlcv, prepare_data)

    # Stage 1: Features
    _run_stage(
        1, config, "run_feature_engineering", config.paths.features, generate_features
    )

    # Stage 2: Labels
    _run_stage(2, config, "run_label_generation", config.paths.labels, generate_labels)

    # Stage 3: Split
    _run_stage(3, config, "run_data_splitting", config.paths.train_data, split_data)

    # Stage 4: Model training
    def _run_train() -> None:
        train_model(config)

    _run_stage(4, config, "run_model_training", None, lambda cfg: _run_train())

    # Stage 5: Backtest
    _run_stage(5, config, "run_backtest", None, run_backtest)

    # Stage 6: Report
    def _run_report() -> None:
        generate_report(config)
        generate_all_charts(config)

    _run_stage(6, config, "run_reporting", None, lambda cfg: _run_report())

    console.print()
    console.rule("[bold green]✓ Pipeline Complete[/]")
    console.print()
