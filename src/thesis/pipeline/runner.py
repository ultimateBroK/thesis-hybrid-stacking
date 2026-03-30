"""Pipeline orchestration for thesis workflow.

Stages:
    1. Data Processing (ticks → OHLCV H1)
    2. Feature Engineering (technical indicators, session encoding)
    3. Label Generation (Triple-Barrier)
    4. Data Splitting (train/val/test with purge/embargo)
    5. LightGBM Model Training
    6. LSTM Model Training
    7. Hybrid Stacking Meta-Learner
    8. Backtesting
    9. Reporting
"""

import logging
from pathlib import Path
from typing import Optional

from thesis.config.loader import Config

logger = logging.getLogger("thesis.pipeline")


class PipelineStage:
    """Base class for pipeline stages."""

    def __init__(self, name: str, config: Config):
        self.name = name
        self.config = config
        self.output_path: Optional[Path] = None

    def check_cache(self) -> bool:
        """Check if stage output exists and is valid."""
        if self.config.workflow.force_rerun:
            return False
        if self.output_path and self.output_path.exists():
            logger.info(f"  Cache hit: {self.output_path}")
            return True
        return False

    def run(self) -> None:
        """Execute the stage. Must be implemented by subclasses."""
        raise NotImplementedError


def run_thesis_workflow(config: Config, stage: Optional[str] = None) -> None:
    """Run the thesis pipeline workflow.

    Args:
        config: Configuration object with all settings.
        stage: Specific stage to run, or None for full pipeline.
    """
    logger.info("=" * 70)
    logger.info("STARTING THESIS PIPELINE")
    logger.info("=" * 70)

    stages = [
        ("data", _run_data_stage),
        ("features", _run_features_stage),
        ("labels", _run_labels_stage),
        ("split", _run_split_stage),
        ("lightgbm", _run_lightgbm_stage),
        ("lstm", _run_lstm_stage),
        ("stacking", _run_stacking_stage),
        ("backtest", _run_backtest_stage),
        ("report", _run_report_stage),
    ]

    if stage and stage != "all":
        # Run specific stage
        stage_map = {name: func for name, func in stages}
        if stage in stage_map:
            logger.info(f"Running stage: {stage}")
            stage_map[stage](config)
        else:
            raise ValueError(f"Unknown stage: {stage}")
    else:
        # Run full pipeline
        for stage_name, stage_func in stages:
            # Check if stage is enabled in config
            if not _is_stage_enabled(stage_name, config):
                logger.info(f"Skipping disabled stage: {stage_name}")
                continue

            logger.info("")
            logger.info(f"{'=' * 70}")
            logger.info(f"STAGE {stage_name.upper()}")
            logger.info(f"{'=' * 70}")

            try:
                stage_func(config)
            except Exception as e:
                logger.exception(f"Stage {stage_name} failed: {e}")
                raise

    logger.info("\n" + "=" * 70)
    logger.info("ALL STAGES COMPLETED")
    logger.info("=" * 70)


def _is_stage_enabled(stage_name: str, config: Config) -> bool:
    """Check if a stage is enabled in the workflow config."""
    mapping = {
        "data": config.workflow.run_data_pipeline,
        "features": config.workflow.run_feature_engineering,
        "labels": config.workflow.run_label_generation,
        "split": config.workflow.run_data_splitting,
        "lightgbm": config.workflow.run_lightgbm,
        "lstm": config.workflow.run_lstm,
        "stacking": config.workflow.run_stacking,
        "backtest": config.workflow.run_backtest,
        "report": config.workflow.run_reporting,
    }
    return mapping.get(stage_name, True)


def _run_data_stage(config: Config) -> None:
    """Stage 1: Process raw tick data to OHLCV H1."""
    from thesis.data.tick_to_ohlcv import process_all_tick_files

    output_path = Path(config.data.ohlcv_path)

    if config.workflow.force_rerun or not output_path.exists():
        logger.info("Processing raw tick data to OHLCV H1...")
        logger.info(f"  Source: {config.data.raw_data_path}")
        logger.info(f"  Output: {output_path}")

        process_all_tick_files(config)
        logger.info(f"  Saved OHLCV data: {output_path}")
    else:
        logger.info(f"  Using cached OHLCV: {output_path}")


def _run_features_stage(config: Config) -> None:
    """Stage 2: Generate features from OHLCV data."""
    from thesis.features.engineering import generate_features

    output_path = Path(config.features.features_path)

    if config.workflow.force_rerun or not output_path.exists():
        logger.info("Generating technical features...")
        logger.info(f"  Input: {config.data.ohlcv_path}")
        logger.info(f"  Output: {output_path}")

        # Check if input exists
        if not Path(config.data.ohlcv_path).exists():
            logger.info("  Running data stage first (dependency)...")
            _run_data_stage(config)

        generate_features(config)
        logger.info(f"  Saved features: {output_path}")
    else:
        logger.info(f"  Using cached features: {output_path}")


def _run_labels_stage(config: Config) -> None:
    """Stage 3: Generate Triple-Barrier labels."""
    from thesis.labels.triple_barrier import generate_labels

    output_path = Path(config.labels.labels_path)

    if config.workflow.force_rerun or not output_path.exists():
        logger.info("Generating Triple-Barrier labels...")
        logger.info(f"  Input: {config.features.features_path}")
        logger.info(f"  Output: {output_path}")
        logger.info(
            f"  Params: TP={config.labels.atr_multiplier_tp}×ATR, SL={config.labels.atr_multiplier_sl}×ATR, horizon={config.labels.horizon_bars} bars"
        )

        # Check dependencies
        if not Path(config.features.features_path).exists():
            logger.info("  Running features stage first (dependency)...")
            _run_features_stage(config)

        generate_labels(config)
        logger.info(f"  Saved labels: {output_path}")
    else:
        logger.info(f"  Using cached labels: {output_path}")


def _run_split_stage(config: Config) -> None:
    """Stage 4: Split data into train/val/test with purge/embargo."""
    from thesis.data.splitting import split_data

    train_path = Path(config.paths.train_data)
    val_path = Path(config.paths.val_data)
    test_path = Path(config.paths.test_data)

    if config.workflow.force_rerun or not all(
        [train_path.exists(), val_path.exists(), test_path.exists()]
    ):
        logger.info("Splitting data into train/val/test...")
        logger.info(
            f"  Train: {config.splitting.train_start} → {config.splitting.train_end} (60%)"
        )
        logger.info(
            f"  Val: {config.splitting.val_start} → {config.splitting.val_end} (15%) - High-rate + Gold ATH (2023)"
        )
        logger.info(
            f"  Test: {config.splitting.test_start} → {config.splitting.test_end} (25%) - New Regime (2024-2026)"
        )
        logger.info(
            f"  Purge: {config.splitting.purge_bars} bars | Embargo: {config.splitting.embargo_bars} bars"
        )

        # Check dependencies
        if not Path(config.labels.labels_path).exists():
            logger.info("  Running labels stage first (dependency)...")
            _run_labels_stage(config)

        split_data(config)
        logger.info("  Saved splits:")
        logger.info(f"    Train: {train_path}")
        logger.info(f"    Val: {val_path}")
        logger.info(f"    Test: {test_path}")
    else:
        logger.info("  Using cached data splits")


def _run_lightgbm_stage(config: Config) -> None:
    """Stage 5: Train LightGBM model."""
    from thesis.models.lightgbm_model import train_lightgbm

    model_path = Path(config.models["tree"].model_path)
    predictions_path = Path(config.models["tree"].predictions_path)

    if config.workflow.force_rerun or not model_path.exists():
        logger.info("Training LightGBM model...")
        logger.info(f"  Train data: {config.paths.train_data}")
        logger.info(f"  Val data: {config.paths.val_data}")

        if config.models["tree"].use_optuna:
            logger.info(
                f"  Hyperparameter tuning: {config.models['tree'].optuna_trials} trials"
            )

        # Check dependencies
        if not Path(config.paths.train_data).exists():
            logger.info("  Running split stage first (dependency)...")
            _run_split_stage(config)

        train_lightgbm(config)
        logger.info(f"  Saved model: {model_path}")
        logger.info(f"  Saved predictions: {predictions_path}")
    else:
        logger.info(f"  Using cached LightGBM: {model_path}")


def _run_lstm_stage(config: Config) -> None:
    """Stage 6: Train LSTM model."""
    from thesis.models.lstm_model import train_lstm

    model_path = Path(config.models["lstm"].model_path)
    predictions_path = Path(config.models["lstm"].predictions_path)

    if config.workflow.force_rerun or not model_path.exists():
        logger.info("Training LSTM model...")
        logger.info(f"  Sequence length: {config.models['lstm'].sequence_length} bars")
        logger.info(f"  Hidden size: {config.models['lstm'].hidden_size}")
        logger.info(f"  Device: {config.models['lstm'].device}")

        # Check dependencies
        if not Path(config.paths.train_data).exists():
            logger.info("  Running split stage first (dependency)...")
            _run_split_stage(config)

        train_lstm(config)
        logger.info(f"  Saved model: {model_path}")
        logger.info(f"  Saved predictions: {predictions_path}")
    else:
        logger.info(f"  Using cached LSTM: {model_path}")


def _run_stacking_stage(config: Config) -> None:
    """Stage 7: Train Hybrid Stacking meta-learner."""
    from thesis.models.stacking import train_stacking

    model_path = Path(config.models["stacking"].model_path)
    predictions_path = Path(config.models["stacking"].meta_predictions_path)

    if config.workflow.force_rerun or not model_path.exists():
        logger.info("Training Hybrid Stacking meta-learner...")
        logger.info(f"  Meta-learner: {config.models['stacking'].meta_learner}")
        logger.info(f"  LightGBM OOF: {config.models['tree'].predictions_path}")
        logger.info(f"  LSTM OOF: {config.models['lstm'].predictions_path}")

        # Check dependencies
        lgbm_oof = Path(config.models["tree"].predictions_path)
        lstm_oof = Path(config.models["lstm"].predictions_path)

        if not lgbm_oof.exists():
            logger.info("  Running LightGBM stage first (dependency)...")
            _run_lightgbm_stage(config)

        if not lstm_oof.exists():
            logger.info("  Running LSTM stage first (dependency)...")
            _run_lstm_stage(config)

        train_stacking(config)
        logger.info(f"  Saved meta-learner: {model_path}")
        logger.info(f"  Saved predictions: {predictions_path}")
    else:
        logger.info(f"  Using cached Stacking: {model_path}")


def _run_backtest_stage(config: Config) -> None:
    """Stage 8: Run CFD backtest."""
    from thesis.backtest.cfd_simulator import run_backtest
    from thesis.models.stacking import generate_test_predictions

    results_path = Path(config.backtest.backtest_results_path)
    final_preds_path = Path(config.paths.final_predictions)

    # Check if backtest results are stale (input predictions newer than results)
    needs_rebuild = config.workflow.force_rerun or not results_path.exists()
    if not needs_rebuild and results_path.exists() and final_preds_path.exists():
        results_mtime = results_path.stat().st_mtime
        preds_mtime = final_preds_path.stat().st_mtime
        if preds_mtime > results_mtime:
            logger.info(
                "  Detected stale backtest results (predictions newer than results)"
            )
            needs_rebuild = True

    if needs_rebuild:
        logger.info("Running CFD backtest...")
        logger.info(f"  Initial capital: ${config.backtest.initial_capital:,.0f}")
        logger.info(f"  Leverage: {config.backtest.leverage}:1")
        logger.info(f"  Spread: {config.backtest.spread_pips} pips")
        logger.info(f"  Risk per trade: {config.backtest.risk_per_trade * 100:.0f}%")

        # Check dependencies - ensure we have trained models
        if not Path(config.models["stacking"].model_path).exists():
            logger.info("  Running stacking stage first (dependency)...")
            _run_stacking_stage(config)

        # Generate test predictions if not exists or force rerun
        if config.workflow.force_rerun or not final_preds_path.exists():
            logger.info("  Generating test set predictions...")
            generate_test_predictions(config)
        else:
            logger.info(f"  Using cached test predictions: {final_preds_path}")

        run_backtest(config)
        logger.info(f"  Saved backtest results: {results_path}")
    else:
        logger.info(f"  Using cached backtest: {results_path}")


def _run_report_stage(config: Config) -> None:
    """Stage 9: Generate final report with SHAP analysis."""
    from thesis.reporting.thesis_report import generate_report

    report_path = Path(config.reporting.report_path)

    logger.info("Generating thesis report...")
    logger.info(f"  Report: {report_path}")
    logger.info(f"  SHAP samples: {config.reporting.shap_samples}")

    # Check dependencies
    if not Path(config.backtest.backtest_results_path).exists():
        logger.info("  Running backtest stage first (dependency)...")
        _run_backtest_stage(config)

    generate_report(config)
    logger.info(f"  Saved report: {report_path}")
