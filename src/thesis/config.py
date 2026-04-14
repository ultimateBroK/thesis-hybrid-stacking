"""Simplified configuration for thesis pipeline.

Uses ``tomllib`` (stdlib) + 3 dataclasses to load a flat TOML config.
No environment-variable overrides, no session-aware ATR, no LSTM config.
"""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Dataclasses — one per TOML section
# ---------------------------------------------------------------------------


@dataclass
class DataConfig:
    """Data loading and OHLCV parameters."""

    symbol: str = "XAUUSD"
    timeframe: str = "1H"
    market_tz: str = "America/New_York"
    start_date: str = "2018-01-01"
    end_date: str = "2026-03-31"
    tick_size: float = 0.01
    contract_size: int = 100
    symbol_download: str = "XAUUSD"
    asset_class: str = "fx"
    download_concurrency: int = 20


@dataclass
class SplittingConfig:
    """Train / val / test date ranges and leakage prevention."""

    train_start: str = "2018-01-01"
    train_end: str = "2022-12-31 23:59:59"
    val_start: str = "2023-01-01"
    val_end: str = "2023-12-31 23:59:59"
    test_start: str = "2024-01-01"
    test_end: str = "2026-03-31 23:59:59"
    purge_bars: int = 25
    embargo_bars: int = 10


@dataclass
class FeaturesConfig:
    """Feature engineering parameters."""

    rsi_period: int = 14
    atr_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    correlation_threshold: float = 0.90


@dataclass
class LabelsConfig:
    """Triple-barrier label parameters (single ATR multiplier, no sessions)."""

    atr_multiplier: float = 1.5
    horizon_bars: int = 10
    num_classes: int = 3
    min_atr: float = 0.0001


@dataclass
class ModelConfig:
    """LightGBM parameters."""

    # LightGBM
    use_optuna: bool = False
    optuna_trials: int = 50
    optuna_timeout: int = 3600
    num_leaves: int = 48
    max_depth: int = 5
    learning_rate: float = 0.03
    n_estimators: int = 150
    min_child_samples: int = 150
    subsample: float = 0.70
    subsample_freq: int = 5
    feature_fraction: float = 0.60
    reg_alpha: float = 0.1
    reg_lambda: float = 5.0
    early_stopping_rounds: int = 30


@dataclass
class BacktestConfig:
    """CFD backtest parameters — thin wrapper for backtesting.py."""

    initial_capital: float = 10_000.0
    leverage: int = 30  # margin = 1/leverage
    spread_ticks: float = 30.0  # → spread param (relative)
    slippage_ticks: float = 3.0  # absorbed into spread
    commission_per_lot: float = 10.0  # → callable commission
    atr_stop_multiplier: float = 0.75
    lots_per_trade: float = 1.0  # fixed lot size per trade
    confidence_threshold: float = 0.0  # min predicted probability to act (0 = disabled)
    contract_size: int = 100
    tick_size: float = 0.01


@dataclass
class GRUConfig:
    """GRU feature extractor parameters."""

    input_size: int = 2  # log_returns + rsi_14
    hidden_size: int = 64
    num_layers: int = 2
    sequence_length: int = 24
    dropout: float = 0.3
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 50
    patience: int = 10


@dataclass
class WorkflowConfig:
    """Pipeline execution toggles and seeds."""

    run_data_pipeline: bool = True
    run_feature_engineering: bool = True
    run_label_generation: bool = True
    run_data_splitting: bool = True
    run_model_training: bool = True
    run_backtest: bool = True
    run_reporting: bool = True
    force_rerun: bool = False
    random_seed: int = 2024
    n_jobs: int = -1
    session_timestamp: str = ""  # Set at runtime


@dataclass
class PathsConfig:
    """Artifact paths with session-based output support."""

    data_raw: str = "data/raw/XAUUSD"
    data_processed: str = "data/processed"
    ohlcv: str = "data/processed/ohlcv.parquet"
    features: str = "data/processed/features.parquet"
    labels: str = "data/processed/labels.parquet"
    train_data: str = "data/processed/train.parquet"
    val_data: str = "data/processed/val.parquet"
    test_data: str = "data/processed/test.parquet"
    model: str = "models/lightgbm_model.pkl"
    gru_model: str = "models/gru_model.pt"
    predictions: str = "data/predictions/final_predictions.parquet"
    backtest_results: str = "results/backtest_results.json"
    report: str = "results/thesis_report.md"
    session_dir: str = ""  # Set at runtime, e.g. "results/XAUUSD_1H_20260414_042000"


# ---------------------------------------------------------------------------
# Top-level Config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """Main configuration — one attribute per TOML section."""

    data: DataConfig = field(default_factory=DataConfig)
    splitting: SplittingConfig = field(default_factory=SplittingConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    labels: LabelsConfig = field(default_factory=LabelsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    gru: GRUConfig = field(default_factory=GRUConfig)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_SECTION_MAP: dict[str, type] = {
    "data": DataConfig,
    "splitting": SplittingConfig,
    "features": FeaturesConfig,
    "labels": LabelsConfig,
    "model": ModelConfig,
    "backtest": BacktestConfig,
    "gru": GRUConfig,
    "workflow": WorkflowConfig,
    "paths": PathsConfig,
}


def load_config(config_path: str | Path = "config.toml") -> Config:
    """Load configuration from a flat TOML file.

    Args:
        config_path: Path to TOML configuration file.

    Returns:
        Fully populated ``Config`` object.

    Raises:
        FileNotFoundError: If *config_path* does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "rb") as f:
        raw: dict[str, Any] = tomllib.load(f)

    cfg = Config()
    for section, cls in _SECTION_MAP.items():
        if section in raw:
            setattr(cfg, section, cls(**raw[section]))

    # Ensure base directories exist
    Path(cfg.paths.data_processed).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.data_raw).mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)

    return cfg
