"""Load and validate pipeline configuration."""

from dataclasses import dataclass, field, fields
from functools import lru_cache
import logging
from pathlib import Path
import tomllib
from typing import Any

from thesis.shared.constants import CORE_STATIC_FEATURES

logger = logging.getLogger("thesis.config")


@dataclass
class DataConfig:
    """Market data settings (excluding data range)."""

    symbol: str = "XAUUSD"
    timeframe: str = "1H"
    market_tz: str = "America/New_York"
    tick_size: float = 0.01
    contract_size: int = 100
    symbol_download: str = "XAUUSD"
    asset_class: str = "fx"
    download_concurrency: int = 20
    download_max_retries: int = 7
    download_force: bool = False
    download_skip_current_month: bool = True


@dataclass
class DataRangeConfig:
    """Data ingestion range (replaces former splitting dates)."""

    start: str = "2021-01-01"
    end: str = "2026-04-30"


@dataclass
class ValidationConfig:
    """Walk-forward validation window sizes."""

    method: str = "sliding"
    train_window_bars: int = 6240
    test_window_bars: int = 1040
    step_bars: int = 1040
    purge_bars: int = 48
    embargo_bars: int = 50
    min_train_bars: int = 6000
    oof_ensemble: bool = True


@dataclass
class MultiTimeframeConfig:
    """Derived multi-timeframe feature params.

    Not exposed in config.toml by default — internal knobs for
    regime detection, rolling windows, and ATR multi-horizon.
    """

    sma_periods: list[int] = field(default_factory=lambda: [50])
    ema_long: int = 200
    bb_period: int = 20
    bb_std: float = 2.0
    return_lookbacks: list[int] = field(default_factory=lambda: [1, 4, 24])
    range_lookback: int = 20
    volume_zscore_period: int = 20
    atr_short_period: int = 5
    atr_long_period: int = 20
    consecutive_bars_window: int = 5
    price_position_window: int = 20
    atr_percentile_window: int = 50
    ohlcv_norm_window: int = 20
    ema_slope_shift: int = 5
    # Volatility regime bucket thresholds (atr_percentile → 0/1/2)
    vol_regime_p33: float = 0.33
    vol_regime_p66: float = 0.66


@dataclass
class FeaturesConfig:
    """Indicator and tabular feature settings."""

    rsi_period: int = 14
    atr_period: int = 14
    adx_period: int = 14
    ema_slope_period: int = 20
    adx_regime_threshold: float = 20.0
    adx_regime_clip_max: float = 3.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    ema_fast_span: int = 34
    ema_slow_span: int = 89
    correlation_threshold: float = 0.75
    static_feature_cols: list[str] = field(
        default_factory=lambda: list(CORE_STATIC_FEATURES)
    )
    multi_timeframe: MultiTimeframeConfig = field(default_factory=MultiTimeframeConfig)


@dataclass
class LabelsConfig:
    """Triple-barrier label settings."""

    atr_tp_multiplier: float = 2.5
    atr_sl_multiplier: float = 2.5
    horizon_bars: int = 24
    num_classes: int = 3
    min_atr: float = 0.5


@dataclass
class LightGBMConfig:
    """LightGBM hyperparameters."""

    num_leaves: int = 31
    max_depth: int = 6
    learning_rate: float = 0.03
    n_estimators: int = 800
    min_child_samples: int = 40
    subsample: float = 0.80
    subsample_freq: int = 5
    feature_fraction: float = 0.80
    reg_alpha: float = 0.05
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 30
    lgbm_expanded_features: bool = False


@dataclass
class RandomForestConfig:
    """Random Forest hyperparameters."""

    n_estimators: int = 300
    max_depth: int = 6
    min_samples_leaf: int = 80


@dataclass
class LogisticRegressionConfig:
    """Logistic Regression hyperparameters (base model)."""

    C: float = 1.0
    max_iter: int = 1000
    solver: str = "lbfgs"


@dataclass
class StackingConfig:
    """Stacking ensemble settings."""

    meta_fraction: float = 0.20


@dataclass
class StackingMetaConfig:
    """Meta learner hyperparameters for stacking."""

    learner: str = "logistic_regression"  # "logistic_regression" or "lightgbm"
    # LR params
    meta_C: float = 1.0
    max_iter: int = 1000
    solver: str = "lbfgs"
    penalty: str = "l2"
    l1_ratio: float | None = None
    # LightGBM params
    num_leaves: int = 7
    max_depth: int = 2
    learning_rate: float = 0.03
    n_estimators: int = 150
    min_child_samples: int = 80
    subsample: float = 0.80
    subsample_freq: int = 1
    feature_fraction: float = 1.0
    reg_alpha: float = 0.5
    reg_lambda: float = 5.0
    min_split_gain: float = 0.01


@dataclass
class ModelConfig:
    """Model training configuration.

    Architecture: Hybrid Stacking (LR + RF + LightGBM → meta learner).
    """

    logistic_regression: LogisticRegressionConfig = field(
        default_factory=LogisticRegressionConfig
    )
    lightgbm: LightGBMConfig = field(default_factory=LightGBMConfig)
    random_forest: RandomForestConfig = field(default_factory=RandomForestConfig)
    stacking: StackingConfig = field(default_factory=StackingConfig)
    stacking_meta: StackingMetaConfig = field(default_factory=StackingMetaConfig)


@dataclass
class BacktestConfig:
    """Trading simulation settings."""

    initial_capital: float = 10_000.0
    leverage: int = 10
    spread_ticks: float = 35.0
    slippage_ticks: float = 5.0
    commission_per_lot: float = 10.0
    atr_stop_multiplier: float = 2.5
    atr_tp_multiplier: float = 2.5
    lots_per_trade: float = 0.02
    min_lots: float = 0.01
    max_lots: float = 0.5
    confidence_threshold: float = 0.50
    min_bars_between_trades: int = 18
    max_drawdown_cutoff: float = 0.30
    dd_cooldown_bars: int = 12
    max_open_positions: int = 1
    daily_loss_limit: float = 0.03
    oob_start_date: str = ""
    oob_end_date: str = ""


@dataclass
class WorkflowConfig:
    """Stage toggles, caching, and reproducibility settings."""

    run_data: bool = True
    run_dataset: bool = True
    run_models: bool = True
    run_reporting: bool = True
    run_backtest_demo: bool = True
    cache_invalidation: str = "path"
    force_rerun: bool = False
    random_seed: int = 2024
    n_jobs: int = -1
    session_timestamp: str = ""


@dataclass
class ReportFiguresConfig:
    """Static chart export settings for thesis report."""

    enabled: bool = True
    format: str = "png"
    dpi: int = 180
    top_n_features: int = 15


@dataclass
class PathsConfig:
    """Default artifact paths."""

    data_raw: str = "data/raw/XAUUSD"
    data_processed: str = "data/processed"
    ohlcv: str = "data/processed/ohlcv.parquet"
    features: str = "data/processed/features.parquet"
    labels: str = "data/processed/labels.parquet"
    ml_dataset: str = "data/modeling/ml_dataset.parquet"
    model: str = "models/lightgbm_model.pkl"
    predictions: str = "data/predictions/final_predictions.csv"
    report: str = "results/thesis_report.md"
    backtest_results: str = "backtest/backtest_results.json"
    data_quality_json: str = "data/processed/data_summary.json"
    test_data: str = ""
    session_dir: str = ""


@dataclass
class Config:
    """Runtime configuration grouped by TOML section."""

    data: DataConfig = field(default_factory=DataConfig)
    data_range: DataRangeConfig = field(default_factory=DataRangeConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    labels: LabelsConfig = field(default_factory=LabelsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    report_figures: ReportFiguresConfig = field(default_factory=ReportFiguresConfig)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


_SECTION_MAP: dict[str, type] = {
    "data": DataConfig,
    "data_range": DataRangeConfig,
    "validation": ValidationConfig,
    "features": FeaturesConfig,
    "labels": LabelsConfig,
    "model": ModelConfig,
    "backtest": BacktestConfig,
    "report_figures": ReportFiguresConfig,
    "workflow": WorkflowConfig,
    "paths": PathsConfig,
}


def _section_kwargs(section: str, cls: type, data: dict[str, Any]) -> dict[str, Any]:
    """Keep known keys and fail fast on misspelled config keys."""
    known = {item.name for item in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        keys = ", ".join(unknown)
        raise ValueError(f"Unknown config key(s) in [{section}]: {keys}")
    return data


def _apply_section(
    cfg: Config, section: str, cls: type, values: dict[str, Any]
) -> None:
    """Apply one validated TOML section to a Config object."""
    section_data = dict(values)
    if section == "features":
        mt_data = section_data.pop("multi_timeframe", None)
        cfg.features = FeaturesConfig(**_section_kwargs(section, cls, section_data))
        if mt_data is not None:
            cfg.features.multi_timeframe = MultiTimeframeConfig(
                **_section_kwargs(
                    "features.multi_timeframe", MultiTimeframeConfig, mt_data
                )
            )
        return

    if section == "model":
        lr_data = section_data.pop("logistic_regression", None)
        lgbm_data = section_data.pop("lightgbm", None)
        rf_data = section_data.pop("random_forest", None)
        stacking_data = section_data.pop("stacking", None)
        meta_data = section_data.pop("stacking_meta", None)
        cfg.model = ModelConfig(**_section_kwargs(section, cls, section_data))
        if lr_data is not None:
            cfg.model.logistic_regression = LogisticRegressionConfig(
                **_section_kwargs(
                    "model.logistic_regression", LogisticRegressionConfig, lr_data
                )
            )
        if lgbm_data is not None:
            cfg.model.lightgbm = LightGBMConfig(
                **_section_kwargs("model.lightgbm", LightGBMConfig, lgbm_data)
            )
        if rf_data is not None:
            cfg.model.random_forest = RandomForestConfig(
                **_section_kwargs("model.random_forest", RandomForestConfig, rf_data)
            )
        if stacking_data is not None:
            cfg.model.stacking = StackingConfig(
                **_section_kwargs("model.stacking", StackingConfig, stacking_data)
            )
        if meta_data is not None:
            cfg.model.stacking_meta = StackingMetaConfig(
                **_section_kwargs("model.stacking_meta", StackingMetaConfig, meta_data)
            )
        return

    setattr(cfg, section, cls(**_section_kwargs(section, cls, section_data)))


def load_config(config_path: str | Path = "config.toml") -> Config:
    """Load a TOML config and fill omitted values from dataclass defaults."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "rb") as f:
        raw: dict[str, Any] = tomllib.load(f)

    unknown_sections = sorted(set(raw) - set(_SECTION_MAP))
    if unknown_sections:
        logger.warning(
            "Ignoring unknown config section(s): %s", ", ".join(unknown_sections)
        )

    cfg = Config()
    for section, cls in _SECTION_MAP.items():
        if section in raw:
            _apply_section(cfg, section, cls, raw[section])

    Path(cfg.paths.data_processed).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.data_raw).mkdir(parents=True, exist_ok=True)
    return cfg


@lru_cache(maxsize=8)
def get_config(config_path: str | Path = "config.toml") -> Config:
    """Return a cached config for scripts, dashboards, and reports."""
    return load_config(Path(config_path))


def reload_config(config_path: str | Path = "config.toml") -> Config:
    """Clear the config cache, then load a fresh config."""
    get_config.cache_clear()
    return get_config(config_path)
