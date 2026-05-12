"""Load and validate pipeline configuration."""

from dataclasses import dataclass, field, fields
from functools import lru_cache
import logging
from pathlib import Path
import re
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
    """Hidden defaults for derived multi-timeframe features."""

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

    atr_tp_multiplier: float = 2.0
    atr_sl_multiplier: float = 2.0
    horizon_bars: int = 24
    num_classes: int = 3
    min_atr: float = 0.5


@dataclass
class LGBMConfig:
    """Tabular model and stacking settings."""

    architecture: str = "stacking"
    objective: str = "multiclass"
    lgbm_expanded_features: bool = False
    num_leaves: int = 15
    max_depth: int = 4
    learning_rate: float = 0.03
    n_estimators: int = 300
    min_child_samples: int = 80
    subsample: float = 0.80
    subsample_freq: int = 5
    feature_fraction: float = 0.70
    reg_alpha: float = 0.05
    reg_lambda: float = 10.0
    early_stopping_rounds: int = 30
    stacking_base_models: list[str] = field(
        default_factory=lambda: ["logistic_regression", "random_forest", "lightgbm"]
    )
    stacking_meta_model: str = "logistic_regression"
    stacking_meta_fraction: float = 0.20
    stacking_passthrough: bool = False
    random_forest_n_estimators: int = 300
    random_forest_max_depth: int = 6
    random_forest_min_samples_leaf: int = 80


@dataclass
class BacktestConfig:
    """Trading simulation settings."""

    initial_capital: float = 10_000.0
    leverage: int = 10
    spread_ticks: float = 35.0
    slippage_ticks: float = 5.0
    commission_per_lot: float = 10.0
    atr_stop_multiplier: float = 2.0
    atr_tp_multiplier: float = 2.0
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

    run_data_pipeline: bool = True
    run_feature_engineering: bool = True
    run_label_generation: bool = True
    run_model_training: bool = True
    run_backtest: bool = True
    run_reporting: bool = True
    cache_invalidation: str = "path"
    force_rerun: bool = False
    random_seed: int = 2024
    n_jobs: int = -1
    session_timestamp: str = ""


@dataclass
class PathsConfig:
    """Default artifact paths."""

    data_raw: str = "data/raw/XAUUSD"
    data_processed: str = "data/processed"
    ohlcv: str = "data/processed/ohlcv.parquet"
    features: str = "data/processed/features.parquet"
    labels: str = "data/processed/labels.parquet"
    train_data: str = "data/processed/train.parquet"
    val_data: str = "data/processed/val.parquet"
    test_data: str = "data/processed/test.parquet"
    model: str = "models/lightgbm_model.pkl"
    predictions: str = "data/predictions/final_predictions.parquet"
    backtest_results: str = "results/backtest_results.json"
    report: str = "results/thesis_report.md"
    data_quality_json: str = "data/processed/data_quality.json"
    session_dir: str = ""


@dataclass
class Config:
    """Runtime configuration grouped by TOML section."""

    data: DataConfig = field(default_factory=DataConfig)
    data_range: DataRangeConfig = field(default_factory=DataRangeConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    labels: LabelsConfig = field(default_factory=LabelsConfig)
    model: LGBMConfig = field(default_factory=LGBMConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


_SECTION_MAP: dict[str, type] = {
    "data": DataConfig,
    "data_range": DataRangeConfig,
    "validation": ValidationConfig,
    "features": FeaturesConfig,
    "labels": LabelsConfig,
    "model": LGBMConfig,
    "backtest": BacktestConfig,
    "workflow": WorkflowConfig,
    "paths": PathsConfig,
}


def _timeframe_to_minutes(timeframe: str) -> int:
    """Return minutes per bar for strings like 15M, 1H, 1D, or 1W."""
    match = re.fullmatch(r"\s*(\d+)\s*([mhdwMHDW])\s*", timeframe)
    if not match:
        raise ValueError(
            "Invalid timeframe format: "
            f"{timeframe!r}. Expected forms like 15M, 1H, 4H, 1D, 1W."
        )

    qty = int(match.group(1))
    unit = match.group(2).upper()
    return qty * {"M": 1, "H": 60, "D": 1440, "W": 10080}[unit]


def _scale_bars_by_timeframe(
    base_bars: int,
    base_timeframe: str,
    target_timeframe: str,
) -> int:
    """Scale a bar count while preserving elapsed time."""
    base_minutes = _timeframe_to_minutes(base_timeframe)
    target_minutes = _timeframe_to_minutes(target_timeframe)
    return max(1, int(round(base_bars * (base_minutes / target_minutes))))


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

    setattr(cfg, section, cls(**_section_kwargs(section, cls, section_data)))


def load_config(config_path: str | Path = "config.toml") -> Config:
    """Load a TOML config and fill omitted values from dataclass defaults."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "rb") as f:
        raw: dict[str, Any] = tomllib.load(f)

    legacy_mt = raw.pop("multi_timeframe", None)
    if legacy_mt is not None:
        features = raw.setdefault("features", {})
        features.setdefault("multi_timeframe", legacy_mt)

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
