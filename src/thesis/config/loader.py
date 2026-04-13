"""Configuration management for thesis pipeline.

Supports TOML config with environment variable overrides using pattern:
THESIS_<SECTION>__<KEY>

Example:
    THESIS_DATA__TIMEFRAME=30m
    THESIS_MODELS__LSTM__LEARNING_RATE=0.0005
"""

import os
import re
import tomllib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


def _parse_env_value(value: str) -> Any:
    """Parse an environment variable string into a Python value.

    Args:
        value: Raw environment variable value.

    Returns:
        Parsed value as ``int``, ``float``, ``bool``, ``list``, or ``str``.
    """
    value = value.strip()

    # Try int
    try:
        return int(value)
    except ValueError:
        pass

    # Try float
    try:
        return float(value)
    except ValueError:
        pass

    # Try bool
    if value.lower() in ("true", "yes", "1"):
        return True
    if value.lower() in ("false", "no", "0"):
        return False

    # Try list (comma-separated)
    if "," in value:
        parts = [p.strip() for p in value.split(",")]
        return [_parse_env_value(p) if not p.startswith("[") else p for p in parts]

    return value


def _apply_env_overrides(
    config: dict[str, Any], prefix: str = "THESIS"
) -> dict[str, Any]:
    """Apply environment variable overrides to a nested config mapping.

    Supports keys shaped as ``<PREFIX>_<SECTION>__<OPTION>`` and nested
    overrides such as ``THESIS_MODELS__LSTM__LEARNING_RATE``.

    Args:
        config: Mutable configuration mapping loaded from TOML.
        prefix: Environment variable prefix.

    Returns:
        The same mapping with applicable overrides applied.
    """
    pattern = re.compile(rf"^{prefix}_(.+)__(.+)$")

    for key, value in os.environ.items():
        match = pattern.match(key)
        if match:
            section = match.group(1).lower()
            option = match.group(2).lower()

            # Handle nested keys (e.g., THESIS_MODELS__LSTM__LEARNING_RATE)
            if "__" in option:
                parts = option.split("__")
                target = config
                for part in parts[:-1]:
                    if part not in target:
                        target[part] = {}
                    target = target[part]
                target[parts[-1]] = _parse_env_value(value)
            else:
                if section in config:
                    config[section][option] = _parse_env_value(value)

    return config


@dataclass
class DataConfig:
    """Data loading and processing configuration."""

    raw_data_path: str = "data/raw/XAUUSD"
    ohlcv_path: str = "data/processed/ohlcv.parquet"
    timeframe: str = "1H"
    timeframe_minutes: int = 60
    market_tz: str = "America/New_York"
    day_roll_hour: int = 17
    start_date: str = "2018-01-01"
    end_date: str = "2026-03-31"
    tick_size: float = 0.01
    contract_size: int = 100

    # Download configuration (Dukascopy)
    symbol: str = "XAUUSD"
    asset_class: str = "fx"
    download_concurrency: int = 8
    download_force: bool = False
    download_skip_current_month: bool = False


@dataclass
class SplittingConfig:
    """Data splitting configuration with market regime-based scheme."""

    train_start: str = "2018-01-01"
    train_end: str = "2022-12-31 23:59:59"
    val_start: str = "2023-01-01"
    val_end: str = "2023-12-31 23:59:59"
    test_start: str = "2024-01-01"
    test_end: str = "2026-03-31 23:59:59"
    purge_bars: int = 15
    embargo_bars: int = 10
    # Walk-Forward Cross-Validation settings
    use_walk_forward_cv: bool = True
    walk_forward_window_type: str = "sliding"  # "sliding" or "expanding"
    walk_forward_train_years: int = 2
    walk_forward_val_years: int = 1
    walk_forward_step_years: int = 1


@dataclass
class FeaturesConfig:
    """Feature engineering configuration."""

    features_path: str = "data/processed/features.parquet"
    feature_list_path: str = "data/processed/feature_list.json"
    use_technical: bool = True
    use_pivots: bool = True
    use_session: bool = True
    use_spread: bool = True
    ema_periods: list[int] = field(default_factory=lambda: [34, 89])
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    pivot_lookback: int = 1
    session_hours: list[list[int]] = field(
        default_factory=lambda: [[0, 8], [8, 17], [17, 21]]
    )
    lag_periods: list[int] = field(default_factory=lambda: [1, 2, 3, 5, 10])
    spread_multiplier: float = 0.5
    drop_high_corr: bool = True
    correlation_threshold: float = 0.90


@dataclass
class SessionDefinition:
    """A single market session with UTC hour range and ATR multipliers.

    Attributes:
        session: Session name (e.g. ``"dead"``, ``"overlap"``).
        start_utc: Start hour in UTC (0-23).
        end_utc: End hour in UTC (0-23). ``0`` means midnight wrap.
        tp_mult: Take-profit ATR multiplier. ``0.0`` forces Hold label.
        sl_mult: Stop-loss ATR multiplier. ``0.0`` forces Hold label.
    """

    session: str = ""
    start_utc: int = 0
    end_utc: int = 0
    tp_mult: float = 1.5
    sl_mult: float = 1.5


@dataclass
class SessionATRConfig:
    """Session-aware ATR multiplier configuration.

    When ``enabled`` is ``True``, the fixed ``atr_multiplier_tp`` /
    ``atr_multiplier_sl`` values are overridden per market session.
    """

    enabled: bool = False
    summer: list[SessionDefinition] = field(default_factory=list)
    winter: list[SessionDefinition] = field(default_factory=list)


@dataclass
class LabelsConfig:
    """Triple-Barrier label configuration."""

    labels_path: str = "data/processed/labels.parquet"
    atr_multiplier_tp: float = 1.5
    atr_multiplier_sl: float = 1.5
    use_fixed_pips: bool = False
    tp_pips: int = 20
    sl_pips: int = 10
    horizon_bars: int = 20
    num_classes: int = 3
    class_labels: dict[str, str] = field(
        default_factory=lambda: {"-1": "Short", "0": "Hold", "1": "Long"}
    )
    min_atr: float = 0.0001
    filter_dead_hours_from_train: bool = True
    session_atr: SessionATRConfig = field(default_factory=SessionATRConfig)


@dataclass
class TreeModelConfig:
    """LightGBM model configuration."""

    model_path: str = "models/lightgbm_model.pkl"
    predictions_path: str = "data/predictions/lightgbm_oof.parquet"
    oof_predictions_path: str = "data/predictions/lightgbm_oof_train.parquet"
    prediction_type: str = "probabilities"
    use_class_weights: bool = True
    use_downsampling: bool = False
    downsampling_ratio: float = 0.5
    use_optuna: bool = True
    optuna_trials: int = 30  # Set to 100 for final thesis run
    optuna_timeout: int = 3600
    cv_folds: int = 5
    cv_purge: int = 10
    cv_embargo: int = 5

    num_leaves: int = 31
    max_depth: int = -1
    learning_rate: float = 0.05
    n_estimators: int = 500
    min_child_samples: int = 20
    subsample: float = 0.8
    subsample_freq: int = 5
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0
    early_stopping_rounds: int = 50
    compute_importance: bool = True
    importance_type: str = "gain"


@dataclass
class LSTMModelConfig:
    """LSTM/GRU model configuration."""

    model_type: str = "gru"  # "gru" or "lstm"
    model_path: str = "models/lstm_model.pt"
    predictions_path: str = "data/predictions/lstm_oof.parquet"
    oof_predictions_path: str = "data/predictions/lstm_oof_train.parquet"
    sequences_path: str = "data/processed/lstm_sequences.npz"
    sequence_length: int = 120
    step_size: int = 1
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = False
    batch_size: int = 128
    epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    patience: int = 10
    min_delta: float = 0.001
    device: str = "auto"
    num_workers: int = 4
    save_best: bool = True


@dataclass
class StackingConfig:
    """Hybrid Stacking meta-learner configuration."""

    model_path: str = "models/stacking_meta_learner.pkl"
    meta_predictions_path: str = "data/predictions/stacking_predictions.parquet"
    meta_learner: str = "logistic_regression"
    n_folds: int = 5
    stacking_purge: int = 10
    stacking_embargo: int = 5
    # Logistic Regression params
    C: float = 1.0
    # ElasticNet params
    alpha: float = 1.0
    l1_ratio: float = 0.5
    # LightGBM meta-learner params (when meta_learner="lightgbm")
    n_estimators: int = 100
    learning_rate: float = 0.1
    # Calibration
    calibrate_probabilities: bool = True
    calibration_method: str = "isotonic"


@dataclass
class BacktestConfig:
    """CFD backtesting configuration."""

    backtest_results_path: str = "results/backtest_results.json"
    spread_pips: float = 2.0
    slippage_pips: float = 1.0
    commission_per_lot: float = 0.0
    swap_long: float = -10.0
    swap_short: float = 5.0
    initial_capital: float = 100000.0
    currency: str = "USD"
    leverage: int = 50  # Reduced from 100 for more realistic simulation
    max_positions: int = 1
    risk_per_trade: float = 0.01
    position_size_method: str = "fixed_fractional"
    margin_call_level: float = 0.5
    stop_out_level: float = 0.2

    # NEW: Additional risk controls
    max_hold_bars: int = 100  # Maximum bars to hold a position
    use_trailing_stop: bool = True  # Enable trailing stop loss
    trailing_stop_atr_multiplier: float = 1.0  # Trailing stop distance
    max_daily_loss_pct: float = 0.05  # Stop trading after 5% daily loss
    max_consecutive_losses: int = 5  # Pause after 5 consecutive losses

    trade_sunday: bool = False
    trade_friday_close: bool = False
    friday_close_hour: int = 21
    metrics: list[str] = field(
        default_factory=lambda: [
            "sharpe",
            "sortino",
            "max_drawdown",
            "calmar",
            "win_rate",
            "profit_factor",
            "avg_trade",
            "total_return",
        ]
    )
    single_oos_run: bool = True


@dataclass
class PlotlyConfig:
    """Plotly interactive visualization configuration."""

    enabled: bool = True
    equity_curve_path: str = "reports/interactive_equity.html"
    confidence_path: str = "reports/interactive_confidence.html"
    trades_path: str = "reports/interactive_trades.html"
    include_annotations: bool = True


@dataclass
class ReportingConfig:
    """Reporting and visualization configuration."""

    report_path: str = "results/thesis_report.md"
    report_json_path: str = "results/thesis_report.json"
    shap_summary_path: str = "results/shap_summary.png"
    feature_importance_path: str = "results/feature_importance.png"
    plot_predictions: bool = True
    plot_backtest_equity: bool = True
    plot_drawdown: bool = True
    plot_trade_distribution: bool = True
    shap_samples: int = 1000
    shap_max_display: int = 20
    export_csv: bool = True
    export_excel: bool = False
    plotly: PlotlyConfig = field(default_factory=PlotlyConfig)


@dataclass
class WorkflowConfig:
    """Pipeline execution configuration."""

    run_data_pipeline: bool = True
    run_feature_engineering: bool = True
    run_label_generation: bool = True
    run_data_splitting: bool = True
    run_lightgbm: bool = True
    run_lstm: bool = True
    run_stacking: bool = True
    run_backtest: bool = True
    run_reporting: bool = True
    force_rerun: bool = False
    n_jobs: int = -1
    random_seed: int = 42


@dataclass
class PathsConfig:
    """Path configuration for all artifacts."""

    # Session configuration
    use_sessions: bool = True  # Always True - sessions are mandatory
    session_id: str = ""  # Auto-generated if empty
    session_path: str = ""  # Computed path to session folder

    # Global paths (shared across sessions)
    data_raw: str = "data/raw/XAUUSD"
    data_processed: str = "data/processed"

    # Artifact files (prefixed with session_path)
    ohlcv: str = "data/processed/ohlcv.parquet"
    features: str = "data/processed/features.parquet"
    labels: str = "data/processed/labels.parquet"
    train_data: str = "data/processed/train.parquet"
    val_data: str = "data/processed/val.parquet"
    test_data: str = "data/processed/test.parquet"
    lgbm_model: str = "models/lightgbm_model.pkl"
    lstm_model: str = "models/lstm_model.pt"
    stacking_model: str = "models/stacking_meta_learner.pkl"
    lgbm_oof: str = "data/predictions/lightgbm_oof.parquet"
    lstm_oof: str = "data/predictions/lstm_oof.parquet"
    final_predictions: str = "data/predictions/final_predictions.parquet"
    backtest_results: str = "results/backtest_results.json"
    final_report: str = "results/thesis_report.md"

    def get_resolved_path(self, path_attr: str) -> str:
        """Get resolved path considering session configuration."""
        path_value = getattr(self, path_attr)

        # Don't modify global data paths (features, labels, etc.)
        if path_attr in [
            "ohlcv",
            "features",
            "labels",
            "train_data",
            "val_data",
            "test_data",
            "data_raw",
            "data_processed",
        ]:
            return path_value

        # Prepend session_path to artifact paths
        if self.session_path:
            return f"{self.session_path}/{path_value}"

        return path_value


@dataclass
class Config:
    """Main configuration class aggregating all sub-configs."""

    data: DataConfig = field(default_factory=DataConfig)
    splitting: SplittingConfig = field(default_factory=SplittingConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    labels: LabelsConfig = field(default_factory=LabelsConfig)
    models: Any = field(default_factory=dict)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    def __post_init__(self):
        """Initialize nested model configs after creation."""
        if not isinstance(self.models, dict) or not self.models:
            self.models = {
                "tree": TreeModelConfig(),
                "lstm": LSTMModelConfig(),
                "stacking": StackingConfig(),
            }


def load_config(config_path: str | Path = "config.toml") -> Config:
    """Load configuration from TOML file with environment overrides.

    Args:
        config_path: Path to TOML configuration file.

    Returns:
        Config object with all settings loaded.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config is invalid.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load TOML
    with open(config_path, "rb") as f:
        raw_config = tomllib.load(f)

    # Apply environment overrides
    raw_config = _apply_env_overrides(raw_config)

    # Build Config object
    config = Config()

    # Populate from TOML
    if "data" in raw_config:
        config.data = DataConfig(**raw_config["data"])

    if "splitting" in raw_config:
        config.splitting = SplittingConfig(**raw_config["splitting"])

    if "features" in raw_config:
        config.features = FeaturesConfig(**raw_config["features"])

    if "labels" in raw_config:
        labels_raw = dict(raw_config["labels"])
        # Extract session_atr separately — nested structures need manual parsing
        session_atr_raw = labels_raw.pop("session_atr", None)
        config.labels = LabelsConfig(**labels_raw)

        if session_atr_raw and isinstance(session_atr_raw, dict):
            summer_defs = [
                SessionDefinition(**s) for s in session_atr_raw.get("summer", [])
            ]
            winter_defs = [
                SessionDefinition(**s) for s in session_atr_raw.get("winter", [])
            ]
            config.labels.session_atr = SessionATRConfig(
                enabled=session_atr_raw.get("enabled", False),
                summer=summer_defs,
                winter=winter_defs,
            )

    if "models" in raw_config:
        models_config = raw_config["models"]
        config.models = {
            "tree": TreeModelConfig(**models_config.get("tree", {})),
            "lstm": LSTMModelConfig(**models_config.get("lstm", {})),
            "stacking": StackingConfig(**models_config.get("stacking", {})),
        }

    if "backtest" in raw_config:
        config.backtest = BacktestConfig(**raw_config["backtest"].get("cfd", {}))

    if "reporting" in raw_config:
        reporting_raw = dict(raw_config["reporting"])
        plotly_raw = reporting_raw.pop("plotly", None)
        config.reporting = ReportingConfig(**reporting_raw)
        if plotly_raw:
            config.reporting.plotly = PlotlyConfig(**plotly_raw)

    if "workflow" in raw_config:
        config.workflow = WorkflowConfig(**raw_config["workflow"])

    if "paths" in raw_config:
        config.paths = PathsConfig(**raw_config["paths"])

    # Ensure directories exist
    _ensure_directories(config)

    return config


def _ensure_directories(config: Config) -> None:
    """Create required top-level data directories.

    Args:
        config: Loaded application configuration.
    """
    dirs_to_create = [
        config.paths.data_raw,
        config.paths.data_processed,
    ]

    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def generate_session_path(
    symbol: str, timeframe: str, timestamp: datetime | None = None
) -> Path:
    """Generate session folder path following naming convention.

    Format: results/{SYMBOL}_{TIMEFRAME}_{YYYYMMDD}_{HHMMSS}/

    Args:
        symbol: Trading pair symbol (e.g., "XAUUSD")
        timeframe: Timeframe string (e.g., "H1", "1H")
        timestamp: Optional timestamp, defaults to current time

    Returns:
        Path to session folder
    """
    if timestamp is None:
        timestamp = datetime.now()

    # Normalize symbol (uppercase, no special chars)
    symbol_clean = symbol.upper().replace("/", "").replace("-", "")

    # Normalize timeframe (uppercase)
    timeframe_clean = timeframe.upper()

    # Format: SYMBOL_TIMEFRAME_YYYYMMDD_HHMMSS
    session_name = f"{symbol_clean}_{timeframe_clean}_{timestamp:%Y%m%d_%H%M%S}"

    return Path("results") / session_name


def initialize_session(config: Config, custom_session_id: str | None = None) -> Path:
    """Initialize a new session with proper folder structure.

    Args:
        config: Configuration object
        custom_session_id: Optional custom session ID, auto-generated if None

    Returns:
        Path to created session folder
    """
    # Generate or use custom session ID
    if custom_session_id:
        session_path = Path("results") / custom_session_id
    else:
        # Generate from symbol and timeframe
        symbol = config.data.symbol
        timeframe = config.data.timeframe
        session_path = generate_session_path(symbol, timeframe)

    # Store in config
    config.paths.session_id = session_path.name
    config.paths.session_path = str(session_path)

    # Create session subdirectories
    subdirs = ["config", "models", "predictions", "reports", "backtest", "logs"]
    for subdir in subdirs:
        (session_path / subdir).mkdir(parents=True, exist_ok=True)

    return session_path


def get_config() -> Config:
    """Load and return configuration using default path.

    Returns:
        Fully initialized configuration object.
    """
    return load_config()
