"""Shared data loading and constants for interactive ECharts charts."""

import json
import logging
from pathlib import Path

import polars as pl

from thesis.config import Config

logger = logging.getLogger("thesis.charts")

# --- Constants ----------------------------------------------------------------

COLORS: dict[str, str] = {
    "primary": "#2563EB",
    "secondary": "#7C3AED",
    "success": "#059669",
    "danger": "#DC2626",
    "warning": "#D97706",
    "gray": "#6B7280",
    "long": "#059669",
    "short": "#DC2626",
    "flat": "#6B7280",
}

EXCLUDED_FEATURE_COLS: frozenset[str] = frozenset(
    {
        "timestamp",
        "label",
        "tp_price",
        "sl_price",
        "touched_bar",
        "open_right",
        "high_right",
        "low_right",
        "close_right",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "avg_spread",
        "tick_count",
        "dead_hour",
        "log_returns",
    }
)


def _get_feature_cols(df: pl.DataFrame) -> list[str]:
    """Return feature column names (excludes metadata/OHLCV columns)."""
    return [c for c in df.columns if c not in EXCLUDED_FEATURE_COLS]


# --- Data Loading -------------------------------------------------------------


def load_session_data(config: Config) -> dict[str, pl.DataFrame | dict | None]:
    """Load all parquet/json artifacts needed for charts.

    Returns dict with keys: ohlcv, features, labels, predictions,
    backtest_results, feature_importance, metrics, trades.
    Missing files return None values.
    """
    data: dict[str, pl.DataFrame | dict | None] = {}

    # OHLCV
    ohlcv_path = Path(config.paths.ohlcv)
    data["ohlcv"] = pl.read_parquet(ohlcv_path) if ohlcv_path.exists() else None

    # Features
    features_path = Path(config.paths.features)
    data["features"] = (
        pl.read_parquet(features_path) if features_path.exists() else None
    )

    # Labels
    labels_path = Path(config.paths.labels)
    data["labels"] = pl.read_parquet(labels_path) if labels_path.exists() else None

    # Predictions
    if config.paths.session_dir:
        preds_path = (
            Path(config.paths.session_dir) / "predictions" / "final_predictions.parquet"
        )
    else:
        preds_path = Path(config.paths.predictions)
    data["predictions"] = pl.read_parquet(preds_path) if preds_path.exists() else None

    # Backtest results (JSON)
    if config.paths.session_dir:
        bt_path = Path(config.paths.session_dir) / "backtest" / "backtest_results.json"
    else:
        bt_path = Path(config.paths.backtest_results)
    if bt_path.exists():
        with open(bt_path) as f:
            bt = json.load(f)
        data["backtest_results"] = bt
        data["trades"] = bt.get("trades", [])
        data["metrics"] = bt.get("metrics", {})
    else:
        data["backtest_results"] = None
        data["trades"] = []
        data["metrics"] = {}

    # Feature importance (JSON)
    if config.paths.session_dir:
        fi_path = Path(config.paths.session_dir) / "reports" / "feature_importance.json"
    else:
        fi_path = Path("results/feature_importance.json")
    if fi_path.exists():
        with open(fi_path) as f:
            data["feature_importance"] = json.load(f)
    else:
        data["feature_importance"] = {}

    logger.info("Session data loaded from %s", config.paths.session_dir or "default")
    return data
