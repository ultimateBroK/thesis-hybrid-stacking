"""Shared data loading and constants for interactive ECharts charts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
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
    """
    Compute the dataframe column names that should be treated as features by excluding metadata and OHLCV-related columns.

    Parameters:
        df (pl.DataFrame): Input dataframe whose columns will be filtered.

    Returns:
        list[str]: Column names from `df` that are not in `EXCLUDED_FEATURE_COLS`, preserving the dataframe's column order.
    """
    return [c for c in df.columns if c not in EXCLUDED_FEATURE_COLS]


# --- Data Loading -------------------------------------------------------------


def load_session_data(config: "Config") -> dict[str, Any]:
    """
    Load all parquet and JSON artifacts required for charting.

    Parameters:
        config (Config): Runtime configuration containing paths used to locate session artifacts.

    Returns:
        data (dict[str, Any]): Mapping with keys:
            session_dir: configured session directory (or falsy value),
            ohlcv: DataFrame or `None` if the OHLCV parquet is missing,
            features: DataFrame or `None` if the features parquet is missing,
            test: DataFrame or `None` if the test parquet is missing (used for manual backtest),
            labels: DataFrame or `None` if the labels parquet is missing,
            predictions: DataFrame or `None` if the predictions parquet is missing,
            backtest_results: parsed JSON object or `None` if missing,
            trades: list of trades (defaults to `[]` when backtest results are missing),
            metrics: dict of metrics (defaults to `{}` when backtest results are missing),
            feature_importance: dict parsed from JSON (defaults to `{}` when missing).
    """
    data: dict[str, Any] = {}

    # Session dir (for download paths)
    data["session_dir"] = config.paths.session_dir

    # OHLCV
    ohlcv_path = Path(config.paths.ohlcv)
    data["ohlcv"] = pl.read_parquet(ohlcv_path) if ohlcv_path.exists() else None

    # Features
    features_path = Path(config.paths.features)
    data["features"] = (
        pl.read_parquet(features_path) if features_path.exists() else None
    )

    # Test data (for manual backtesting)
    test_path = Path(config.paths.test_data)
    data["test"] = pl.read_parquet(test_path) if test_path.exists() else None

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

    # SHAP values (JSON)
    if config.paths.session_dir:
        shap_path = Path(config.paths.session_dir) / "reports" / "shap_values.json"
    else:
        shap_path = Path("results/shap_values.json")
    if shap_path.exists():
        with open(shap_path) as f:
            data["shap_values"] = json.load(f)
    else:
        data["shap_values"] = None

    logger.info("Session data loaded from %s", config.paths.session_dir or "default")
    return data
