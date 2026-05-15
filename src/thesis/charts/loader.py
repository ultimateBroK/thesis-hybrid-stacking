"""Session artifact loading."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from thesis.shared.config import Config

logger = logging.getLogger("thesis.charts")


def load_session_data(config: Config) -> dict[str, Any]:
    """Load session artifacts for chart builders."""
    data: dict[str, Any] = {}

    data["session_dir"] = config.paths.session_dir

    ohlcv_path = Path(config.paths.ohlcv)
    data["ohlcv"] = pl.read_parquet(ohlcv_path) if ohlcv_path.exists() else None

    features_path = Path(config.paths.features)
    data["features"] = (
        pl.read_parquet(features_path) if features_path.exists() else None
    )

    test_path = Path(config.paths.test_data)
    data["test"] = pl.read_parquet(test_path) if test_path.exists() else None

    labels_path = Path(config.paths.labels)
    data["labels"] = pl.read_parquet(labels_path) if labels_path.exists() else None

    preds_path = (
        Path(config.paths.session_dir) / "predictions" / "final_predictions.csv"
        if config.paths.session_dir
        else Path(config.paths.predictions)
    )
    data["predictions"] = pl.read_csv(preds_path) if preds_path.exists() else None

    bt_path = (
        Path(config.paths.session_dir) / "backtest" / "backtest_results.json"
        if config.paths.session_dir
        else Path(config.paths.backtest_results)
    )
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

    fi_path = (
        Path(config.paths.session_dir) / "reports" / "feature_importance.json"
        if config.paths.session_dir
        else Path("results/feature_importance.json")
    )
    if fi_path.exists():
        with open(fi_path) as f:
            data["feature_importance"] = json.load(f)
    else:
        data["feature_importance"] = {}

    logger.info("Session data loaded from %s", config.paths.session_dir or "default")
    return data


__all__ = ["load_session_data"]
