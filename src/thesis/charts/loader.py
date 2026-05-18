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


def _read_parquet(path: Path) -> pl.DataFrame | None:
    """Read parquet if available; stale paths should not crash dashboard."""
    if not _is_artifact_file(path):
        return None
    try:
        return pl.read_parquet(path)
    except (FileNotFoundError, OSError) as exc:
        logger.warning("Dashboard artifact unreadable: %s (%s)", path, exc)
        return None


def _read_csv(path: Path) -> pl.DataFrame | None:
    """Read CSV if available; stale paths should not crash dashboard."""
    if not _is_artifact_file(path):
        return None
    try:
        return pl.read_csv(path)
    except (FileNotFoundError, OSError) as exc:
        logger.warning("Dashboard artifact unreadable: %s (%s)", path, exc)
        return None


def _is_artifact_file(path: Path) -> bool:
    """True only for concrete files; ignore empty/default directory paths."""
    return path not in {Path(""), Path(".")} and path.is_file()


def load_session_data(config: Config) -> dict[str, Any]:
    """Load session artifacts for chart builders."""
    data: dict[str, Any] = {}

    data["session_dir"] = config.paths.session_dir

    ohlcv_path = Path(config.paths.ohlcv)
    data["ohlcv"] = _read_parquet(ohlcv_path)

    features_path = Path(config.paths.features)
    data["features"] = _read_parquet(features_path)

    test_path = Path(config.paths.test_data)
    data["test"] = _read_parquet(test_path)

    labels_path = Path(config.paths.labels)
    data["labels"] = _read_parquet(labels_path)

    preds_path = (
        Path(config.paths.session_dir) / "predictions" / "final_predictions.csv"
        if config.paths.session_dir
        else Path(config.paths.predictions)
    )
    data["predictions"] = _read_csv(preds_path)

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

    comparison_csv = Path(config.paths.session_dir) / "reports" / "model_comparison.csv"
    if config.paths.session_dir and comparison_csv.exists():
        comparison = _read_csv(comparison_csv)
        data["model_comparison"] = (
            comparison.to_dicts() if comparison is not None else []
        )
    else:
        data["model_comparison"] = []

    logger.info("Session data loaded from %s", config.paths.session_dir or "default")
    return data


__all__ = ["load_session_data"]
