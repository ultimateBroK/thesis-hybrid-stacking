"""Session artifact loading for charts and dashboard."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from thesis.shared.config import Config

logger = logging.getLogger("thesis.visualization")


def _is_artifact_file(path: Path) -> bool:
    return path not in {Path(""), Path(".")} and path.is_file()


def _safe_read(path: Path, reader: callable, *errors: type) -> Any | None:
    if not _is_artifact_file(path):
        return None
    try:
        return reader(path)
    except errors as exc:
        logger.warning("Dashboard artifact unreadable: %s (%s)", path, exc)
        return None


def _read_parquet(path: Path) -> pl.DataFrame | None:
    return _safe_read(path, pl.read_parquet, FileNotFoundError, OSError)


def _read_csv(path: Path) -> pl.DataFrame | None:
    return _safe_read(path, pl.read_csv, FileNotFoundError, OSError)


def _read_json(path: Path) -> dict | None:
    return _safe_read(
        path,
        lambda p: json.loads(p.read_text()),
        FileNotFoundError,
        OSError,
        json.JSONDecodeError,
    )


def _read_predictions(config: Config) -> pl.DataFrame | None:
    path = (
        Path(config.paths.session_dir) / "predictions" / "final_predictions.csv"
        if config.paths.session_dir
        else Path(config.paths.predictions)
    )
    return _read_csv(path)


def _read_backtest_results(config: Config) -> dict[str, Any]:
    path = (
        Path(config.paths.session_dir) / "backtest" / "backtest_results.json"
        if config.paths.session_dir
        else Path(config.paths.backtest_results)
    )
    bt = _read_json(path)
    if bt is None:
        return {"backtest_results": None, "trades": [], "metrics": {}}
    return {
        "backtest_results": bt,
        "trades": bt.get("trades", []),
        "metrics": bt.get("metrics", {}),
    }


def _read_feature_importance(config: Config) -> dict[str, float]:
    path = (
        Path(config.paths.session_dir) / "reports" / "feature_importance.json"
        if config.paths.session_dir
        else Path("results/feature_importance.json")
    )
    data = _read_json(path)
    return data if data is not None else {}


def _read_model_comparison(config: Config) -> list[dict]:
    path = Path(config.paths.session_dir) / "reports" / "model_comparison.csv"
    if config.paths.session_dir and path.exists():
        comparison = _read_csv(path)
        return comparison.to_dicts() if comparison is not None else []
    return []


def load_dashboard_artifacts(config: Config) -> dict[str, Any]:
    """Load session artifacts for chart builders and dashboard."""
    data: dict[str, Any] = {"session_dir": config.paths.session_dir}

    data["ohlcv"] = _read_parquet(Path(config.paths.ohlcv))
    data["features"] = _read_parquet(Path(config.paths.features))
    data["test"] = _read_parquet(Path(config.paths.test_data))
    data["labels"] = _read_parquet(Path(config.paths.labels))
    data["predictions"] = _read_predictions(config)
    data.update(_read_backtest_results(config))
    data["feature_importance"] = _read_feature_importance(config)
    data["model_comparison"] = _read_model_comparison(config)

    logger.info("Session data loaded from %s", config.paths.session_dir or "default")
    return data


__all__ = ["load_dashboard_artifacts"]
