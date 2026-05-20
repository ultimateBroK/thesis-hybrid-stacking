"""Artifact writers for Stage 3 model experiments."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from thesis.models.estimators import CLASS_ORDER
from thesis.shared.config import Config

logger = logging.getLogger("thesis")


def proba_columns(
    proba: np.ndarray,
    class_order: list[int] | np.ndarray = CLASS_ORDER,
) -> dict[str, np.ndarray]:
    """Probability columns in canonical report format.

    Maps class labels to named columns: Short (-1), Hold (0), Long (1).
    Dynamic — works for binary [-1,1] or 3-class [-1,0,1].
    """
    suffix = {-1: "minus1", 0: "0", 1: "1"}
    index_map = {int(c): i for i, c in enumerate(class_order)}
    return {
        f"pred_proba_class_{suffix[int(cls)]}": proba[:, index_map[int(cls)]]
        if int(cls) in index_map
        else np.zeros(len(proba))
        for cls in class_order
    }


def save_oof_predictions(df: pl.DataFrame, config: Config) -> None:
    """Write canonical OOF predictions CSV."""
    path = Path(config.paths.predictions)
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.is_empty():
        raise ValueError(f"Predictions are empty: {path}")
    if df["timestamp"].n_unique() < len(df):
        raise ValueError(f"OOF predictions contain duplicate timestamps: {path}")
    df.sort("timestamp").write_csv(path)


def save_model_artifact(model: Any, config: Config) -> None:
    """Persist final model bundle."""
    import joblib

    path = Path(config.paths.model)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def save_model_comparison(metrics: dict[str, dict[str, Any]], config: Config) -> None:
    """Write model comparison JSON used by report stage."""
    if not config.paths.session_dir:
        return
    path = Path(config.paths.session_dir) / "reports" / "model_comparison.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))


def save_feature_importance(
    model: Any,
    feature_cols: list[str],
    config: Config,
) -> None:
    """Write LightGBM feature importance if available."""
    importance = getattr(model, "feature_importances_", None)
    if importance is None or not config.paths.session_dir:
        return
    pairs = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)
    path = Path(config.paths.session_dir) / "reports" / "feature_importance.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: float(v) for k, v in pairs}, indent=2))


def save_training_history(payload: dict[str, Any], config: Config) -> None:
    """Write concise training history."""
    if not config.paths.session_dir:
        return
    path = Path(config.paths.session_dir) / "models" / "training_history.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def save_walk_forward_history(payload: dict[str, Any], config: Config) -> None:
    """Write walk-forward window history."""
    if not config.paths.session_dir:
        return
    path = Path(config.paths.session_dir) / "reports" / "walk_forward_history.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def save_model_experiment(experiment: Any, config: Config) -> None:
    """Persist all Stage 3 experiment artifacts."""
    save_oof_predictions(experiment.oof_predictions, config)
    save_model_artifact(experiment.final_model, config)
    save_model_comparison(experiment.model_comparison, config)
    save_feature_importance(
        experiment.final_lightgbm_model,
        experiment.feature_cols,
        config,
    )
    save_training_history(experiment.training_history, config)
    save_walk_forward_history(experiment.walk_forward_history, config)
    logger.info(
        "Model experiment saved: %d predictions, %d models",
        len(experiment.oof_predictions),
        len(experiment.model_comparison),
    )
