"""Stage 3 model experiment orchestration."""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from thesis.models.artifacts import save_model_experiment
from thesis.models.experiment import run_model_experiment
from thesis.models.validation import build_walk_forward_windows
from thesis.shared.config import Config

logger = logging.getLogger("thesis")

__all__ = [
    "build_walk_forward_windows",
    "choose_model_features",
    "load_model_dataset",
    "train_walk_forward",
]


def load_model_dataset(config: Config) -> pl.DataFrame:
    """Load Stage 3 classification dataset."""
    path = Path(config.paths.ml_dataset)
    if not path.exists():
        raise FileNotFoundError(f"ML dataset not found: {path}")
    df = pl.read_parquet(path)
    if "label" not in df.columns:
        raise ValueError(f"ML dataset missing label column: {path}")
    return df


def choose_model_features(df: pl.DataFrame, config: Config) -> list[str]:
    """Choose fixed Stage 2 feature columns for Stage 3."""
    features = [c for c in config.features.static_feature_cols if c in df.columns]
    if not features:
        raise ValueError("No configured model features found in ML dataset")
    return features


def train_walk_forward(config: Config) -> None:
    """Run fixed-feature walk-forward model experiment."""
    if config.model.objective != "multiclass":
        raise ValueError(
            "Stage 3 is classification-only; set model.objective='multiclass'"
        )

    dataset = load_model_dataset(config)
    feature_cols = choose_model_features(dataset, config)
    windows = build_walk_forward_windows(dataset, config)
    if not windows:
        raise RuntimeError("No valid walk-forward windows")

    logger.info(
        "Stage 3: %d windows, %d fixed features",
        len(windows),
        len(feature_cols),
    )
    experiment = run_model_experiment(dataset, feature_cols, windows, config)
    save_model_experiment(experiment, config)
