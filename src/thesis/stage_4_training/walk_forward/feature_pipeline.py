"""Feature pipeline helpers for walk-forward training."""

from __future__ import annotations

import logging

from feature_engine.selection import (
    DropCorrelatedFeatures,
    DropDuplicateFeatures,
)
import numpy as np
import polars as pl
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from thesis.shared.config import Config

logger = logging.getLogger("thesis.pipeline")


def _select_static_feature_cols(
    config: Config,
    df: pl.DataFrame,
    candidate_cols: list[str],
) -> list[str]:
    """Return static features for LightGBM, preferring the config whitelist."""
    available = [c for c in config.features.static_feature_cols if c in df.columns]
    if available:
        return available
    return [c for c in candidate_cols if c in df.columns]


def fit_static_feature_pipeline(
    config: Config,
    train_df: pl.DataFrame,
    static_cols: list[str],
    y_train: np.ndarray,
) -> tuple[Pipeline, list[str]]:
    """Fit train-only scaler/selector pipeline for static features."""
    if not static_cols:
        raise ValueError("No static feature columns available for selection")
    X_train = train_df.select(static_cols).to_pandas()
    X_train.columns = static_cols
    if X_train.empty:
        raise ValueError("Training split is empty; cannot fit static pipeline")

    k_best = min(max(5, len(static_cols) // 2), len(static_cols))
    feature_pipeline = Pipeline(
        steps=[
            ("drop_duplicate", DropDuplicateFeatures(missing_values="ignore")),
            (
                "drop_correlated",
                DropCorrelatedFeatures(
                    threshold=config.features.correlation_threshold,
                    method="pearson",
                ),
            ),
            ("scaler", RobustScaler()),
            ("select_k_best", SelectKBest(score_func=f_classif, k=k_best)),
        ]
    )
    try:
        feature_pipeline.fit(X_train, y_train)
        preselect = feature_pipeline[:-1].transform(X_train)
        preselect_cols = feature_pipeline[:-1].get_feature_names_out()
        if not isinstance(preselect, pl.DataFrame):
            preselect = pl.DataFrame(preselect, schema=preselect_cols)
        selected_mask = feature_pipeline.named_steps["select_k_best"].get_support()
        selected_cols = [
            str(col)
            for col, keep in zip(preselect_cols, selected_mask, strict=False)
            if keep
        ]
        if not selected_cols:
            selected_cols = list(preselect.columns[: min(5, preselect.width)])
        return feature_pipeline, selected_cols
    except ValueError as exc:
        logger.warning("Static feature selection fallback activated: %s", str(exc))
        fallback_cols = list(static_cols)
        fallback_pipeline = Pipeline(steps=[("scaler", RobustScaler())])
        fallback_pipeline.fit(X_train[fallback_cols], y_train)
        X_train[fallback_cols] = fallback_pipeline.transform(X_train[fallback_cols])
        return fallback_pipeline, fallback_cols
