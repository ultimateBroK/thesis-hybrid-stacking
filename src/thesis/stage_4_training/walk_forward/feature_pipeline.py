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
from thesis.shared.feature_registry import REGIME_FEATURES

logger = logging.getLogger("thesis.pipeline")


def _select_static_feature_cols(
    config: Config,
    df: pl.DataFrame,
    candidate_cols: list[str],
) -> list[str]:
    """Return static features for LightGBM, preferring the config whitelist.

    When ``enable_regime_features`` is True, dynamically adds any regime
    feature columns present in *df* (including label-prior features that
    were computed in stage 4's data preparation step).
    """
    available = [c for c in config.features.static_feature_cols if c in df.columns]
    if not available:
        available = [c for c in candidate_cols if c in df.columns]

    # Add regime features dynamically when enabled and present in the dataframe
    if getattr(config.features, "enable_regime_features", False):
        for c in REGIME_FEATURES:
            if c in df.columns and c not in available:
                available.append(c)

    return available


def _add_label_prior_features(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Compute leakage-safe label prior regime features.

    Adds ``label_prior_long_lag1`` and ``label_prior_short_lag1`` — rolling
    100-bar fraction of LONG/SHORT labels from **past** bars only.

    The shift by ``horizon_bars + 1`` ensures that at bar T we only use labels
    whose event-end is strictly before T, preventing any lookahead leakage
    from the triple-barrier labeling process.
    """
    if "label" not in df.columns:
        return df

    horizon = config.labels.horizon_bars
    shift_n = horizon + 1

    is_long = (pl.col("label") == 1).cast(pl.Float64)
    is_short = (pl.col("label") == -1).cast(pl.Float64)

    return df.with_columns(
        [
            is_long.shift(shift_n)
            .rolling_mean(window_size=100)
            .fill_null(0.0)
            .alias("label_prior_long_lag1"),
            is_short.shift(shift_n)
            .rolling_mean(window_size=100)
            .fill_null(0.0)
            .alias("label_prior_short_lag1"),
        ]
    )


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
    logger.info(
        "Feature pipeline: %d static cols → k_best=%d (SelectKBest)",
        len(static_cols),
        k_best,
    )
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
        preselect_cols = [
            str(col) for col in feature_pipeline[:-1].get_feature_names_out()
        ]
        if not isinstance(preselect, pl.DataFrame):
            preselect = pl.DataFrame(preselect, schema=preselect_cols)
        selected_mask = feature_pipeline.named_steps["select_k_best"].get_support()
        selected_cols = [
            col
            for col, keep in zip(preselect_cols, selected_mask, strict=False)
            if keep
        ]
        if not selected_cols:
            selected_cols = list(preselect.columns[: min(5, preselect.width)])
        logger.info(
            "Selected %d/%d features: %s",
            len(selected_cols),
            len(preselect_cols),
            selected_cols,
        )
        return feature_pipeline, selected_cols
    except ValueError as exc:
        logger.warning("Static feature selection fallback activated: %s", str(exc))
        fallback_cols = list(static_cols)
        fallback_pipeline = Pipeline(steps=[("scaler", RobustScaler())])
        fallback_pipeline.fit(X_train[fallback_cols], y_train)
        X_train[fallback_cols] = fallback_pipeline.transform(X_train[fallback_cols])
        return fallback_pipeline, fallback_cols
