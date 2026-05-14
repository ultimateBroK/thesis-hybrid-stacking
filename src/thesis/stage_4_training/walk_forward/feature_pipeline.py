"""Feature pipeline: select + scale static features for tabular models."""

from __future__ import annotations

import logging

from feature_engine.selection import DropCorrelatedFeatures, DropDuplicateFeatures
import numpy as np
import polars as pl
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from thesis.shared.config import Config
from thesis.shared.feature_registry import REGIME_FEATURES

logger = logging.getLogger("thesis")


def select_static_cols(
    config: Config,
    df: pl.DataFrame,
    candidates: list[str],
) -> list[str]:
    """Pick static feature columns. Prefer config whitelist, fallback to candidates.

    If enable_regime_features is True, add any regime columns present in df.
    """
    available = [c for c in config.features.static_feature_cols if c in df.columns]
    if not available:
        available = [c for c in candidates if c in df.columns]

    if getattr(config.features, "enable_regime_features", False):
        for c in REGIME_FEATURES:
            if c in df.columns and c not in available:
                available.append(c)

    return available


def _add_label_prior_features(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Add leakage-safe label prior regime features.

    Rolling 100-bar fraction of LONG/SHORT labels from past bars only.
    Shift by horizon_bars + 1 prevents label lookahead.
    """
    if "label" not in df.columns:
        return df

    h = config.labels.horizon_bars
    shift_n = h + 1

    is_long = (pl.col("label") == 1).cast(pl.Float64)
    is_short = (pl.col("label") == -1).cast(pl.Float64)

    return df.with_columns(
        [
            is_long.shift(shift_n)
            .rolling_mean(100)
            .fill_null(0.0)
            .alias("label_prior_long_lag1"),
            is_short.shift(shift_n)
            .rolling_mean(100)
            .fill_null(0.0)
            .alias("label_prior_short_lag1"),
        ]
    )


def fit_static_feature_pipeline(
    config: Config,
    train_df: pl.DataFrame,
    cols: list[str],
    y_train: np.ndarray,
) -> tuple[Pipeline, list[str]]:
    """Fit scaler + selector pipeline on static features.

    Steps:
        1. DropDuplicateFeatures — remove constant/duplicate columns
        2. DropCorrelatedFeatures — remove highly correlated pairs
        3. RobustScaler — median/mad normalisation, robust to outliers
        4. SelectKBest — univariate feature selection (f_classif)

    Falls back to scaler-only if selection fails.
    """
    X = train_df.select(cols).to_pandas()
    X.columns = cols

    k_best = min(max(5, len(cols) // 2), len(cols))
    logger.info("  Feature pipeline: %d cols → k_best=%d", len(cols), k_best)

    pipe = Pipeline(
        [
            ("dedup", DropDuplicateFeatures(missing_values="ignore")),
            (
                "decorr",
                DropCorrelatedFeatures(
                    threshold=config.features.correlation_threshold,
                    method="pearson",
                ),
            ),
            ("scaler", RobustScaler()),
            ("select", SelectKBest(score_func=f_classif, k=k_best)),
        ]
    )

    try:
        pipe.fit(X, y_train)

        # Get selected column names after pipeline transform
        pre_select = pipe[:-1].get_feature_names_out()
        pre_cols = [str(c) for c in pre_select]
        mask = pipe.named_steps["select"].get_support()
        selected = [c for c, m in zip(pre_cols, mask, strict=False) if m]

        if not selected:
            selected = list(pre_cols[: min(5, len(pre_cols))])

        logger.info(
            "  Selected %d/%d features: %s", len(selected), len(pre_cols), selected
        )
        return pipe, selected

    except ValueError as exc:
        logger.warning("  Feature selection fallback (all cols scaled): %s", exc)
        fallback = Pipeline([("scaler", RobustScaler())])
        fallback.fit(X[cols], y_train)
        return fallback, list(cols)
