"""Model factories for Stage 3 classification experiments."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from thesis.shared.config import Config

logger = logging.getLogger("thesis")

CLASS_ORDER = np.array([-1, 0, 1], dtype=np.int32)


def wrap_feature_matrix(X: np.ndarray, feature_cols: list[str]) -> Any:
    """Return DataFrame when estimator needs stable feature names."""
    import pandas as pd

    return pd.DataFrame(X, columns=feature_cols)


def compute_class_weights(y: np.ndarray) -> dict[int, float]:
    """Balanced class weights for observed labels."""
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def align_proba(proba: np.ndarray, class_order: list[int] | np.ndarray) -> np.ndarray:
    """Align estimator probabilities to [-1, 0, 1]."""
    aligned = np.zeros((len(proba), len(CLASS_ORDER)), dtype=np.float64)
    index_map = {int(c): i for i, c in enumerate(class_order)}
    for target_idx, cls in enumerate(CLASS_ORDER):
        src = index_map.get(int(cls))
        if src is not None:
            aligned[:, target_idx] = proba[:, src]
    return aligned


def predict_proba_aligned(
    model: Any,
    X: np.ndarray,
    feature_cols: list[str] | None = None,
) -> np.ndarray:
    """Reorder probability columns to [-1, 0, 1].

    Estimators may return classes in arbitrary order depending on
    the order classes appear in training data.
    """
    X_input = X
    fitted_names = getattr(model, "feature_names_in_", None)
    if (
        fitted_names is not None
        and feature_cols is not None
        and len(feature_cols) == X.shape[1]
    ):
        X_input = wrap_feature_matrix(X, feature_cols)
    return align_proba(model.predict_proba(X_input), model.classes_)


def build_logistic_regression(config: Config) -> Any:
    """Build scaled multinomial logistic regression."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler

    return Pipeline(
        [
            ("scaler", RobustScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=config.workflow.random_seed,
                ),
            ),
        ]
    )


def build_random_forest(config: Config) -> Any:
    """Build random forest baseline model."""
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=config.model.random_forest_n_estimators,
        max_depth=config.model.random_forest_max_depth,
        min_samples_leaf=config.model.random_forest_min_samples_leaf,
        class_weight="balanced_subsample",
        random_state=config.workflow.random_seed,
        n_jobs=config.workflow.n_jobs,
    )


def build_lightgbm(config: Config) -> Any:
    """Build LightGBM multiclass classifier."""
    import lightgbm as lgb

    m = config.model
    return lgb.LGBMClassifier(
        num_leaves=m.num_leaves,
        max_depth=m.max_depth,
        learning_rate=m.learning_rate,
        n_estimators=m.n_estimators,
        min_child_samples=m.min_child_samples,
        subsample=m.subsample,
        subsample_freq=m.subsample_freq,
        colsample_bytree=m.feature_fraction,
        reg_alpha=m.reg_alpha,
        reg_lambda=m.reg_lambda,
        interaction_constraints=[],
        objective="multiclass",
        num_class=3,
        random_state=config.workflow.random_seed,
        n_jobs=config.workflow.n_jobs,
        verbose=-1,
        use_missing=False,
        zero_as_missing=False,
    )


def build_base_models(config: Config) -> dict[str, Any]:
    """Build fixed Stage 3 model set excluding Hybrid Stacking."""
    return {
        "logistic_regression": build_logistic_regression(config),
        "random_forest": build_random_forest(config),
        "lightgbm": build_lightgbm(config),
    }


def is_lightgbm_model(model: Any) -> bool:
    """Check if model is a LightGBM classifier."""
    return model.__class__.__name__ == "LGBMClassifier"


def fit_lightgbm(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_cols: list[str],
) -> Any:
    """Fit LightGBM with class weights and named feature matrix."""
    return model.set_params(class_weight=compute_class_weights(y)).fit(
        wrap_feature_matrix(X, feature_cols),
        y,
    )


def fit_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_cols: list[str] | None = None,
) -> Any:
    """Fit estimator, using dummy fallback for single-class folds."""
    if len(np.unique(y)) < 2:
        from sklearn.dummy import DummyClassifier

        return DummyClassifier(strategy="most_frequent").fit(X, y)
    if feature_cols and is_lightgbm_model(model):
        return fit_lightgbm(model, X, y, feature_cols)
    return model.fit(X, y)


def fit_base_models(
    X: np.ndarray,
    y: np.ndarray,
    config: Config,
    feature_cols: list[str],
) -> dict[str, Any]:
    """Fit LR, RF, and LightGBM on one training slice."""
    return {
        name: fit_model(model, X, y, feature_cols)
        for name, model in build_base_models(config).items()
    }


def predict_base_probabilities(
    models: dict[str, Any],
    X: np.ndarray,
    feature_cols: list[str],
) -> dict[str, np.ndarray]:
    """Predict base-model probabilities for meta features."""
    return {
        name: predict_proba_aligned(model, X, feature_cols)
        for name, model in models.items()
    }
