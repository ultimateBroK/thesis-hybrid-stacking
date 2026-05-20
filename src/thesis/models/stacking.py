"""Hybrid stacking classifier for Stage 3."""

from __future__ import annotations

from typing import Any

import numpy as np

from thesis.models.estimators import (
    CLASS_ORDER,
    fit_base_models,
    predict_base_probabilities,
    predict_proba_aligned,
)
from thesis.shared.config import Config


def chronological_meta_split(
    X: np.ndarray,
    y: np.ndarray,
    meta_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split train data into base-train and meta-train slices."""
    base_rows = max(1, int(round(len(X) * (1.0 - meta_fraction))))
    if base_rows >= len(X):
        base_rows = max(1, len(X) - 1)
    return X[:base_rows], X[base_rows:], y[:base_rows], y[base_rows:]


def stack_probability_features(
    base_outputs: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    """Convert base model probabilities into meta features."""
    matrices, names = [], []
    for name in sorted(base_outputs):
        matrices.append(base_outputs[name])
        names.extend(f"{name}_proba_{label}" for label in ("short", "hold", "long"))
    return np.hstack(matrices), names


def fit_meta_model(X: np.ndarray, y: np.ndarray, config: Config) -> Any:
    """Fit meta-learner on stacked base-model probability features."""
    if len(np.unique(y)) < 2:
        from sklearn.dummy import DummyClassifier

        return DummyClassifier(strategy="most_frequent").fit(X, y)

    meta = config.model.stacking_meta
    n_classes = len(CLASS_ORDER)

    if meta.learner == "lightgbm":
        import lightgbm as lgb

        classes, counts = np.unique(y, return_counts=True)
        weight_dict = {c: len(y) / (n_classes * cnt) for c, cnt in zip(classes, counts)}
        sample_weights = np.array([weight_dict[v] for v in y])

        return lgb.LGBMClassifier(
            objective="multiclass",
            num_class=n_classes,
            num_leaves=meta.num_leaves,
            max_depth=meta.max_depth,
            learning_rate=meta.learning_rate,
            n_estimators=meta.n_estimators,
            min_child_samples=meta.min_child_samples,
            subsample=meta.subsample,
            subsample_freq=meta.subsample_freq,
            colsample_bytree=meta.feature_fraction,
            reg_alpha=meta.reg_alpha,
            reg_lambda=meta.reg_lambda,
            min_split_gain=meta.min_split_gain,
            random_state=config.workflow.random_seed,
            n_jobs=config.workflow.n_jobs,
            verbose=-1,
        ).fit(X, y, sample_weight=sample_weights)

    # Default: logistic regression
    from sklearn.linear_model import LogisticRegression

    lr_penalty = meta.penalty
    lr_solver = meta.solver
    lr_kwargs: dict[str, Any] = {}

    if meta.l1_ratio is not None and meta.penalty == "elasticnet":
        lr_kwargs["l1_ratio"] = meta.l1_ratio

    if lr_penalty in ("l1", "elasticnet"):
        lr_solver = "saga"

    return LogisticRegression(
        C=meta.meta_C,
        penalty=lr_penalty,
        class_weight="balanced",
        max_iter=meta.max_iter,
        solver=lr_solver,
        random_state=config.workflow.random_seed,
        **lr_kwargs,
    ).fit(X, y)


class HybridStackingClassifier:
    """LR/RF/LightGBM base probabilities feeding a configurable meta learner."""

    def __init__(self, config: Config, feature_cols: list[str]) -> None:
        """Store config and fixed feature names."""
        self.config = config
        self.feature_cols = feature_cols
        self.classes_ = CLASS_ORDER
        self.base_models: dict[str, Any] = {}
        self.meta_model: Any | None = None
        self.meta_feature_names: list[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> HybridStackingClassifier:
        """Fit base models then meta model with chronological holdout."""
        base_X, meta_X, base_y, meta_y = chronological_meta_split(
            X,
            y,
            self.config.model.stacking.meta_fraction,
        )
        self.base_models = fit_base_models(
            base_X,
            base_y,
            self.config,
            self.feature_cols,
        )
        meta_outputs = predict_base_probabilities(
            self.base_models,
            meta_X,
            self.feature_cols,
        )
        meta_features, self.meta_feature_names = stack_probability_features(
            meta_outputs
        )
        self.meta_model = fit_meta_model(meta_features, meta_y, self.config)

        # Refit bases on all train data for final test-window prediction.
        self.base_models = fit_base_models(X, y, self.config, self.feature_cols)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities through base and meta learners."""
        if self.meta_model is None:
            raise RuntimeError("HybridStackingClassifier is not fitted")
        base_outputs = predict_base_probabilities(
            self.base_models,
            X,
            self.feature_cols,
        )
        meta_features, _ = stack_probability_features(base_outputs)
        return predict_proba_aligned(
            self.meta_model,
            meta_features,
            self.meta_feature_names,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class label in {-1, 0, 1}."""
        return CLASS_ORDER[np.argmax(self.predict_proba(X), axis=1)].astype(np.int32)
