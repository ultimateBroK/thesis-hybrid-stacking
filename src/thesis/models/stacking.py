"""Hybrid stacking classifier for Stage 3."""

from __future__ import annotations

from typing import Any

import numpy as np

from thesis.models.estimators import (
    fit_base_models,
    get_class_order,
    predict_base_probabilities,
    predict_proba_aligned,
)
from thesis.shared.config import Config

CLASS_LABEL_NAMES = {-1: "short", 0: "hold", 1: "long"}


def chronological_meta_split(
    X: np.ndarray,
    y: np.ndarray,
    meta_fraction: float,
    sample_weight: np.ndarray | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Split train data into base-train and meta-train slices.

    Returns (X_base, X_meta, y_base, y_meta, sw_base, sw_meta).
    Weight arrays are None when sample_weight is None.
    """
    base_rows = max(1, int(round(len(X) * (1.0 - meta_fraction))))
    if base_rows >= len(X):
        base_rows = max(1, len(X) - 1)
    splits = (X[:base_rows], X[base_rows:], y[:base_rows], y[base_rows:])
    if sample_weight is not None:
        return (*splits, sample_weight[:base_rows], sample_weight[base_rows:])
    return (*splits, None, None)


def stack_probability_features(
    base_outputs: dict[str, np.ndarray],
    class_labels: tuple[str, ...] = ("short", "long"),
) -> tuple[np.ndarray, list[str]]:
    """Convert base model probabilities into meta feature columns.

    Each base model contributes one probability column per class label,
    producing n_models * n_classes meta features total.
    """
    matrices, names = [], []
    for name in sorted(base_outputs):
        matrices.append(base_outputs[name])
        names.extend(f"{name}_proba_{label}" for label in class_labels)
    return np.hstack(matrices), names


def fit_meta_model(
    X: np.ndarray,
    y: np.ndarray,
    config: Config,
    num_classes: int = 2,
) -> Any:
    """Fit meta-learner on stacked base-model probability features.

    Uses weighted LightGBM or balanced Logistic Regression depending
    on stacking_meta config.  Switches to multiclass objective when
    num_classes > 2.
    """
    if len(np.unique(y)) < 2:
        from sklearn.dummy import DummyClassifier

        return DummyClassifier(strategy="most_frequent").fit(X, y)

    meta = config.model.stacking_meta
    n_classes = num_classes

    if meta.learner == "lightgbm":
        import lightgbm as lgb

        objective = "multiclass" if n_classes > 2 else "binary"
        classes, counts = np.unique(y, return_counts=True)
        weight_dict = {c: len(y) / (n_classes * cnt) for c, cnt in zip(classes, counts)}
        sample_weights = np.array([weight_dict[v] for v in y])

        kw: dict[str, Any] = dict(
            objective=objective,
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
        )
        if n_classes > 2:
            kw["num_class"] = n_classes

        return lgb.LGBMClassifier(**kw).fit(X, y, sample_weight=sample_weights)

    # Default: logistic regression (handles multiclass natively)
    from sklearn.linear_model import LogisticRegression

    lr_solver = meta.solver
    lr_kwargs: dict[str, Any] = {}

    # sklearn 1.8+: map penalty -> l1_ratio to avoid deprecation
    if meta.penalty == "elasticnet":
        lr_kwargs["penalty"] = "elasticnet"
        lr_kwargs["l1_ratio"] = meta.l1_ratio if meta.l1_ratio is not None else 0.5
        lr_solver = "saga"
    elif meta.penalty == "l1":
        lr_kwargs["l1_ratio"] = 1.0
        lr_solver = "saga"
    elif meta.penalty == "none":
        lr_kwargs["C"] = np.inf
    else:  # "l2" (default)
        lr_kwargs["l1_ratio"] = 0.0

    return LogisticRegression(
        C=meta.meta_C,
        class_weight="balanced",
        max_iter=meta.max_iter,
        solver=lr_solver,
        random_state=config.workflow.random_seed,
        **lr_kwargs,
    ).fit(X, y)


class HybridStackingClassifier:
    """LR/RF/LightGBM base probabilities feeding a configurable meta learner.

    Supports 2-class (short/long) and 3-class (short/hold/long) labeling.
    """

    def __init__(self, config: Config, feature_cols: list[str]) -> None:
        """Store config, feature names, and dynamic class metadata."""
        self.config = config
        self.feature_cols = feature_cols
        self.num_classes = config.labels.num_classes
        self.class_order = get_class_order(self.num_classes)
        self.class_labels = tuple(CLASS_LABEL_NAMES[int(c)] for c in self.class_order)
        self.classes_ = self.class_order
        self.base_models: dict[str, Any] = {}
        self.meta_model: Any | None = None
        self.meta_feature_names: list[str] = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> HybridStackingClassifier:
        """Fit base models on base-split, meta model on meta-split.

        Chronological holdout: earlier rows train base models,
        later rows train the meta learner.  Base models are then
        refit on all training data for final prediction.
        """
        base_X, meta_X, base_y, meta_y, base_sw, meta_sw = chronological_meta_split(
            X,
            y,
            self.config.model.stacking.meta_fraction,
            sample_weight,
        )

        # Fit base models on base-split portion
        self.base_models = fit_base_models(
            base_X,
            base_y,
            self.config,
            self.feature_cols,
            sample_weight=base_sw,
            num_classes=self.num_classes,
        )

        # Generate meta features from base model probabilities
        meta_outputs = predict_base_probabilities(
            self.base_models,
            meta_X,
            self.feature_cols,
        )
        meta_features, self.meta_feature_names = stack_probability_features(
            meta_outputs,
            self.class_labels,
        )

        self.meta_model = fit_meta_model(
            meta_features,
            meta_y,
            self.config,
            self.num_classes,
        )

        # Refit bases on all train data for final test-window prediction
        self.base_models = fit_base_models(
            X,
            y,
            self.config,
            self.feature_cols,
            sample_weight=sample_weight,
            num_classes=self.num_classes,
        )
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
        meta_features, _ = stack_probability_features(
            base_outputs,
            self.class_labels,
        )
        return predict_proba_aligned(
            self.meta_model,
            meta_features,
            self.meta_feature_names,
            target_order=self.class_order,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class label from fitted meta-learner probabilities."""
        return self.class_order[np.argmax(self.predict_proba(X), axis=1)].astype(
            np.int32
        )
