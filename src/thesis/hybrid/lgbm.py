"""LightGBM utilities shared between hybrid training and ablation."""

import logging
from typing import Any

import numpy as np

from thesis.config import Config

logger = logging.getLogger("thesis.hybrid.lgbm")


def _wrap_np(X: np.ndarray, feature_cols: list[str]) -> Any:
    """Wrap numpy array in DataFrame to preserve feature names for LightGBM."""
    import pandas as pd

    return pd.DataFrame(X, columns=feature_cols)


_EXCLUDE_COLS = frozenset(
    [
        "timestamp",
        "label",
        "tp_price",
        "sl_price",
        "touched_bar",
        "open_right",  # Label-derived — pure look-ahead
        "high_right",  # Label-derived — pure look-ahead
        "low_right",  # Label-derived — pure look-ahead
        "close_right",  # Label-derived — pure look-ahead
        "open",
        "high",
        "low",
        "close",
        "volume",
        "avg_spread",
        "tick_count",
        "dead_hour",
        "log_returns",  # GRU input — not a static feature
    ]
)


def _compute_class_weights(y: np.ndarray) -> dict[int, float]:
    """Compute balanced class weights."""
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def _train_fixed(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weights: dict[int, float],
    config: Config,
    feature_cols: list[str],
) -> Any:
    """Train LightGBM with fixed hyperparameters."""
    import lightgbm as lgb

    m = config.model
    model = lgb.LGBMClassifier(
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
        class_weight=class_weights,
        objective="multiclass",
        num_class=3,
        random_state=config.workflow.random_seed,
        n_jobs=config.workflow.n_jobs,
        verbose=-1,
    )
    model.fit(
        _wrap_np(X_train, feature_cols),
        y_train,
        eval_set=[(_wrap_np(X_val, feature_cols), y_val)],
        callbacks=[lgb.early_stopping(m.early_stopping_rounds, verbose=False)],
    )
    logger.info("Best iteration: %d", model.best_iteration_)
    return model


def _train_optuna(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weights: dict[int, float],
    config: Config,
    feature_cols: list[str],
) -> Any:
    """Train LightGBM with Optuna hyperparameter optimisation."""
    import lightgbm as lgb
    import optuna
    from sklearn.metrics import f1_score
    from sklearn.model_selection import TimeSeriesSplit

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    seed = config.workflow.random_seed

    def objective(trial: Any) -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 150),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "class_weight": class_weights,
            "objective": "multiclass",
            "num_class": 3,
            "random_state": seed,
            "n_jobs": -1,
            "verbose": -1,
        }

        tscv = TimeSeriesSplit(n_splits=3, gap=config.splitting.purge_bars)
        scores = []
        for tr_idx, va_idx in tscv.split(X_train):
            m = lgb.LGBMClassifier(**params)
            m.fit(
                _wrap_np(X_train[tr_idx], feature_cols),
                y_train[tr_idx],
                eval_set=[(_wrap_np(X_train[va_idx], feature_cols), y_train[va_idx])],
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )
            preds = m.predict(_wrap_np(X_train[va_idx], feature_cols))
            scores.append(f1_score(y_train[va_idx], preds, average="macro"))

        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(
        objective,
        n_trials=config.model.optuna_trials,
        timeout=config.model.optuna_timeout,
    )

    logger.info("Optuna best F1: %.4f", study.best_value)
    logger.info("Optuna best params: %s", study.best_params)

    best = study.best_params
    model = lgb.LGBMClassifier(
        **best,
        class_weight=class_weights,
        objective="multiclass",
        num_class=3,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        _wrap_np(X_train, feature_cols),
        y_train,
        eval_set=[(_wrap_np(X_val, feature_cols), y_val)],
        callbacks=[
            lgb.early_stopping(config.model.early_stopping_rounds, verbose=False)
        ],
    )
    return model
