"""LightGBM utilities shared between hybrid training and ablation."""

import logging
import time
from typing import Any

import numpy as np
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from thesis.config import Config

logger = logging.getLogger("thesis.hybrid.lgbm")


def _wrap_np(X: np.ndarray, feature_cols: list[str]) -> Any:
    """
    Convert a NumPy feature matrix into a pandas DataFrame with the given column names to preserve feature names.

    Parameters:
        X (np.ndarray): 2-D array of shape (n_samples, n_features) containing feature values.
        feature_cols (list[str]): Column names to assign to the DataFrame; length must equal the number of columns in `X`.

    Returns:
        pandas.DataFrame: DataFrame representation of `X` with columns named according to `feature_cols`.
    """
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
    """
    Compute balanced class weights for multiclass labels.

    Calculates weights inversely proportional to class frequencies so that each class contributes equally during training. The returned mapping uses integer class labels as keys and their corresponding weight as float values.

    Returns:
        class_weights (dict[int, float]): Mapping from class label to its computed weight.
    """
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
    """
    Train a LightGBM multiclass classifier using fixed hyperparameters from `config` and early stopping on the provided validation set.

    Parameters:
        X_train (np.ndarray): Training feature matrix.
        y_train (np.ndarray): Training labels.
        X_val (np.ndarray): Validation feature matrix used for early stopping.
        y_val (np.ndarray): Validation labels.
        class_weights (dict[int, float]): Mapping from class index to weight used for `class_weight`.
        config (Config): Configuration containing model hyperparameters and workflow settings.
        feature_cols (list[str]): Column names applied to feature matrices to preserve feature names for LightGBM.

    Returns:
        model: Trained `lightgbm.LGBMClassifier` instance with the fitted state (including `best_iteration_`).
    """
    import lightgbm as lgb

    m = config.model
    logger.info(
        "LightGBM: leaves=%d depth=%d lr=%.4f n_est=%d",
        m.num_leaves,
        m.max_depth,
        m.learning_rate,
        m.n_estimators,
    )

    start_time = time.perf_counter()

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

    # Rich progress bar over boosting iterations
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold magenta]LightGBM boosting"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TextColumn("[cyan]v_loss={task.fields[v_loss]:.4f}"),
        TimeElapsedColumn(),
        transient=False,
    )

    with progress:
        task = progress.add_task("iter", total=m.n_estimators, v_loss=0.0)

        def _progress_cb(env: Any) -> None:
            """
            Advance the Rich progress task by one iteration and set the `v_loss` field from the LightGBM callback environment.

            Parameters:
                env (Any): LightGBM callback environment; `env.evaluation_result_list[0][2]` is used as the validation loss when available, otherwise `0.0`.
            """
            progress.update(
                task,
                advance=1,
                v_loss=env.evaluation_result_list[0][2]
                if env.evaluation_result_list
                else 0.0,
            )

        model.fit(
            _wrap_np(X_train, feature_cols),
            y_train,
            eval_set=[(_wrap_np(X_val, feature_cols), y_val)],
            callbacks=[
                lgb.early_stopping(m.early_stopping_rounds, verbose=False),
                _progress_cb,
            ],
        )

    train_time = time.perf_counter() - start_time
    logger.info(
        "LightGBM done: best_iter=%d (%.1fs)",
        model.best_iteration_,
        train_time,
    )
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
    """
    Perform an Optuna hyperparameter search for a LightGBM multiclass classifier using time-series cross-validation, then train and return a final LightGBM model using the best-found parameters.

    Parameters:
        class_weights (dict[int, float]): Mapping from class index to weight applied during training.
        config (Config): Configuration object controlling randomness, Optuna budget, early stopping, and related training settings.
        feature_cols (list[str]): Column names used to wrap NumPy feature matrices so LightGBM preserves feature names.

    Returns:
        model: A fitted LightGBM classifier trained on the provided training set and validated using the supplied validation set.
    """
    import lightgbm as lgb
    import optuna
    from sklearn.metrics import f1_score
    from sklearn.model_selection import TimeSeriesSplit

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    seed = config.workflow.random_seed

    def objective(trial: Any) -> float:
        """
        Evaluate hyperparameters proposed by an Optuna `trial` using 3-fold time-series cross-validation and return the mean macro F1 score.

        Parameters:
            trial: An Optuna trial object that suggests hyperparameter values for a LightGBM multiclass classifier.

        Returns:
            float: Mean macro F1 score across the three TimeSeriesSplit folds (uses a gap of `config.splitting.purge_bars` and LightGBM early stopping of 30 rounds).
        """
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
            "n_jobs": config.workflow.n_jobs,
            "verbose": -1,
        }

        tscv = TimeSeriesSplit(
            n_splits=3, gap=config.splitting.purge_bars + config.splitting.embargo_bars
        )
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

    n_trials = config.model.optuna_trials

    # Rich progress for Optuna trials
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold yellow]Optuna tuning"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TextColumn("[green]best_F1={task.fields[best_f1]:.4f}"),
        TimeElapsedColumn(),
        transient=False,
    )

    best_f1 = 0.0

    with progress:
        task = progress.add_task("trials", total=n_trials, best_f1=0.0)

        def _optuna_cb(
            study: optuna.study.Study, trial: optuna.trial.FrozenTrial
        ) -> None:
            """
            Update the external Rich progress task and store a new best F1 score when the study improves.

            Parameters:
                study (optuna.study.Study): The Optuna study to read the current best value from.
                trial (optuna.trial.FrozenTrial): The trial that just finished (unused except for callback signature).
            """
            nonlocal best_f1
            if study.best_value > best_f1:
                best_f1 = study.best_value
            progress.update(task, advance=1, best_f1=best_f1)

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=config.model.optuna_timeout,
            callbacks=[_optuna_cb],
        )

    logger.info(
        "Optuna done: best_F1=%.4f (trial #%d)",
        study.best_value,
        study.best_trial.number,
    )

    # Final model with best params
    best = study.best_params
    model = lgb.LGBMClassifier(
        **best,
        class_weight=class_weights,
        objective="multiclass",
        num_class=3,
        random_state=seed,
        n_jobs=config.workflow.n_jobs,
        verbose=-1,
    )

    n_est = best.get("n_estimators", 500)
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold magenta]Final LightGBM"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TextColumn("[cyan]v_loss={task.fields[v_loss]:.4f}"),
        TimeElapsedColumn(),
        transient=False,
    )

    with progress:
        task = progress.add_task("iter", total=n_est, v_loss=0.0)

        def _progress_cb(env: Any) -> None:
            """
            Advance the Rich progress task by one iteration and set the `v_loss` field from the LightGBM callback environment.

            Parameters:
                env (Any): LightGBM callback environment; `env.evaluation_result_list[0][2]` is used as the validation loss when available, otherwise `0.0`.
            """
            progress.update(
                task,
                advance=1,
                v_loss=env.evaluation_result_list[0][2]
                if env.evaluation_result_list
                else 0.0,
            )

        model.fit(
            _wrap_np(X_train, feature_cols),
            y_train,
            eval_set=[(_wrap_np(X_val, feature_cols), y_val)],
            callbacks=[
                lgb.early_stopping(config.model.early_stopping_rounds, verbose=False),
                _progress_cb,
            ],
        )

    logger.info("Final model: best_iteration=%d", model.best_iteration_)
    return model
