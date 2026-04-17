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


def _compute_sharpe_from_predictions(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    confidence_threshold: float = 0.0,
) -> float:
    """
    Compute Sharpe Ratio from prediction labels using a simplified trade simulation.

    Simulates fixed-lot trades based on predicted direction vs actual direction.
    Uses simplified returns without full backtesting.py overhead for fast Optuna evaluation.

    Parameters:
        y_true: True labels (-1, 0, 1)
        y_pred_proba: Predicted class probabilities (3 columns for classes -1, 0, 1),
                      or 1D array of hard class predictions.
        confidence_threshold: Minimum probability threshold to trade (0 = trade all)

    Returns:
        Sharpe Ratio (annualized), or 0.0 if insufficient trades.
    """
    # Handle 1D hard predictions by converting to probability format
    if y_pred_proba.ndim == 1:
        n = len(y_pred_proba)
        proba = np.zeros((n, 3), dtype=np.float64)
        for i, pred in enumerate(y_pred_proba):
            if pred == -1:
                proba[i, 0] = 1.0
            elif pred == 0:
                proba[i, 1] = 1.0
            else:  # pred == 1
                proba[i, 2] = 1.0
        y_pred_proba = proba

    # 1. Find the class with highest probability and its confidence value
    max_probs = np.max(y_pred_proba, axis=1)
    pred_indices = np.argmax(y_pred_proba, axis=1)

    # 2. Map index (0, 1, 2) back to original labels (-1, 0, 1).
    # LightGBM always sorts classes, so the default order is [-1, 0, 1]
    classes = np.array([-1, 0, 1])
    y_pred = classes[pred_indices]

    # 3. Apply confidence threshold: if confidence is below threshold, force to Hold (0)
    y_pred[max_probs < confidence_threshold] = 0

    # Filter to non-hold predictions only
    mask = y_pred != 0
    if mask.sum() < 10:
        return 0.0

    correct = y_pred == y_true
    # Simple return: +1 for correct direction, -1 for wrong direction
    # (actual P&L would depend on price move, but for direction this is a proxy)
    direction_returns = np.where(correct, 1.0, -1.0)

    # Remove hold periods (zeros)
    returns = direction_returns[mask]

    if len(returns) < 10:
        return 0.0

    # Annualization factor (1H bars → ~8760 bars/year)
    annualization = np.sqrt(8760 / len(returns))

    mean_ret = np.mean(returns)
    std_ret = np.std(returns)

    if std_ret == 0:
        return 0.0

    sharpe = (mean_ret / std_ret) * annualization
    return float(sharpe)


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
    Perform an Optuna hyperparameter search for a LightGBM multiclass classifier using time-series cross-validation optimizing for Sharpe Ratio, then train and return a final LightGBM model using the best-found parameters.

    Parameters:
        X_train (np.ndarray): Training feature matrix.
        y_train (np.ndarray): Training labels.
        X_val (np.ndarray): Validation feature matrix used for early stopping of final model.
        y_val (np.ndarray): Validation labels.
        class_weights (dict[int, float]): Mapping from class index to weight applied during training.
        config (Config): Configuration object controlling randomness, Optuna budget, early stopping, and related training settings.
        feature_cols (list[str]): Column names used to wrap NumPy feature matrices so LightGBM preserves feature names.

    Returns:
        model: A fitted LightGBM classifier trained on the provided training set and validated using the supplied validation set.
    """
    import lightgbm as lgb
    import optuna
    from sklearn.model_selection import TimeSeriesSplit

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    seed = config.workflow.random_seed

    def objective(trial: Any) -> float:
        """
        Evaluate hyperparameters proposed by an Optuna `trial` using 3-fold time-series cross-validation and return the mean Sharpe Ratio.

        Optimizes for trading performance (Sharpe Ratio) rather than classification accuracy (F1).

        Parameters:
            trial: An Optuna trial object that suggests hyperparameter values for a LightGBM multiclass classifier.

        Returns:
            float: Mean Sharpe Ratio across the three TimeSeriesSplit folds.
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
            )
            preds_proba = m.predict_proba(_wrap_np(X_train[va_idx], feature_cols))
            sharpe = _compute_sharpe_from_predictions(
                y_train[va_idx], preds_proba, config.backtest.confidence_threshold
            )
            scores.append(sharpe)

        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )

    n_trials = config.model.optuna_trials

    # Rich progress for Optuna trials
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold yellow]Optuna tuning (Sharpe)"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TextColumn("[green]best_sharpe={task.fields[best_sharpe]:.4f}"),
        TimeElapsedColumn(),
        transient=False,
    )

    best_sharpe = -float("inf")

    with progress:
        task = progress.add_task("trials", total=n_trials, best_sharpe=0.0)

        def _optuna_cb(
            study: optuna.study.Study, trial: optuna.trial.FrozenTrial
        ) -> None:
            """
            Update the external Rich progress task and store a new best Sharpe when the study improves.

            Parameters:
                study: The Optuna study to read the current best value from.
                trial: The trial that just finished (unused except for callback signature).
            """
            nonlocal best_sharpe
            if study.best_value > best_sharpe:
                best_sharpe = study.best_value
            progress.update(task, advance=1, best_sharpe=best_sharpe)

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=config.model.optuna_timeout,
            callbacks=[_optuna_cb],
        )

    logger.info(
        "Optuna done: best_sharpe=%.4f (trial #%d)",
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
                env: LightGBM callback environment; `env.evaluation_result_list[0][2]` is used as the validation loss when available, otherwise `0.0`.
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

    logger.info(f"Final model: best_iteration={model.best_iteration_}")
    return model
