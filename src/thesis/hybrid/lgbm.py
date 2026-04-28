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
    """Wrap a NumPy matrix as a pandas DataFrame.

    Args:
        X: Feature matrix of shape ``(n_samples, n_features)``.
        feature_cols: Feature names aligned to matrix columns.

    Returns:
        A pandas DataFrame preserving feature names for LightGBM.
    """
    import pandas as pd

    return pd.DataFrame(X, columns=feature_cols)


# Column sets — imported from thesis.constants (single source of truth)


def _compute_class_weights(y: np.ndarray) -> dict[int, float]:
    """Compute balanced class weights for multiclass labels.

    Args:
        y: Target labels.

    Returns:
        Mapping from class label to balanced class weight.
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
    """Train LightGBM with fixed hyperparameters.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        X_val: Validation feature matrix.
        y_val: Validation labels.
        class_weights: Balanced class weights.
        config: Resolved application configuration.
        feature_cols: Ordered feature names.

    Returns:
        Fitted ``lightgbm.LGBMClassifier`` model.
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
            """Update progress bar from LightGBM callback state.

            Args:
                env: LightGBM callback environment.
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


# Default bars per year for H1 timeframe (approx 23h/day × 365 days)
_H1_BARS_PER_YEAR = 8400


def _convert_hard_to_proba(y_pred: np.ndarray) -> np.ndarray:
    """Convert 1D hard class predictions to 3-column probability matrix.

    Args:
        y_pred: 1D array of hard class labels (-1, 0, 1).

    Returns:
        2D array of shape (n, 3) with one-hot encoded probabilities.
    """
    n = len(y_pred)
    proba = np.zeros((n, 3), dtype=np.float64)
    for i, pred in enumerate(y_pred):
        if pred == -1:
            proba[i, 0] = 1.0
        elif pred == 0:
            proba[i, 1] = 1.0
        else:
            proba[i, 2] = 1.0
    return proba


def _apply_confidence_filter(
    y_pred: np.ndarray,
    max_probs: np.ndarray,
    confidence_threshold: float,
) -> np.ndarray:
    """Apply confidence threshold by forcing low-confidence predictions to Hold.

    Args:
        y_pred: Predicted class indices.
        max_probs: Confidence values (max probability per sample).
        confidence_threshold: Minimum confidence to trade.

    Returns:
        Modified predictions where low-confidence are set to 0 (Hold).
    """
    y_pred = y_pred.copy()
    y_pred[max_probs < confidence_threshold] = 0
    return y_pred


def _compute_trade_returns(
    correct: np.ndarray,
    spread_cost: float,
) -> np.ndarray:
    """Compute cost-aware returns from correct/incorrect predictions.

    Args:
        correct: Boolean array indicating winning trades.
        spread_cost: Round-trip spread cost fraction.

    Returns:
        Array of returns per trade.
    """
    return np.where(correct, 1.0 - spread_cost, -1.0 - spread_cost)


def _compute_sharpe_from_predictions(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    confidence_threshold: float = 0.0,
    spread_cost: float = 0.0002,
    annualize: bool = False,
    bars_per_year: int = _H1_BARS_PER_YEAR,
) -> float:
    """Compute Sharpe ratio from predicted class probabilities.

    Args:
        y_true: True labels in ``{-1, 0, 1}``.
        y_pred_proba: Class probabilities or hard class predictions.
        confidence_threshold: Minimum confidence required to trade.
        spread_cost: Round-trip transaction cost fraction.
        annualize: Whether to annualize Sharpe.
        bars_per_year: Bars used for annualization.

    Returns:
        Sharpe ratio, or ``0.0`` when trades are insufficient.
    """
    if y_pred_proba.ndim == 1:
        y_pred_proba = _convert_hard_to_proba(y_pred_proba)

    max_probs = np.max(y_pred_proba, axis=1)
    pred_indices = np.argmax(y_pred_proba, axis=1)
    classes = np.array([-1, 0, 1])
    y_pred = classes[pred_indices]
    y_pred = _apply_confidence_filter(y_pred, max_probs, confidence_threshold)

    mask = y_pred != 0
    if mask.sum() < 10:
        return 0.0

    correct = y_pred == y_true
    returns = _compute_trade_returns(correct, spread_cost)[mask]

    if len(returns) < 10:
        return 0.0

    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)

    if std_ret == 0:
        return 0.0

    sharpe = mean_ret / std_ret
    n_trades = len(returns)

    # Annualize only when actual trade count is known (e.g., final backtest).
    # During Optuna CV we return unannualized Sharpe to avoid inflated estimates
    # from using theoretical bars_per_year instead of actual trades per year.
    if annualize and n_trades > 0:
        # Use actual trade count for annualization (known in backtest context)
        trades_per_year = min(bars_per_year, n_trades * 2)  # cap at bars_per_year
        sharpe = sharpe * np.sqrt(trades_per_year)

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
    """Tune and train LightGBM with Optuna.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        X_val: Validation feature matrix.
        y_val: Validation labels.
        class_weights: Balanced class weights.
        config: Resolved application configuration.
        feature_cols: Ordered feature names.

    Returns:
        Fitted ``lightgbm.LGBMClassifier`` model with best Optuna params.
    """
    import lightgbm as lgb
    import optuna
    from sklearn.model_selection import TimeSeriesSplit

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    seed = config.workflow.random_seed

    def objective(trial: Any) -> float:
        """Score a trial using time-series CV Sharpe ratio.

        Args:
            trial: Optuna trial proposing hyperparameters.

        Returns:
            Mean Sharpe ratio across CV folds.
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
            """Update Optuna progress tracking after each trial.

            Args:
                study: Optuna study state.
                trial: Completed trial.
            """
            nonlocal best_sharpe
            _ = trial
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
            """Update final-model progress from callback state.

            Args:
                env: LightGBM callback environment.
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
