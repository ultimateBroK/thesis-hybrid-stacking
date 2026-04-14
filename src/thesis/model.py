"""Hybrid GRU + LightGBM model training.

Architecture:
    1. Train GRU feature extractor on log_returns + rsi_14 sequences
    2. Extract GRU hidden states (64-dim) for train/val/test
    3. Concatenate hidden states with static features → LightGBM
    4. LightGBM is the sole decision maker (no meta-learner)

The GRU is a sequence encoder — it does NOT predict directly.
LightGBM consumes both temporal (GRU hidden state) and static features.
"""

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl

from thesis.config import Config
from thesis.gru_model import (
    extract_hidden_states,
    prepare_sequences,
    save_gru_model,
    train_gru,
)

logger = logging.getLogger("thesis.model")


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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train_model(config: Config) -> None:
    """Train hybrid GRU + LightGBM model and save predictions.

    Args:
        config: Loaded application configuration.
    """
    train_path = Path(config.paths.train_data)
    val_path = Path(config.paths.val_data)
    test_path = Path(config.paths.test_data)

    for p in (train_path, val_path, test_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Split data not found: {p}. Run split stage first."
            )

    # Load splits
    train_df = pl.read_parquet(train_path)
    val_df = pl.read_parquet(val_path)
    test_df = pl.read_parquet(test_path)

    logger.info(
        "Loaded splits — train: %d, val: %d, test: %d",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    # --- 1. Train GRU feature extractor ---
    logger.info("=== Stage 1: GRU Feature Extractor ===")
    gru_model, _gru_classifier, train_hidden, val_hidden = train_gru(
        config, train_df, val_df
    )

    # Save GRU model
    gru_path = Path(config.paths.model).parent / "gru_model.pt"
    save_gru_model(gru_model, config, gru_path)

    # Extract hidden states for test set
    gru_cols = ["log_returns", "rsi_14", "atr_14", "macd_hist"]
    test_seq, _, _ = prepare_sequences(test_df, gru_cols, config.gru.sequence_length)
    test_hidden = extract_hidden_states(gru_model, test_seq, config.gru.batch_size)

    # --- 2. Align DataFrames with GRU sequences ---
    seq_len = config.gru.sequence_length
    # GRU produces (n - seq_len + 1) outputs; skip first (seq_len - 1) rows
    train_aligned = train_df.slice(seq_len - 1, len(train_hidden))
    val_aligned = val_df.slice(seq_len - 1, len(val_hidden))
    test_aligned = test_df.slice(seq_len - 1, len(test_hidden))

    logger.info(
        "After GRU alignment — train: %d, val: %d, test: %d",
        len(train_aligned),
        len(val_aligned),
        len(test_aligned),
    )

    # --- 3. Build hybrid feature matrix (GRU hidden + static) ---
    logger.info("=== Stage 2: LightGBM on GRU + Static Features ===")

    static_cols = [c for c in train_aligned.columns if c not in _EXCLUDE_COLS]
    logger.info("Static features (%d): %s", len(static_cols), static_cols)

    # Build combined feature names: gru_h0..gru_h63 + static_cols
    hidden_size = config.gru.hidden_size
    gru_feat_names = [f"gru_h{i}" for i in range(hidden_size)]
    all_feature_cols = gru_feat_names + static_cols

    # Concatenate GRU hidden states with static features
    X_train = np.concatenate(
        [train_hidden, train_aligned.select(static_cols).to_numpy()], axis=1
    )
    X_val = np.concatenate(
        [val_hidden, val_aligned.select(static_cols).to_numpy()], axis=1
    )
    X_test = np.concatenate(
        [test_hidden, test_aligned.select(static_cols).to_numpy()], axis=1
    )

    y_train = train_aligned["label"].to_numpy().astype(np.int32)
    y_val = val_aligned["label"].to_numpy().astype(np.int32)
    y_test = test_aligned["label"].to_numpy().astype(np.int32)

    logger.info(
        "Hybrid feature matrix: %d features (%d GRU + %d static)",
        len(all_feature_cols),
        hidden_size,
        len(static_cols),
    )

    # --- 4. Train LightGBM ---
    class_weights = _compute_class_weights(y_train)

    if config.model.use_optuna:
        model = _train_optuna(
            X_train, y_train, X_val, y_val, class_weights, config, all_feature_cols
        )
    else:
        model = _train_fixed(
            X_train, y_train, X_val, y_val, class_weights, config, all_feature_cols
        )

    # Save LightGBM model
    model_path = Path(config.paths.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info("Saved LightGBM model: %s", model_path)

    # --- 5. Generate test predictions ---
    logger.info("Generating test predictions...")
    proba = model.predict_proba(_wrap_np(X_test, all_feature_cols))
    preds = np.argmax(proba, axis=1) - 1  # Map 0,1,2 → -1,0,1

    acc = (preds == y_test).mean()
    logger.info("Test accuracy: %.4f", acc)

    preds_df = pl.DataFrame(
        {
            "timestamp": test_aligned["timestamp"],
            "true_label": y_test,
            "pred_label": preds.astype(np.int32),
            "pred_proba_class_minus1": proba[:, 0],
            "pred_proba_class_0": proba[:, 1],
            "pred_proba_class_1": proba[:, 2],
        }
    )

    preds_path = Path(config.paths.predictions)
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    preds_df.write_parquet(preds_path)
    logger.info("Saved predictions: %s (%d rows)", preds_path, len(preds_df))

    # --- 6. SHAP feature importance ---
    _compute_shap(model, X_test, all_feature_cols, config)

    # --- 7. Save feature importance ---
    _save_feature_importance(model, all_feature_cols, config)


# ---------------------------------------------------------------------------
# LightGBM training
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# SHAP + feature importance
# ---------------------------------------------------------------------------


def _compute_shap(
    model: Any, X_test: np.ndarray, feature_cols: list[str], config: Config
) -> None:
    """Compute and save SHAP summary."""
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        n_samples = min(500, len(X_test))
        X_sample = _wrap_np(X_test[:n_samples], feature_cols)
        shap_values = explainer.shap_values(X_sample)

        # Multiclass models return 3-D (samples × features × classes).
        # Convert to a list of 2-D arrays so summary_plot handles each
        # class correctly instead of misinterpreting as interaction values.
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = [shap_values[:, :, i] for i in range(shap_values.shape[2])]

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 8))
        rng = np.random.default_rng(config.workflow.random_seed)
        shap.summary_plot(
            shap_values, X_sample, feature_names=feature_cols, show=False, rng=rng
        )
        if config.paths.session_dir:
            out = Path(config.paths.session_dir) / "reports" / "shap_summary.png"
        else:
            out = Path("results/shap_summary.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("SHAP summary saved: %s", out)
    except Exception as e:
        logger.warning("SHAP computation failed: %s", e)


def _save_feature_importance(
    model: Any, feature_cols: list[str], config: Config
) -> None:
    """Save feature importance as JSON."""
    try:
        imp = model.feature_importances_
        pairs = sorted(zip(feature_cols, imp), key=lambda x: x[1], reverse=True)
        if config.paths.session_dir:
            out_path = (
                Path(config.paths.session_dir) / "reports" / "feature_importance.json"
            )
        else:
            out_path = Path("results/feature_importance.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({name: float(val) for name, val in pairs}, f, indent=2)
        logger.info(
            "Feature importance saved: %s (top 5: %s)",
            out_path,
            [p[0] for p in pairs[:5]],
        )
    except Exception as e:
        logger.warning("Feature importance save failed: %s", e)
