"""Hybrid GRU + LightGBM model training — main orchestrator."""

import joblib
import logging
import numpy as np
import polars as pl
from pathlib import Path

from thesis.config import Config
from thesis.gru import (
    extract_hidden_states,
    prepare_sequences,
    save_gru_model,
    train_gru,
)
from thesis.hybrid.lgbm import (
    _EXCLUDE_COLS,
    _wrap_np,
    _compute_class_weights,
    _train_fixed,
    _train_optuna,
)
from thesis.hybrid.interpret import _compute_shap, _save_feature_importance

logger = logging.getLogger("thesis.hybrid.train")


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
