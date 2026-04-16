"""Hybrid GRU + LightGBM model training — main orchestrator."""

import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from thesis.config import Config
from thesis.gru import (
    extract_hidden_states,
    prepare_sequences,
    save_gru_model,
    train_gru,
)
from thesis.hybrid.interpret import _compute_shap, _save_feature_importance
from thesis.hybrid.lgbm import (
    _EXCLUDE_COLS,
    _compute_class_weights,
    _train_fixed,
    _train_optuna,
    _wrap_np,
)

logger = logging.getLogger("thesis.hybrid.train")
_console = Console()


def train_model(config: Config) -> None:
    """
    Orchestrates training of a hybrid GRU feature extractor and LightGBM classifier, evaluates on the test split, and saves models, predictions, and artifacts.

    Parameters:
        config (Config): Application configuration containing paths for data, model, and predictions, plus GRU and LightGBM training settings.

    Raises:
        FileNotFoundError: If any of the required split files (train/val/test) do not exist.
    """
    stage_start = time.perf_counter()

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
        "Splits: train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df)
    )

    # --- 1. Train GRU feature extractor ---
    _console.print(
        Panel(
            "Stage 4.1: [bold]GRU Feature Extractor[/]", style="magenta", padding=(0, 2)
        )
    )
    gru_model, _gru_classifier, train_hidden, val_hidden, gru_history = train_gru(
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
    train_aligned = train_df.slice(seq_len - 1, len(train_hidden))
    val_aligned = val_df.slice(seq_len - 1, len(val_hidden))
    test_aligned = test_df.slice(seq_len - 1, len(test_hidden))

    logger.info(
        "Aligned: train=%d val=%d test=%d",
        len(train_aligned),
        len(val_aligned),
        len(test_aligned),
    )

    # --- 3. Build hybrid feature matrix ---
    static_cols = [c for c in train_aligned.columns if c not in _EXCLUDE_COLS]
    hidden_size = config.gru.hidden_size
    gru_feat_names = [f"gru_h{i}" for i in range(hidden_size)]
    all_feature_cols = gru_feat_names + static_cols

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
        "Features: %d total (%d GRU + %d static)",
        len(all_feature_cols),
        hidden_size,
        len(static_cols),
    )

    # --- 4. Train LightGBM ---
    method = "Optuna" if config.model.use_optuna else "Fixed"
    _console.print(
        Panel(
            f"Stage 4.2: [bold]LightGBM[/] ({method})", style="magenta", padding=(0, 2)
        )
    )
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
    logger.info("Saved model: %s", model_path)

    # Save training history
    models_dir = model_path.parent
    history_path = models_dir / "training_history.json"
    lgbm_info: dict = {
        "best_iteration": int(model.best_iteration_)
        if hasattr(model, "best_iteration_")
        else None,
        "n_features": len(all_feature_cols),
        "n_classes": int(model.n_classes_) if hasattr(model, "n_classes_") else None,
    }
    training_history = {
        "gru": gru_history,
        "lightgbm": lgbm_info,
    }
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2)

    # --- 5. Generate test predictions ---
    _console.print(
        Panel(
            "Stage 4.3: [bold]Predictions & Evaluation[/]",
            style="magenta",
            padding=(0, 2),
        )
    )
    proba = model.predict_proba(_wrap_np(X_test, all_feature_cols))
    preds = model.classes_[np.argmax(proba, axis=1)]  # Explicit class mapping

    acc = (preds == y_test).mean()

    # Rich table for per-class results
    table = Table(title="Test Set Results", show_header=True, header_style="bold")
    table.add_column("Class", style="cyan")
    table.add_column("Samples", justify="right")
    table.add_column("Accuracy", justify="right", style="green")
    table.add_column("Predicted", justify="right")

    label_map = {-1: "SELL", 0: "HOLD", 1: "BUY"}
    for cls in [-1, 0, 1]:
        mask = y_test == cls
        if mask.sum() > 0:
            cls_acc = (preds[mask] == cls).mean()
            table.add_row(
                f"{label_map[cls]} ({cls})",
                str(mask.sum()),
                f"{cls_acc:.3f}",
                str((preds == cls).sum()),
            )

    _console.print(table)
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

    # CSV mirror
    csv_path = preds_path.with_suffix(".csv")
    preds_df.write_csv(csv_path)

    # --- 6. SHAP ---
    _console.print(
        Panel(
            "Stage 4.4: [bold]SHAP Feature Importance[/]",
            style="magenta",
            padding=(0, 2),
        )
    )
    _compute_shap(model, X_test, all_feature_cols, config)

    # --- 7. Feature importance ---
    _save_feature_importance(model, all_feature_cols, config)

    # Final summary panel
    stage_time = time.perf_counter() - stage_start
    _console.print(
        Panel(
            f"[bold green]Stage 4 complete[/]\n"
            f"  Accuracy: [bold]{acc:.4f}[/]\n"
            f"  GRU: {hidden_size} features ({config.gru.num_layers} layers)\n"
            f"  LightGBM: {len(all_feature_cols)} features, best_iter={getattr(model, 'best_iteration_', 'N/A')}\n"
            f"  Time: {stage_time:.1f}s",
            style="green",
            padding=(0, 2),
        )
    )
