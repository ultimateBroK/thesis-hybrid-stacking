"""LightGBM-only training — static train_model orchestrator."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time

import joblib
import numpy as np
import polars as pl

from thesis.shared.config import Config
from thesis.shared.constants import EXCLUDE_COLS
from thesis.shared.ui import console
from thesis.stage_4_training.lgbm.utils import (
    _compute_class_weights,
    _save_feature_importance,
    _train_fixed,
)

logger = logging.getLogger("thesis.model")


def _normalize_label(lbl: int) -> str:
    if lbl < 0:
        return f"minus{abs(lbl)}"
    return str(lbl)


def _save_predictions(
    test_aligned: pl.DataFrame,
    y_test: np.ndarray,
    preds: np.ndarray,
    proba: np.ndarray,
    class_order: list,
    preds_path: Path,
) -> None:
    proba_cols = {
        f"pred_proba_class_{_normalize_label(cls)}": proba[:, idx]
        for idx, cls in enumerate(class_order)
    }
    preds_df = pl.DataFrame(
        {
            "timestamp": test_aligned["timestamp"],
            "true_label": y_test,
            "pred_label": preds.astype(np.int32),
            **proba_cols,
        }
    )
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    preds_df.write_parquet(preds_path)
    csv_path = preds_path.with_suffix(".csv")
    preds_df.write_csv(csv_path)


def train_model(config: Config) -> None:
    """Train and evaluate the LightGBM model on tabular features.

    Args:
        config: Resolved application configuration.

    Raises:
        FileNotFoundError: If required split parquet files are missing.
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

    with console.status("[cyan]Loading train/val/test splits[/]"):
        train_df = pl.read_parquet(train_path)
        val_df = pl.read_parquet(val_path)
        test_df = pl.read_parquet(test_path)

    logger.info(
        "Splits: train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df)
    )

    static_cols = [c for c in train_df.columns if c not in EXCLUDE_COLS]
    logger.info("Static features: %d", len(static_cols))

    X_train = train_df.select(static_cols).to_pandas()
    X_val = val_df.select(static_cols).to_pandas()
    X_test = test_df.select(static_cols).to_pandas()

    y_train = train_df["label"].to_numpy().astype(np.int32)
    y_val = val_df["label"].to_numpy().astype(np.int32)
    y_test = test_df["label"].to_numpy().astype(np.int32)
    train_weights = (
        train_df["sample_weight"].to_numpy().astype(np.float64)
        if "sample_weight" in train_df.columns
        else None
    )

    class_weights = _compute_class_weights(y_train)
    model = _train_fixed(
        X_train,
        y_train,
        X_val,
        y_val,
        class_weights,
        config,
        static_cols,
        sample_weight=train_weights,
    )

    model_path = Path(config.paths.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info("Saved model: %s", model_path)

    is_regression = config.model.objective == "regression"

    models_dir = model_path.parent
    history_path = models_dir / "training_history.json"
    lgbm_info: dict = {
        "best_iteration": int(model.best_iteration_)
        if hasattr(model, "best_iteration_")
        else None,
        "n_features": len(static_cols),
        "objective": config.model.objective,
        "n_classes": int(model.n_classes_) if hasattr(model, "n_classes_") else None,
    }
    training_history = {
        "lightgbm": lgbm_info,
    }
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2)

    logger.info("Stage 4.2: Predictions & Evaluation")

    if is_regression:
        raw_preds = model.predict(X_test)
        preds = np.where(raw_preds > 0, 1, np.where(raw_preds < 0, -1, 0))
        proba = None
    else:
        proba = model.predict_proba(X_test)
        preds = model.classes_[np.argmax(proba, axis=1)]

    acc = (preds == y_test).mean()

    label_map = {-1: "SELL", 0: "HOLD", 1: "BUY"}
    for cls in [-1, 0, 1]:
        mask = y_test == cls
        if mask.sum() > 0:
            cls_acc = (preds[mask] == cls).mean()
            logger.info(
                "Class %s (%d): samples=%d accuracy=%.3f predicted=%d",
                label_map[cls],
                cls,
                int(mask.sum()),
                float(cls_acc),
                int((preds == cls).sum()),
            )
    logger.info("Test accuracy: %.4f", acc)

    if is_regression:
        preds_path = Path(config.paths.predictions)
        preds_path.parent.mkdir(parents=True, exist_ok=True)
        preds_df = pl.DataFrame(
            {
                "timestamp": test_df["timestamp"],
                "true_label": y_test,
                "pred_label": preds.astype(np.int32),
                "pred_raw": raw_preds.astype(np.float64),
            }
        )
        preds_df.write_parquet(preds_path)
        csv_path = preds_path.with_suffix(".csv")
        preds_df.write_csv(csv_path)
    else:
        class_order = model.classes_.tolist()
        preds_path = Path(config.paths.predictions)
        _save_predictions(test_df, y_test, preds, proba, class_order, preds_path)

    _save_feature_importance(model, static_cols, config)

    stage_time = time.perf_counter() - stage_start
    logger.info(
        "Stage 4 complete | accuracy=%.4f lgbm_features=%d best_iter=%s time=%.1fs",
        acc,
        len(static_cols),
        getattr(model, "best_iteration_", "N/A"),
        stage_time,
    )
