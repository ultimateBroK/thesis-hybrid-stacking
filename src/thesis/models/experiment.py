"""Walk-forward model experiment for Stage 3."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import numpy as np
import polars as pl

from thesis.models.artifacts import proba_columns
from thesis.models.baselines import compute_metrics, predict_majority_label
from thesis.models.estimators import (
    build_base_models,
    fit_model,
    get_class_order,
    predict_proba_aligned,
)
from thesis.models.stacking import HybridStackingClassifier
from thesis.models.validation import WalkForwardWindow
from thesis.shared.config import Config

logger = logging.getLogger("thesis")


@dataclass(frozen=True)
class WindowDataset:
    """Feature/label/weight arrays for one walk-forward window."""

    train_x: np.ndarray
    train_y: np.ndarray
    train_w: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    test_timestamps: np.ndarray


@dataclass(frozen=True)
class ModelPrediction:
    """Predicted labels and probabilities from one model."""

    model_name: str
    pred_label: np.ndarray
    pred_proba: np.ndarray


@dataclass(frozen=True)
class WindowResult:
    """Training and evaluation output for one window."""

    window_index: int
    predictions: list[ModelPrediction]
    metrics: dict[str, dict[str, float]]
    final_model: Any
    final_lightgbm_model: Any | None
    train_rows: int
    test_rows: int
    test_timestamps: np.ndarray
    test_y: np.ndarray


@dataclass(frozen=True)
class ModelExperiment:
    """Aggregated Stage 3 experiment output."""

    feature_cols: list[str]
    window_results: list[WindowResult]
    model_comparison: dict[str, dict[str, Any]]
    oof_predictions: pl.DataFrame
    final_model: Any
    final_lightgbm_model: Any | None
    training_history: dict[str, Any]
    walk_forward_history: dict[str, Any]


def _model_display_name(name: str) -> str:
    """Human-readable label for reports and charts (snake_case → Title Case)."""
    return {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "lightgbm": "LightGBM",
        "hybrid_stacking": "Hybrid Stacking",
    }.get(name, name)


def slice_window_dataset(
    dataset: pl.DataFrame,
    feature_cols: list[str],
    window: WalkForwardWindow,
) -> WindowDataset:
    """Extract train/test arrays with sample weights for one window."""
    train_df = dataset.slice(window.train_start_idx, window.train_len)
    test_df = dataset.slice(window.test_start_idx, window.test_len)
    w_col = "sample_weight"
    train_w = (
        train_df[w_col].to_numpy().astype(np.float64)
        if w_col in train_df.columns
        else np.ones(len(train_df), dtype=np.float64)
    )
    return WindowDataset(
        train_x=train_df.select(feature_cols).to_numpy(),
        train_y=train_df["label"].to_numpy().astype(np.int32),
        train_w=train_w,
        test_x=test_df.select(feature_cols).to_numpy(),
        test_y=test_df["label"].to_numpy().astype(np.int32),
        test_timestamps=test_df["timestamp"].to_numpy(),
    )


def train_models_for_window(
    data: WindowDataset,
    feature_cols: list[str],
    config: Config,
) -> dict[str, Any]:
    """Fit LR, RF, LightGBM, and Hybrid Stacking with sample weights."""
    models = {
        name: fit_model(model, data.train_x, data.train_y, feature_cols, data.train_w)
        for name, model in build_base_models(
            config, num_classes=config.labels.num_classes
        ).items()
    }
    models["hybrid_stacking"] = HybridStackingClassifier(config, feature_cols).fit(
        data.train_x,
        data.train_y,
        sample_weight=data.train_w,
    )
    return models


def predict_models_for_window(
    models: dict[str, Any],
    data: WindowDataset,
    feature_cols: list[str],
    config: Config,
) -> list[ModelPrediction]:
    """Predict all trained models on the window test slice."""
    class_order = get_class_order(config.labels.num_classes)
    predictions: list[ModelPrediction] = []
    for name, model in models.items():
        proba = predict_proba_aligned(
            model, data.test_x, feature_cols, target_order=class_order
        )
        predictions.append(
            ModelPrediction(
                model_name=name,
                pred_label=class_order[np.argmax(proba, axis=1)].astype(np.int32),
                pred_proba=proba,
            )
        )
    return predictions


def evaluate_window_predictions(
    predictions: list[ModelPrediction],
    y_true: np.ndarray,
    train_y: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Compute core classification metrics for each model plus majority baseline."""
    metrics = {p.model_name: compute_metrics(y_true, p.pred_label) for p in predictions}
    baseline_pred, maj_label = predict_majority_label(train_y, len(y_true))
    metrics["majority_baseline"] = compute_metrics(y_true, baseline_pred)
    metrics["majority_baseline"]["majority_class_label"] = maj_label
    return metrics


def run_one_window(
    dataset: pl.DataFrame,
    feature_cols: list[str],
    window: WalkForwardWindow,
    window_index: int,
    config: Config,
) -> WindowResult:
    """Train and evaluate one walk-forward window."""
    data = slice_window_dataset(dataset, feature_cols, window)
    models = train_models_for_window(data, feature_cols, config)
    predictions = predict_models_for_window(models, data, feature_cols, config)
    hybrid_pred = next(p for p in predictions if p.model_name == "hybrid_stacking")
    logger.info(
        "Window %d: train=%d test=%d hybrid_acc=%.4f",
        window_index,
        len(data.train_y),
        len(data.test_y),
        compute_metrics(data.test_y, hybrid_pred.pred_label)["accuracy"],
    )
    return WindowResult(
        window_index=window_index,
        predictions=predictions,
        metrics=evaluate_window_predictions(predictions, data.test_y, data.train_y),
        final_model=models["hybrid_stacking"],
        final_lightgbm_model=models.get("lightgbm"),
        train_rows=len(data.train_y),
        test_rows=len(data.test_y),
        test_timestamps=data.test_timestamps,
        test_y=data.test_y,
    )


def _combine_model_metrics(results: list[WindowResult]) -> dict[str, dict[str, Any]]:
    """Compute metrics on full OOF predictions instead of per-window mean."""
    model_names = {p.model_name for r in results for p in r.predictions}
    model_names.add("majority_baseline")
    comparison: dict[str, dict[str, Any]] = {}
    for name in sorted(model_names):
        all_true: list[np.ndarray] = []
        all_pred: list[np.ndarray] = []
        for r in results:
            if name == "majority_baseline":
                maj_label = r.metrics.get("majority_baseline", {}).get(
                    "majority_class_label", 0
                )
                all_true.append(r.test_y)
                all_pred.append(np.full(len(r.test_y), maj_label, dtype=np.int32))
            else:
                pred = next((p for p in r.predictions if p.model_name == name), None)
                if pred is None:
                    continue
                all_true.append(r.test_y)
                all_pred.append(pred.pred_label)
        if all_true:
            y_true = np.concatenate(all_true)
            y_pred = np.concatenate(all_pred)
            m = compute_metrics(y_true, y_pred)
            comparison[name] = {
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "directional_accuracy": m["directional_accuracy"],
            }
    return comparison


def _build_predictions_frame(
    results: list[WindowResult],
    class_order: np.ndarray,
) -> pl.DataFrame:
    """Collect hybrid-stacking OOF predictions + probabilities across all windows."""
    chunks: list[pl.DataFrame] = []
    for result in results:
        pred = next(p for p in result.predictions if p.model_name == "hybrid_stacking")
        chunks.append(
            pl.DataFrame(
                {
                    "timestamp": result.test_timestamps,
                    "true_label": result.test_y,
                    "pred_label": pred.pred_label,
                    **proba_columns(pred.pred_proba, class_order),
                }
            )
        )
    return pl.concat(chunks).sort("timestamp")


def _build_training_history(
    results: list[WindowResult],
    feature_cols: list[str],
    config: Config,
) -> dict[str, Any]:
    """Architecture summary for the experiment manifest."""
    return {
        "architecture": "hybrid_stacking",
        "validation_protocol": "walk_forward_with_chronological_meta_split",
        "base_models": ["logistic_regression", "random_forest", "lightgbm"],
        "meta_model": config.model.stacking_meta.learner,
        "meta_fraction": config.model.stacking.meta_fraction,
        "n_features": len(feature_cols),
        "windows": len(results),
        "last_window_accuracy": results[-1].metrics["hybrid_stacking"]["accuracy"],
    }


def _build_walk_forward_history(
    windows: list[WalkForwardWindow],
    results: list[WindowResult],
) -> dict[str, Any]:
    """Per-window index ranges, row counts, and accuracy for reproducibility logs."""
    return {
        "num_windows": len(windows),
        "total_oof_predictions": sum(r.test_rows for r in results),
        "window_details": [
            {
                "window": result.window_index,
                "train_start_idx": window.train_start_idx,
                "train_end_idx": window.train_end_idx,
                "test_start_idx": window.test_start_idx,
                "test_end_idx": window.test_end_idx,
                "train_rows": result.train_rows,
                "test_rows": result.test_rows,
                "accuracy": result.metrics["hybrid_stacking"]["accuracy"],
            }
            for window, result in zip(windows, results)
        ],
    }


def run_model_experiment(
    dataset: pl.DataFrame,
    feature_cols: list[str],
    windows: list[WalkForwardWindow],
    config: Config,
) -> ModelExperiment:
    """Run full Stage 3 classification experiment."""
    results = [
        run_one_window(dataset, feature_cols, window, idx + 1, config)
        for idx, window in enumerate(windows)
    ]
    if not results:
        raise RuntimeError("No model experiment windows produced results")
    comparison = _combine_model_metrics(results)
    return ModelExperiment(
        feature_cols=feature_cols,
        window_results=results,
        model_comparison=comparison,
        oof_predictions=_build_predictions_frame(
            results, get_class_order(config.labels.num_classes)
        ),
        final_model=results[-1].final_model,
        final_lightgbm_model=results[-1].final_lightgbm_model,
        training_history=_build_training_history(results, feature_cols, config),
        walk_forward_history=_build_walk_forward_history(windows, results),
    )
