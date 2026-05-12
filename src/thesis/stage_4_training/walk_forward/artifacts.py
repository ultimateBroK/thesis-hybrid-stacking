"""Walk-forward artifact persistence helpers.

Shared by LightGBM-only and stacking walk-forward workflows.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time
from typing import Any

import polars as pl

from thesis.shared.config import Config
from thesis.stage_4_training.walk_forward.predictions import (
    _add_confidence_columns,
    _validate_predictions,
    _write_prediction_manifest,
)

logger = logging.getLogger("thesis.pipeline")


def _build_lgbm_info(
    last_lgbm_model: Any,
    last_feature_cols: list[str],
    last_window_accuracy: float | None,
    window_index: int | None = None,
    total_windows: int = 0,
    window_train_dates: dict[str, str] | None = None,
    window_test_dates: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build LightGBM metadata dict for training history JSON."""
    info: dict[str, Any] = {
        "artifact_strategy": "last_walk_forward_window",
        "validation_protocol": {
            "outer_windows": "bar_based_walk_forward_with_purge_embargo",
            "lgbm_validation": "tail_20_percent_of_outer_train",
        },
        "last_window_accuracy": last_window_accuracy,
        "best_iteration": int(last_lgbm_model.best_iteration_)
        if hasattr(last_lgbm_model, "best_iteration_")
        else None,
        "n_features": len(last_feature_cols),
        "n_classes": len(last_lgbm_model.classes_)
        if hasattr(last_lgbm_model, "classes_")
        else None,
    }
    if window_index is not None:
        info["window_index"] = window_index
        info["total_windows"] = total_windows
        info["window_train_date_range"] = window_train_dates or {}
        info["window_test_date_range"] = window_test_dates or {}
        info["window_oof_accuracy"] = last_window_accuracy
    return info


def _build_wf_history(
    windows: list,
    window_diagnostics: list[dict[str, Any]],
    oof_len: int,
) -> dict[str, Any]:
    """Build walk-forward history dict with per-window details."""
    return {
        "num_windows": len(windows),
        "total_oof_predictions": oof_len,
        "window_details": [
            {
                "window": i + 1,
                "train_start_idx": w.train_start_idx,
                "train_end_idx": w.train_end_idx,
                "test_start_idx": w.test_start_idx,
                "test_end_idx": w.test_end_idx,
                **next(
                    (item for item in window_diagnostics if item["window"] == i + 1),
                    {},
                ),
            }
            for i, w in enumerate(windows)
        ],
    }


def _save_oof_predictions(
    config: Config,
    *,
    all_oof_preds: list[pl.DataFrame],
    window_diagnostics: list[dict[str, Any]],
) -> pl.DataFrame:
    """Persist concatenated OOF predictions + manifest; returns OOF dataframe."""
    if not all_oof_preds:
        raise RuntimeError(
            "No OOF predictions generated — all walk-forward windows were skipped"
        )

    oof_df = _add_confidence_columns(pl.concat(all_oof_preds))
    preds_path = Path(config.paths.predictions)
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    _validate_predictions(oof_df, preds_path)
    oof_df.write_parquet(preds_path)
    oof_df.write_csv(preds_path.with_suffix(".csv"))
    _write_prediction_manifest(
        oof_df,
        preds_path,
        windows_count=len(window_diagnostics),
    )
    return oof_df


def _save_training_history(config: Config, payload: dict[str, Any]) -> None:
    """Write ``models/training_history.json`` under the session dir if enabled."""
    if not config.paths.session_dir:
        return
    models_dir = Path(config.paths.session_dir) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    history_path = models_dir / "training_history.json"
    with history_path.open("w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Training history saved to %s", history_path)


def _save_walk_forward_history(
    config: Config,
    *,
    windows: list,
    window_diagnostics: list[dict[str, Any]],
    oof_len: int,
    architecture: str | None = None,
) -> None:
    """Write ``reports/walk_forward_history.json`` under the session dir if enabled."""
    if not config.paths.session_dir:
        return
    wf_path = Path(config.paths.session_dir) / "reports" / "walk_forward_history.json"
    wf_path.parent.mkdir(parents=True, exist_ok=True)

    wf_history = _build_wf_history(windows, window_diagnostics, oof_len)
    if architecture is not None:
        wf_history["architecture"] = architecture
    with wf_path.open("w") as f:
        json.dump(wf_history, f, indent=2)


def _log_walk_forward_complete(
    *,
    arch_name: str,
    windows_count: int,
    oof_len: int,
    stage_start: float,
    prefix: str = "Walk-forward complete",
) -> None:
    logger.info(
        "%s (%s): %d windows, %d OOF predictions (%.1fs)",
        prefix,
        arch_name,
        windows_count,
        oof_len,
        time.perf_counter() - stage_start,
    )


def _save_arch_copy(oof_df: pl.DataFrame, arch_name: str, config: Config) -> None:
    """Save per-architecture prediction copy for multi-arch comparison."""
    if not config.paths.session_dir:
        return
    session_dir = Path(config.paths.session_dir)
    preds_dir = session_dir / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)
    arch_path = preds_dir / f"preds_{arch_name}.parquet"
    oof_df.write_parquet(arch_path)
    logger.info("Per-arch predictions saved: %s", arch_path)
