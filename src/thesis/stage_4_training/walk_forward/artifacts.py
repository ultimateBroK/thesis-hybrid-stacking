"""Persist OOF, history, feature importance."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from thesis.shared.config import Config
from thesis.stage_4_training.validation import WalkForwardWindow
from thesis.stage_4_training.walk_forward.diagnostics import _shannon_entropy
from thesis.stage_4_training.walk_forward.predictions import (
    _add_confidence_columns,
    _validate_predictions,
    _write_prediction_manifest,
)

logger = logging.getLogger("thesis")


# OOF


def _save_oof_predictions(
    config: Config,
    *,
    all_oof_preds: list[pl.DataFrame],
    window_diagnostics: list[dict[str, Any]],
) -> pl.DataFrame:
    """Write validated OOF predictions."""
    if not all_oof_preds:
        raise RuntimeError("No OOF predictions generated")

    oof_df = _add_confidence_columns(pl.concat(all_oof_preds))
    preds_path = Path(config.paths.predictions)
    preds_path.parent.mkdir(parents=True, exist_ok=True)

    _validate_predictions(oof_df, preds_path)
    oof_df.write_csv(preds_path)
    _write_prediction_manifest(
        oof_df, preds_path, windows_count=len(window_diagnostics)
    )

    return oof_df


# Training history


def _save_training_history(config: Config, payload: dict[str, Any]) -> None:
    """Write training history when session exists."""
    if not config.paths.session_dir:
        return
    path = Path(config.paths.session_dir) / "models" / "training_history.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


# Walk-forward history


def _aggregate_oof_summary(diags: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate OOF distribution and guardrails."""
    total_long = total_short = total_hold = 0
    entropies, sample_entropies = [], []
    flagged = []

    for d in diags:
        pc = d.get("prediction_counts", {})
        total_long += pc.get("1", 0)
        total_short += pc.get("-1", 0)
        total_hold += pc.get("0", 0)
        if d.get("prediction_entropy") is not None:
            entropies.append(d["prediction_entropy"])
        if d.get("mean_sample_entropy") is not None:
            sample_entropies.append(d["mean_sample_entropy"])
        if d.get("ls_ratio_flagged"):
            flagged.append(d["window"])

    total = total_long + total_short + total_hold
    ls = total_long / total_short if total_short > 0 else float("inf")

    pred_probs = (
        np.array([total_long, total_hold, total_short]) / total
        if total > 0
        else np.array([])
    )

    return {
        "total_predictions": total,
        "long_count": total_long,
        "short_count": total_short,
        "hold_count": total_hold,
        "long_pct": round(total_long / total * 100, 2) if total else None,
        "short_pct": round(total_short / total * 100, 2) if total else None,
        "hold_pct": round(total_hold / total * 100, 2) if total else None,
        "aggregate_ls_ratio": round(ls, 4) if total_short > 0 else None,
        "aggregate_prediction_entropy": (
            round(_shannon_entropy(pred_probs), 4) if pred_probs.size else None
        ),
        "mean_prediction_entropy": round(float(np.mean(entropies)), 4)
        if entropies
        else None,
        "mean_sample_entropy": round(float(np.mean(sample_entropies)), 4)
        if sample_entropies
        else None,
        "ls_ratio_flagged_windows": flagged,
        "ls_ratio_flagged_count": len(flagged),
    }


def _build_wf_history(
    windows: list[WalkForwardWindow],
    diags: list[dict[str, Any]],
    oof_len: int,
    architecture: str | None = None,
) -> dict[str, Any]:
    """Build per-window history payload."""
    return {
        "num_windows": len(windows),
        "total_oof_predictions": oof_len,
        "aggregate_oof_summary": _aggregate_oof_summary(diags),
        "architecture": architecture,
        "window_details": [
            {
                "window": i + 1,
                "train_start_idx": w.train_start_idx,
                "train_end_idx": w.train_end_idx,
                "test_start_idx": w.test_start_idx,
                "test_end_idx": w.test_end_idx,
                **(next((d for d in diags if d.get("window") == i + 1), {})),
            }
            for i, w in enumerate(windows)
        ],
    }


def _save_walk_forward_history(
    config: Config,
    *,
    windows: list[WalkForwardWindow],
    window_diagnostics: list[dict[str, Any]],
    oof_len: int,
    architecture: str | None = None,
) -> None:
    """Write walk-forward history when session exists."""
    if not config.paths.session_dir:
        return
    path = Path(config.paths.session_dir) / "reports" / "walk_forward_history.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            _build_wf_history(windows, window_diagnostics, oof_len, architecture),
            f,
            indent=2,
        )


# Per-architecture copy


def _save_arch_copy(oof_df: pl.DataFrame, arch_name: str, config: Config) -> None:
    """Save architecture OOF copy."""
    if not config.paths.session_dir:
        return
    path = Path(config.paths.session_dir) / "predictions" / f"preds_{arch_name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    oof_df.write_csv(path)


# Feature importance


def _save_feature_importance(
    model: Any, feature_cols: list[str], config: Config
) -> None:
    """Save sorted feature importance JSON."""
    try:
        imp = model.feature_importances_
        pairs = sorted(zip(feature_cols, imp), key=lambda x: x[1], reverse=True)
        out_path = (
            Path(config.paths.session_dir) / "reports" / "feature_importance.json"
            if config.paths.session_dir
            else Path("results/feature_importance.json")
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({name: float(val) for name, val in pairs}, f, indent=2)
        logger.info("  Feature importance saved (top 5: %s)", [p[0] for p in pairs[:5]])
    except (OSError, ValueError) as e:
        logger.warning("Feature importance save failed: %s", e)
