"""Model evaluation: prediction helpers, diagnostics, artifact persistence.

Merged from:
  stage_4_training/walk_forward/predictions.py
  stage_4_training/walk_forward/diagnostics.py
  stage_4_training/walk_forward/artifacts.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import precision_recall_fscore_support

from thesis.shared.config import Config

logger = logging.getLogger("thesis")

# ── constants ──────────────────────────────────────────────────────────

_CLASS_ORDER = np.array([-1, 0, 1], dtype=np.int32)


# ── predictions.py ─────────────────────────────────────────────────────


def _apply_confidence_threshold(proba: np.ndarray, threshold: float) -> np.ndarray:
    """Gate LONG/SHORT by confidence.

    When threshold > 0:
        LONG  (1) if P(LONG)  - P(SHORT) > threshold
        SHORT (-1) if P(SHORT) - P(LONG)  > threshold
        HOLD  (0) otherwise
    When threshold == 0: standard argmax.
    """
    if threshold <= 0:
        return _CLASS_ORDER[np.argmax(proba, axis=1)]

    diff = proba[:, 2] - proba[:, 0]  # Long edge over short
    return np.where(diff > threshold, 1, np.where(diff < -threshold, -1, 0)).astype(
        np.int32
    )


def _align_proba(proba: np.ndarray, class_order: list[int] | np.ndarray) -> np.ndarray:
    """Align probabilities to [-1, 0, 1]."""
    aligned = np.zeros((len(proba), 3), dtype=np.float64)
    index_map = {int(c): i for i, c in enumerate(class_order)}
    for target_idx, cls in enumerate(_CLASS_ORDER):
        src = index_map.get(int(cls))
        if src is not None:
            aligned[:, target_idx] = proba[:, src]
    return aligned


def proba_columns(
    proba: np.ndarray,
    class_order: list[int] | np.ndarray,
) -> dict[str, np.ndarray]:
    """Probability columns in canonical order."""
    aligned = _align_proba(proba, class_order)
    return {
        f"pred_proba_class_{'minus' + str(abs(c)) if c < 0 else str(c)}": aligned[:, i]
        for i, c in enumerate(_CLASS_ORDER)
    }


def _label_suffix(cls: int) -> str:
    """Probability column suffix."""
    return f"minus{abs(cls)}" if cls < 0 else str(cls)


def one_hot_proba(
    preds: np.ndarray,
    *,
    prefix: str = "pred_proba_class_",
) -> dict[str, np.ndarray]:
    """One-hot probabilities from labels."""
    preds = np.asarray(preds, dtype=np.int32)
    return {
        f"{prefix}{_label_suffix(int(c))}": (preds == c).astype(np.float64)
        for c in _CLASS_ORDER
    }


def _validate_predictions(df: pl.DataFrame, path: Path) -> None:
    """Validate OOF before write.

    Catch missing columns, nulls, duplicate time, bad labels.
    """
    required = {"timestamp", "pred_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Predictions missing columns {sorted(missing)}: {path}")
    if df.is_empty():
        raise ValueError(f"Predictions are empty: {path}")

    ts = df["timestamp"]
    if ts.null_count() > 0:
        raise ValueError(f"Timestamp has nulls ({ts.null_count()}): {path}")
    if ts.n_unique() < len(ts):
        raise ValueError(f"OOF predictions contain duplicate timestamps: {path}")
    if ts.to_list() != sorted(ts.to_list()):
        raise ValueError(f"OOF predictions not sorted by timestamp: {path}")

    pred = df["pred_label"]
    if pred.null_count() > 0:
        raise ValueError(f"pred_label has nulls: {path}")
    invalid = sorted(set(pred.unique().to_list()) - {-1, 0, 1})
    if invalid:
        raise ValueError(
            f"Invalid pred_label values: expected {{-1,0,1}}, got {invalid}: {path}"
        )

    null_cols = {c: df[c].null_count() for c in df.columns if df[c].null_count()}
    if null_cols:
        raise ValueError(f"Predictions contain nulls: {null_cols}: {path}")


_PROBA_COLS = ("pred_proba_class_minus1", "pred_proba_class_0", "pred_proba_class_1")


def _add_confidence_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Add confidence score and bin."""
    if not all(c in df.columns for c in _PROBA_COLS):
        return df
    return df.with_columns(
        pl.max_horizontal([pl.col(c) for c in _PROBA_COLS]).alias("max_confidence"),
    ).with_columns(
        pl.when(pl.col("max_confidence") >= 0.6)
        .then(pl.lit("high"))
        .when(pl.col("max_confidence") >= 0.4)
        .then(pl.lit("medium"))
        .otherwise(pl.lit("low"))
        .alias("confidence_bin"),
    )


def _write_prediction_manifest(
    df: pl.DataFrame,
    path: Path,
    *,
    windows_count: int,
) -> None:
    """Write prediction manifest beside CSV."""
    manifest = {
        "row_count": len(df),
        "start": str(df["timestamp"][0]),
        "end": str(df["timestamp"][-1]),
        "label_distribution": (
            _counts(df["true_label"].to_numpy()) if "true_label" in df.columns else {}
        ),
        "prediction_distribution": _counts(df["pred_label"].to_numpy()),
        "mean_confidence": (
            float(df["max_confidence"].mean())
            if "max_confidence" in df.columns
            else None
        ),
        "windows_count": windows_count,
    }
    with open(path.with_name("prediction_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


# ── diagnostics.py ─────────────────────────────────────────────────────


def _shannon_entropy(p: np.ndarray) -> float:
    """Base-2 entropy."""
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


def _counts(values: np.ndarray) -> dict[str, int]:
    """Class counts. String keys for JSON."""
    if values.size == 0:
        return {}
    labels, counts = np.unique(values.astype(np.int32), return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(labels, counts)}


def _pct(counts: dict[str, int]) -> dict[str, float]:
    """Counts to percentages."""
    total = sum(counts.values())
    if total == 0:
        return {}
    return {k: round(v / total * 100.0, 2) for k, v in counts.items()}


def _dates(df: pl.DataFrame) -> dict[str, str]:
    """Window start/end timestamps."""
    if df.is_empty() or "timestamp" not in df.columns:
        return {"start": "", "end": ""}
    return {"start": str(df["timestamp"][0]), "end": str(df["timestamp"][-1])}


def _window_diagnostics(
    w_idx: int,
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Build label diagnostics for one window."""
    train_c = _counts(y_train)
    test_c = _counts(y_test)
    diag: dict[str, Any] = {
        "window": w_idx,
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "train_dates": _dates(train_df),
        "test_dates": _dates(test_df),
        "train_label_counts": train_c,
        "train_label_pct": _pct(train_c),
        "test_label_counts": test_c,
        "test_label_pct": _pct(test_c),
    }
    logger.info(
        "  Window %d labels | train=%s | test=%s",
        w_idx,
        diag["train_label_pct"],
        diag["test_label_pct"],
    )
    return diag


def _per_class_metrics(
    preds: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Per-class classification metrics."""
    classes = np.array([-1, 0, 1], dtype=np.int32)
    p, r, f1, s = precision_recall_fscore_support(
        y_test, preds, labels=classes, zero_division=0
    )
    return {
        str(int(cls)): {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(s[i]),
        }
        for i, cls in enumerate(classes)
    }


def _add_prediction_diagnostics(
    diag: dict[str, Any],
    preds: np.ndarray,
    y_test: np.ndarray,
    proba: np.ndarray,
    *,
    confidence_threshold: float = 0.0,
) -> None:
    """Attach prediction diagnostics to window."""
    pred_c = _counts(preds)
    confidence = np.max(proba, axis=1) if len(proba) else np.array([])
    n_total = sum(pred_c.values())
    n_long = pred_c.get("1", 0)
    n_short = pred_c.get("-1", 0)
    n_hold = pred_c.get("0", 0)

    # Prediction distribution entropy
    if n_total > 0:
        pred_probs = np.array([pred_c[k] / n_total for k in sorted(pred_c)])
        pred_entropy = round(_shannon_entropy(pred_probs), 4)
    else:
        pred_entropy = None

    # Mean sample uncertainty
    if len(proba):
        sample_entropies = np.array(
            [_shannon_entropy(proba[i]) for i in range(len(proba))]
        )
        mean_sample_entropy = round(float(sample_entropies.mean()), 4)
    else:
        mean_sample_entropy = None

    # Flag extreme long/short imbalance
    ls_ratio = n_long / n_short if n_short > 0 else float("inf")
    ls_flagged = False
    if n_short > 0 and n_long > 0:
        ls_flagged = ls_ratio < 0.2 or ls_ratio > 5.0

    diag.update(
        {
            "prediction_counts": pred_c,
            "prediction_pct": _pct(pred_c),
            "accuracy": float((preds == y_test).mean()) if len(y_test) else None,
            "mean_confidence": float(confidence.mean()) if len(confidence) else None,
            "high_conf_70_pct": (
                float((confidence >= 0.70).mean() * 100.0) if len(confidence) else None
            ),
            "ls_ratio": round(ls_ratio, 4) if n_short > 0 else None,
            "ls_ratio_flagged": ls_flagged,
            "prediction_entropy": pred_entropy,
            "mean_sample_entropy": mean_sample_entropy,
            "confidence_threshold": confidence_threshold,
            "hold_count": n_hold,
            "hold_pct": round(n_hold / n_total * 100.0, 2) if n_total else None,
            "per_class": _per_class_metrics(preds, y_test) if len(y_test) else {},
        }
    )

    logger.info(
        "  Window %d preds | %s | acc=%.4f conf=%.3f L/S=%.3f hold=%d/%d (%.1f%%)",
        diag["window"],
        diag["prediction_pct"],
        diag["accuracy"] or 0.0,
        diag["mean_confidence"] or 0.0,
        ls_ratio if n_short > 0 else float("nan"),
        n_hold,
        n_total,
        diag["hold_pct"] or 0.0,
    )

    # Guardrail warnings
    if n_short == 0 and n_long > 0:
        logger.warning("  Window %d: No SHORT predictions", diag["window"])
    elif n_long == 0 and n_short > 0:
        logger.warning("  Window %d: No LONG predictions", diag["window"])
    elif ls_flagged:
        logger.warning(
            "  Window %d: L/S ratio %.2f outside [0.2, 5.0]",
            diag["window"],
            ls_ratio,
        )


# ── artifacts.py ───────────────────────────────────────────────────────


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


def _save_training_history(config: Config, payload: dict[str, Any]) -> None:
    """Write training history when session exists."""
    if not config.paths.session_dir:
        return
    path = Path(config.paths.session_dir) / "models" / "training_history.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


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
    windows: list[Any],
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
    windows: list[Any],
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


def _save_arch_copy(oof_df: pl.DataFrame, arch_name: str, config: Config) -> None:
    """Save architecture OOF copy."""
    if not config.paths.session_dir:
        return
    path = Path(config.paths.session_dir) / "predictions" / f"preds_{arch_name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    oof_df.write_csv(path)


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
