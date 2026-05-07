"""GRU-only walk-forward training loop.

Runs the sequence model as a standalone classifier while preserving the same
OOF prediction and reporting contract used by static and hybrid workflows.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np
import polars as pl

from thesis.shared.config import Config
from thesis.shared.ui import console
from thesis.stage_4_training.gru import (
    predict_gru_proba,
    prepare_sequences,
    save_gru_model,
    train_gru,
)
from thesis.stage_4_training.walk_forward.artifacts import _build_wf_history
from thesis.stage_4_training.walk_forward.hybrid import _prepare_wf_data
from thesis.stage_4_training.walk_forward.utils import (
    _CLASS_ORDER,
    _add_confidence_columns,
    _add_prediction_diagnostics,
    _probability_columns,
    _validate_predictions,
    _window_diagnostics,
    _write_prediction_manifest,
)

logger = logging.getLogger("thesis.pipeline")

_VALIDATION_SPLIT_FRACTION = 0.2


def _run_gru_only_window(
    config: Config,
    w_idx: int,
    window: Any,
    df: pl.DataFrame,
) -> dict[str, Any] | None:
    """Train and evaluate one GRU-only walk-forward window."""
    if config.gru.objective != "multiclass":
        raise ValueError(
            "GRU-only walk-forward currently supports gru.objective='multiclass' "
            f"(got {config.gru.objective!r})."
        )

    train_df = df.slice(
        window.train_start_idx, window.train_end_idx - window.train_start_idx
    )
    test_df = df.slice(
        window.test_start_idx, window.test_end_idx - window.test_start_idx
    )
    seq_len = config.gru.sequence_length
    if len(train_df) < seq_len or len(test_df) < seq_len:
        logger.warning(
            "GRU-only window %d too small; train=%d test=%d seq_len=%d",
            w_idx + 1,
            len(train_df),
            len(test_df),
            seq_len,
        )
        return None

    val_split = max(1, int(len(train_df) * _VALIDATION_SPLIT_FRACTION))
    gru_train_df = train_df.head(len(train_df) - val_split)
    gru_val_df = train_df.tail(val_split)
    model, classifier, _, _, history, mean, std, gru_cols = train_gru(
        config, gru_train_df, gru_val_df, window_index=w_idx
    )

    train_seq, y_train, _ = prepare_sequences(train_df, gru_cols, seq_len)
    test_seq, y_test, _ = prepare_sequences(test_df, gru_cols, seq_len)
    if y_train is None or y_test is None:
        raise ValueError("GRU-only requires a label column in walk-forward data.")

    train_aligned = train_df.slice(seq_len - 1, len(train_seq))
    test_aligned = test_df.slice(seq_len - 1, len(test_seq))
    if train_aligned.is_empty() or test_aligned.is_empty():
        logger.warning("GRU-only window %d aligned data empty; skipping", w_idx + 1)
        return None

    proba = predict_gru_proba(
        model,
        classifier,
        test_seq,
        batch_size=config.gru.batch_size,
        mean=mean,
        std=std,
    )
    preds = _CLASS_ORDER[np.argmax(proba, axis=1)].astype(np.int32)
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)

    diag = _window_diagnostics(w_idx + 1, train_aligned, test_aligned, y_train, y_test)
    diag["class_weights"] = None
    diag["shift_weights_per_class"] = None
    _add_prediction_diagnostics(diag, preds, y_test, proba)
    accuracy = float((preds == y_test).mean())
    logger.info(
        "GRU-only window %d: accuracy=%.4f, test_samples=%d",
        w_idx + 1,
        accuracy,
        len(y_test),
    )

    oof_chunk = pl.DataFrame(
        {
            "timestamp": test_aligned["timestamp"],
            "true_label": y_test,
            "pred_label": preds,
            **_probability_columns(proba, _CLASS_ORDER),
        }
    )
    return {
        "oof_chunk": oof_chunk,
        "model": model,
        "classifier": classifier,
        "mean": mean,
        "std": std,
        "history": history,
        "accuracy": accuracy,
        "diag": diag,
    }


def _save_gru_only_artifacts(
    config: Config,
    all_oof_preds: list[pl.DataFrame],
    last_model: Any,
    last_classifier: Any,
    last_mean: Any,
    last_std: Any,
    last_history: list[dict],
    last_window_accuracy: float | None,
    last_window_index: int,
    windows: list[Any],
    window_diagnostics: list[dict[str, Any]],
    stage_start: float,
) -> None:
    """Validate and persist GRU-only walk-forward artifacts."""
    if not all_oof_preds or last_model is None or last_classifier is None:
        raise RuntimeError("No GRU-only OOF predictions generated")

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

    if config.paths.session_dir:
        session_dir = Path(config.paths.session_dir)
        models_dir = session_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        save_gru_model(
            last_model,
            config,
            models_dir / "gru_model.pt",
            mean=last_mean,
            std=last_std,
            classifier=last_classifier,
        )

        per_window_accuracies = {
            str(d.get("window")): d.get("accuracy") for d in window_diagnostics
        }
        history_path = models_dir / "training_history.json"
        with history_path.open("w") as f:
            json.dump(
                {
                    "architecture": "gru",
                    "gru": last_history,
                    "deployment_note": (
                        f"Model saved from window {last_window_index}/{len(windows)} "
                        "(the last chronological walk-forward window). "
                        "This model has NOT seen any future data beyond its "
                        "training window."
                    ),
                    "last_window_accuracy": last_window_accuracy,
                    "per_window_accuracies": per_window_accuracies,
                },
                f,
                indent=2,
            )

        wf_path = session_dir / "reports" / "walk_forward_history.json"
        wf_path.parent.mkdir(parents=True, exist_ok=True)
        wf_history = _build_wf_history(windows, window_diagnostics, len(oof_df))
        wf_history["architecture"] = "gru"
        with wf_path.open("w") as f:
            json.dump(wf_history, f, indent=2)

    logger.info(
        "GRU-only walk-forward complete: %d windows, %d OOF predictions (%.1fs)",
        len(windows),
        len(oof_df),
        time.perf_counter() - stage_start,
    )


def _run_walk_forward_gru_only(config: Config) -> None:
    """Execute standalone GRU walk-forward training across all windows."""
    df, windows, _, _ = _prepare_wf_data(config)

    all_oof_preds: list[pl.DataFrame] = []
    last_model = None
    last_classifier = None
    last_mean = None
    last_std = None
    last_history: list[dict] = []
    last_window_accuracy: float | None = None
    last_window_index = 0
    window_diagnostics: list[dict[str, Any]] = []
    stage_start = time.perf_counter()

    for w_idx, window in enumerate(windows):
        window_start = time.perf_counter()
        console.rule(
            f"[bold cyan]GRU-only window {w_idx + 1}/{len(windows)}[/]",
            style="cyan",
        )
        logger.info(
            "=== GRU-only window %d/%d: train=[%d:%d] test=[%d:%d] ===",
            w_idx + 1,
            len(windows),
            window.train_start_idx,
            window.train_end_idx,
            window.test_start_idx,
            window.test_end_idx,
        )

        result = _run_gru_only_window(config, w_idx, window, df)
        if result is None:
            continue

        all_oof_preds.append(result["oof_chunk"])
        window_diagnostics.append(result["diag"])
        last_model = result["model"]
        last_classifier = result["classifier"]
        last_mean = result["mean"]
        last_std = result["std"]
        last_history = result["history"]
        last_window_accuracy = result["accuracy"]
        last_window_index = w_idx + 1

        logger.info(
            "GRU-only window %d done (%.1fs)",
            w_idx + 1,
            time.perf_counter() - window_start,
        )

    _save_gru_only_artifacts(
        config,
        all_oof_preds,
        last_model,
        last_classifier,
        last_mean,
        last_std,
        last_history,
        last_window_accuracy,
        last_window_index,
        windows,
        window_diagnostics,
        stage_start,
    )
