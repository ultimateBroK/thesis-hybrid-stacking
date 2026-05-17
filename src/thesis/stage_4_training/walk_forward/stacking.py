"""Stacking trainer. Base learners feed meta learner."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from thesis.shared.config import Config
from thesis.stage_4_training.lgbm.utils import _compute_class_weights, _train_lgbm
from thesis.stage_4_training.validation import WalkForwardWindow
from thesis.stage_4_training.walk_forward.artifacts import (
    _save_arch_copy,
    _save_feature_importance,
    _save_oof_predictions,
    _save_training_history,
    _save_walk_forward_history,
)
from thesis.stage_4_training.walk_forward.diagnostics import (
    _add_prediction_diagnostics,
    _window_diagnostics,
)
from thesis.stage_4_training.walk_forward.feature_pipeline import (
    fit_static_feature_pipeline,
    select_static_cols,
)
from thesis.stage_4_training.walk_forward.predictions import (
    _align_proba,
    _apply_confidence_threshold,
    proba_columns,
)

logger = logging.getLogger("thesis")

_CLASS_ORDER = np.array([-1, 0, 1], dtype=np.int32)
_BASE_ALIASES = {
    "logistic_regression": "logreg",
    "random_forest": "rf",
    "lightgbm": "lgbm",
}
_MIN_SPLIT = 4
_MIN_CAL = 50


# Base model registry


def _build_base(name: str, config: Config) -> Any:
    """Build base learner."""
    if name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=config.workflow.random_seed,
        )
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=config.model.random_forest_n_estimators,
            max_depth=config.model.random_forest_max_depth,
            min_samples_leaf=config.model.random_forest_min_samples_leaf,
            class_weight="balanced_subsample",
            random_state=config.workflow.random_seed,
            n_jobs=config.workflow.n_jobs,
        )
    raise ValueError(f"Unsupported base model: {name!r}")


def _fit_safe(model: Any, X: np.ndarray, y: np.ndarray) -> Any:
    """Fit model. Dummy protects single-class folds."""
    if len(np.unique(y)) < 2:
        from sklearn.dummy import DummyClassifier

        model = DummyClassifier(strategy="most_frequent")
    model.fit(X, y)
    return model


# Meta model registry


def _build_meta(name: str, config: Config, X: np.ndarray, y: np.ndarray) -> Any:
    """Build meta learner. Dummy protects single-class meta labels."""
    if len(np.unique(y)) < 2:
        from sklearn.dummy import DummyClassifier

        return DummyClassifier(strategy="most_frequent").fit(X, y)
    if name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=config.workflow.random_seed,
        ).fit(X, y)
    if name == "lightgbm":
        val_split = max(1, int(len(X) * 0.2))
        if len(X) <= _MIN_SPLIT:
            # Too small — disable early stopping, use conservative rounds
            import lightgbm as lgb

            return lgb.LGBMClassifier(
                objective="multiclass",
                num_class=len(np.unique(y)),
                n_estimators=50,
                class_weight="balanced",
                random_state=config.workflow.random_seed,
                n_jobs=-1,
                verbose=-1,
            ).fit(X, y)
        X_tr, y_tr = X[:-val_split], y[:-val_split]
        X_val, y_val = X[-val_split:], y[-val_split:]
        return _train_lgbm(
            X_tr,
            y_tr,
            X_val,
            y_val,
            _compute_class_weights(y_tr),
            config,
            [f"m{i}" for i in range(X.shape[1])],
        )
    raise ValueError(f"Unsupported meta model: {name!r}")


# Probability helpers


def _aligned_proba(
    model: Any, X: np.ndarray, feature_names: list[str] | None = None
) -> np.ndarray:
    """Predict probabilities in [-1, 0, 1] order."""
    X_p = X
    fitted_names = getattr(model, "feature_names_in_", None)
    if (
        fitted_names is not None
        and feature_names is not None
        and len(feature_names) == X.shape[1]
    ):
        import pandas as pd

        X_p = pd.DataFrame(X, columns=feature_names)
    proba = model.predict_proba(X_p)
    return _align_proba(proba, model.classes_)


def _stack_features(
    base_outputs: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    """Stack base probabilities into meta features."""
    if not base_outputs:
        raise ValueError("No base model outputs for stacking")
    row_counts = {n: m.shape[0] for n, m in base_outputs.items()}
    if len(set(row_counts.values())) != 1:
        raise ValueError(f"Base row counts differ: {row_counts}")

    matrices, names = [], []
    for name in sorted(base_outputs):
        mat = np.asarray(base_outputs[name], dtype=np.float64)
        matrices.append(mat)
        names.extend(f"{name}_proba_{s}" for s in ["short", "hold", "long"])
    return np.hstack(matrices), names


def _classification_summary(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    """Build model comparison metrics."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_recall_fscore_support,
    )

    if not y_true:
        return {"accuracy": None, "macro_f1": None, "per_class": {}}
    labels = [-1, 0, 1]
    p, r, f1, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "per_class": {
            str(label): {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f1[i]),
                "support": int(s[i]),
            }
            for i, label in enumerate(labels)
        },
    }


# Expanding-origin OOF


def _internal_folds(
    train_len: int, n_folds: int, purge_bars: int
) -> list[WalkForwardWindow]:
    """Build internal expanding-origin folds.

    Purge gap keeps meta features out-of-sample.
    """
    fold_size = train_len // n_folds
    boundaries = [i * fold_size for i in range(n_folds)] + [train_len]
    folds: list[WalkForwardWindow] = []
    for i in range(1, n_folds):
        pred_start = boundaries[i]
        pred_end = boundaries[i + 1]
        adj_train_end = max(0, pred_start - purge_bars)
        adj_test_start = min(pred_start + purge_bars, pred_end)
        if adj_train_end > 0 and adj_test_start < pred_end:
            folds.append(
                WalkForwardWindow(
                    train_start_idx=0,
                    train_end_idx=adj_train_end,
                    test_start_idx=adj_test_start,
                    test_end_idx=pred_end,
                )
            )
    return folds


def _expanding_origin_oof(
    config: Config,
    train_df: pl.DataFrame,
    feature_cols: list[str],
    purge_bars: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, Any, list[str], list[str], np.ndarray | None]:
    """Generate OOF meta features.

    Meta learner trains only on base out-of-sample outputs.
    """
    n_folds = config.model.stacking_internal_folds
    if n_folds < 2:
        raise ValueError("Need at least 2 folds for OOF stacking")

    internal = _internal_folds(len(train_df), n_folds, purge_bars)
    if not internal:
        raise ValueError(
            f"Expanding-origin produced 0 usable folds from {len(train_df)} rows"
        )

    y_full = train_df["label"].to_numpy().astype(np.int32)
    static_cols = select_static_cols(config, train_df, feature_cols)
    pipeline, selected = fit_static_feature_pipeline(
        config, train_df, static_cols, y_full
    )
    X_full = pipeline.transform(train_df.select(static_cols).to_pandas())

    oof_parts: dict[str, list[np.ndarray]] = {}
    y_oof_parts: list[np.ndarray] = []
    x_oof_parts: list[np.ndarray] = []

    for fi, fold in enumerate(internal):
        X_tr = X_full[fold.train_start_idx : fold.train_end_idx]
        y_tr = y_full[fold.train_start_idx : fold.train_end_idx]
        X_pr = X_full[fold.test_start_idx : fold.test_end_idx]
        y_pr = y_full[fold.test_start_idx : fold.test_end_idx]

        if len(np.unique(y_tr)) < 2:
            logger.warning("  Internal fold %d: single-class train, skipping", fi)
            continue

        for configured_name in config.model.stacking_base_models:
            short = _BASE_ALIASES.get(configured_name, configured_name)
            if configured_name == "lightgbm":
                model = _train_lgbm(
                    X_tr,
                    y_tr,
                    X_pr,
                    y_pr,
                    _compute_class_weights(y_tr),
                    config,
                    selected,
                )
            else:
                model = _fit_safe(_build_base(configured_name, config), X_tr, y_tr)
            proba = _aligned_proba(model, X_pr, selected)
            oof_parts.setdefault(short, []).append(proba)

        y_oof_parts.append(y_pr)
        x_oof_parts.append(np.asarray(X_pr))

    if not oof_parts:
        raise ValueError("Expanding-origin OOF produced no predictions")

    meta_train_outputs = {n: np.concatenate(p, axis=0) for n, p in oof_parts.items()}
    y_meta = np.concatenate(y_oof_parts, axis=0).astype(np.int32)
    X_meta_raw = np.concatenate(x_oof_parts, axis=0) if x_oof_parts else None
    logger.info(
        "  Expanding-origin OOF: %d meta rows from %d folds", len(y_meta), len(internal)
    )
    return meta_train_outputs, y_meta, pipeline, selected, static_cols, X_meta_raw


# Calibration


def _calibrate_models(
    base_models: dict[str, Any],
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    n_models: int,
) -> dict[str, Any]:
    """Calibrate base probabilities with sigmoid.

    Skip when calibration rows too few.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator

    if len(y_cal) < _MIN_CAL * n_models:
        logger.warning(
            "  Calibration set too small (%d rows for %d models) — skipping",
            len(y_cal),
            n_models,
        )
        return dict(base_models)

    calibrated = {}
    for name, model in base_models.items():
        frozen = FrozenEstimator(model)
        cal = CalibratedClassifierCV(estimator=frozen, method="sigmoid")
        cal.fit(X_cal, y_cal)
        calibrated[name] = cal
        logger.info("  %s: calibrated on %d rows", name, len(y_cal))
    return calibrated


# Train one outer window


def _train_stacking_window(
    config: Config,
    w_idx: int,
    window: WalkForwardWindow,
    df: pl.DataFrame,
    feature_cols: list[str],
    *,
    is_regression: bool = False,
) -> dict[str, Any] | None:
    """Train/predict one stacking window."""
    train_df = df.slice(window.train_start_idx, window.train_len)
    test_df = df.slice(window.test_start_idx, window.test_len)
    if train_df.is_empty() or test_df.is_empty():
        return None

    y_test = test_df["label"].to_numpy().astype(np.int32)
    n_folds = config.model.stacking_internal_folds
    do_cal = (
        config.model.stacking_calibrate_base
        and config.model.stacking_calibration_fraction > 0.0
    )
    cal_frac = config.model.stacking_calibration_fraction

    diag_extra: dict[str, Any] = {}

    # OOF path
    if n_folds >= 2:
        purge_bars = config.model.stacking_internal_purge
        if purge_bars <= 0:
            purge_bars = config.labels.horizon_bars

        if do_cal:
            # Chronological base/calibration split
            cal_rows = max(1, int(round(len(train_df) * cal_frac)))
            base_cal_df = train_df.slice(0, len(train_df) - cal_rows)
            cal_df = train_df.slice(len(train_df) - cal_rows, cal_rows)
            y_cal = cal_df["label"].to_numpy().astype(np.int32)
            logger.info(
                "Window %d: OOF (%d folds, purge=%d) + calibration (%d rows)",
                w_idx + 1,
                n_folds,
                purge_bars,
                len(y_cal),
            )
            meta_train_outputs, y_meta, pipeline, selected, static_cols, X_meta_raw = (
                _expanding_origin_oof(config, base_cal_df, feature_cols, purge_bars)
            )
            y_base = base_cal_df["label"].to_numpy().astype(np.int32)
            X_base = pipeline.transform(base_cal_df.select(static_cols).to_pandas())
            X_cal = pipeline.transform(cal_df.select(static_cols).to_pandas())
        else:
            logger.info(
                "Window %d: expanding-origin OOF (%d folds, purge=%d)",
                w_idx + 1,
                n_folds,
                purge_bars,
            )
            meta_train_outputs, y_meta, pipeline, selected, static_cols, X_meta_raw = (
                _expanding_origin_oof(config, train_df, feature_cols, purge_bars)
            )
            y_base = train_df["label"].to_numpy().astype(np.int32)
            X_base = pipeline.transform(train_df.select(static_cols).to_pandas())
            X_cal = None
            y_cal = None

        X_test = pipeline.transform(test_df.select(static_cols).to_pandas())

        # Refit bases on deployable train slice
        # Split base train for early stopping — never use X_test as eval
        _val_split = max(1, int(len(X_base) * 0.2))
        X_tr_base, y_tr_base = X_base[:-_val_split], y_base[:-_val_split]
        X_val_base, y_val_base = X_base[-_val_split:], y_base[-_val_split:]

        base_models: dict[str, Any] = {}
        for configured_name in config.model.stacking_base_models:
            short = _BASE_ALIASES.get(configured_name, configured_name)
            if configured_name == "lightgbm":
                class_weights = (
                    _compute_class_weights(y_tr_base)
                    if len(np.unique(y_tr_base)) > 1
                    else None
                )
                model = _train_lgbm(
                    X_tr_base,
                    y_tr_base,
                    X_val_base,
                    y_val_base,
                    class_weights,
                    config,
                    selected,
                )
            else:
                model = _fit_safe(_build_base(configured_name, config), X_base, y_base)
            base_models[short] = model

        if do_cal and X_cal is not None and y_cal is not None:
            base_models = _calibrate_models(
                base_models, X_cal, y_cal, len(config.model.stacking_base_models)
            )
            diag_extra["calibration"] = {
                "calibrated": True,
                "n_calibration": len(y_cal),
            }

        test_outputs = {
            name: _aligned_proba(model, X_test, selected)
            for name, model in base_models.items()
        }
        diag_extra["stacking_mode"] = "expanding_origin_oof"
        diag_extra["internal_folds"] = n_folds

    # Legacy single-holdout path
    else:
        meta_frac = config.model.stacking_meta_fraction
        base_rows = int(round(len(train_df) * (1 - meta_frac)))
        base_df, meta_df = (
            train_df.slice(0, base_rows),
            train_df.slice(base_rows, len(train_df) - base_rows),
        )
        y_base = base_df["label"].to_numpy().astype(np.int32)
        y_meta = meta_df["label"].to_numpy().astype(np.int32)

        static_cols = select_static_cols(config, train_df, feature_cols)
        pipeline, selected = fit_static_feature_pipeline(
            config, base_df, static_cols, y_base
        )

        X_base = pipeline.transform(base_df.select(static_cols).to_pandas())
        X_meta = pipeline.transform(meta_df.select(static_cols).to_pandas())
        X_test = pipeline.transform(test_df.select(static_cols).to_pandas())

        base_models = {}
        for configured_name in config.model.stacking_base_models:
            short = _BASE_ALIASES.get(configured_name, configured_name)
            model = _fit_safe(_build_base(configured_name, config), X_base, y_base)
            base_models[short] = model

        test_outputs = {
            name: _aligned_proba(model, X_test, selected)
            for name, model in base_models.items()
        }
        meta_train_outputs = {
            name: _aligned_proba(model, X_meta, selected)
            for name, model in base_models.items()
        }
        X_meta_raw = np.asarray(X_meta) if config.model.stacking_passthrough else None
        diag_extra["stacking_mode"] = "single_holdout"

    # Soft-voting baseline
    soft_vote = np.mean(list(test_outputs.values()), axis=0)
    soft_vote_preds = _CLASS_ORDER[np.argmax(soft_vote, axis=1)]

    # Meta model — optionally pass raw features through
    X_meta_stack, meta_names = _stack_features(meta_train_outputs)
    X_test_stack, _ = _stack_features(test_outputs)
    passthrough = config.model.stacking_passthrough
    if passthrough and X_meta_raw is not None:
        X_meta_stack = np.hstack([X_meta_stack, X_meta_raw])
        X_test_stack = np.hstack([X_test_stack, np.asarray(X_test)])
        raw_names = [f"raw_{c}" for c in selected]
        meta_names = meta_names + raw_names
    meta_model = _build_meta(
        config.model.stacking_meta_model, config, X_meta_stack, y_meta
    )
    final_proba = _aligned_proba(meta_model, X_test_stack, meta_names)
    threshold = config.model.prediction_confidence_threshold
    final_preds = _apply_confidence_threshold(final_proba, threshold)

    # Diagnostics
    diag = _window_diagnostics(
        w_idx + 1, train_df, test_df, train_df["label"].to_numpy(), y_test
    )
    diag.update(
        {
            "base_train_rows": len(train_df),
            "meta_train_rows": len(y_meta),
            "base_models": list(base_models),
            "meta_model": config.model.stacking_meta_model,
            "base_model_accuracy": {
                name: float((preds.argmax(axis=1) == y_test).mean())
                for name, preds in test_outputs.items()
            },
            "mean_base_prob_sum": float(
                np.mean(list(test_outputs.values())[0].sum(axis=1))
            ),
        }
    )
    diag.update(diag_extra)

    _add_prediction_diagnostics(
        diag, final_preds, y_test, final_proba, confidence_threshold=threshold
    )

    oof_chunk = pl.DataFrame(
        {
            "timestamp": test_df["timestamp"],
            "true_label": y_test,
            "pred_label": final_preds.astype(np.int32),
            **proba_columns(final_proba, _CLASS_ORDER),
        }
    )

    # Base model predictions for comparison
    base_preds = {
        name: _CLASS_ORDER[preds.argmax(axis=1)].astype(np.int32).tolist()
        for name, preds in test_outputs.items()
    }

    return {
        "oof_chunk": oof_chunk,
        "bundle": {
            "architecture": "stacking",
            "feature_pipeline": pipeline,
            "feature_cols": selected,
            "static_cols": static_cols,
            "base_models": base_models,
            "meta_model": meta_model,
            "meta_feature_names": meta_names,
            "class_order": _CLASS_ORDER.tolist(),
        },
        "lgbm_model": base_models.get("lgbm"),
        "feature_cols": selected,
        "accuracy": diag["accuracy"],
        "diag": diag,
        "soft_vote_preds": soft_vote_preds.tolist(),
        "final_preds": final_preds.tolist(),
        "y_true": y_test.tolist(),
        "base_preds": base_preds,
    }


# Save


def _save_results(
    config: Config,
    results: list[dict[str, Any]],
    windows: list[WalkForwardWindow],
    _elapsed: float,
) -> None:
    """Persist stacking artifacts and comparison."""
    import joblib

    if not results:
        raise RuntimeError("No stacking OOF predictions generated")

    all_oof = [r["oof_chunk"] for r in results]
    diags = [r["diag"] for r in results]
    last = results[-1]
    last_bundle = last["bundle"]

    oof_df = _save_oof_predictions(
        config, all_oof_preds=all_oof, window_diagnostics=diags
    )

    model_path = Path(config.paths.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(last_bundle, model_path)

    if last["lgbm_model"] is not None:
        _save_feature_importance(last["lgbm_model"], last["feature_cols"], config)

    # Model comparison payload
    _BASE_COMPARE_NAMES = {
        "logreg": "Logistic Regression",
        "rf": "Random Forest",
        "lgbm": "LightGBM",
    }
    comparison: dict[str, dict[str, list[int]]] = {
        "hybrid_stacking": {"true": [], "pred": []}
    }
    for r in results:
        y_t = [int(x) for x in r["y_true"]]
        comparison["hybrid_stacking"]["true"].extend(y_t)
        comparison["hybrid_stacking"]["pred"].extend([int(x) for x in r["final_preds"]])
        for short_name, preds_list in r.get("base_preds", {}).items():
            key = _BASE_COMPARE_NAMES.get(short_name, short_name)
            comparison.setdefault(key, {"true": [], "pred": []})
            comparison[key]["true"].extend(y_t)
            comparison[key]["pred"].extend([int(x) for x in preds_list])

    out_cmp = {
        name: _classification_summary(v["true"], v["pred"])
        for name, v in comparison.items()
    }
    cmp_path = (
        Path(config.paths.session_dir or "results")
        / "reports"
        / "model_comparison.json"
    )
    cmp_path.parent.mkdir(parents=True, exist_ok=True)
    cmp_path.write_text(json.dumps(out_cmp, indent=2))

    n_folds = config.model.stacking_internal_folds
    stacking_mode = "expanding_origin_oof" if n_folds >= 2 else "single_holdout"
    purge_bars = (
        config.model.stacking_internal_purge
        if config.model.stacking_internal_purge > 0
        else config.labels.horizon_bars
    )

    _save_training_history(
        config,
        {
            "architecture": "stacking",
            "stacking": {
                "artifact_strategy": "last_walk_forward_window",
                "validation_protocol": {
                    "outer_windows": "bar_based_walk_forward_with_purge_embargo",
                    "stacking_mode": stacking_mode,
                    **(
                        {"internal_folds": n_folds, "internal_purge": purge_bars}
                        if stacking_mode == "expanding_origin_oof"
                        else {
                            "base_meta_split": "chronological",
                            "meta_fraction": config.model.stacking_meta_fraction,
                        }
                    ),
                },
                "base_models": config.model.stacking_base_models,
                "meta_model": config.model.stacking_meta_model,
                "last_window_accuracy": last["accuracy"],
                "n_features": len(last["feature_cols"]),
            },
            "deployment_note": (
                f"Stacking bundle from window {len(results)}/{len(windows)} "
                "(last chronological window, no future data seen)"
            ),
            "per_window_accuracies": {
                str(d.get("window")): d.get("accuracy") for d in diags
            },
        },
    )

    _save_walk_forward_history(
        config,
        windows=windows,
        window_diagnostics=diags,
        oof_len=len(oof_df),
        architecture="stacking",
    )
    _save_arch_copy(oof_df, "stacking", config)
    logger.info(
        "Stacking walk-forward complete: %d windows, %d OOF rows",
        len(windows),
        len(oof_df),
    )


# Entry point


def train_stacking_walk_forward(config: Config) -> None:
    """Run stacking walk-forward training."""
    from thesis.stage_4_training.walk_forward.loop import run_walk_forward

    if config.model.objective != "multiclass":
        raise ValueError("Stacking architecture requires objective='multiclass'")

    run_walk_forward(
        config,
        prepare_fn=_prepare_for_stacking,
        window_fn=_train_stacking_window,
        save_fn=_save_results,
    )


def _prepare_for_stacking(
    config: Config,
) -> tuple[pl.DataFrame, list[WalkForwardWindow], list[str], dict[str, Any]]:
    """Prepare stacking data. Multiclass only."""
    from thesis.stage_4_training.validation import generate_windows
    from thesis.stage_4_training.walk_forward.lgbm import _load_labeled_data

    df, is_reg = _load_labeled_data(config)
    if is_reg:
        raise ValueError("Stacking architecture does not support regression objective")

    event_end = df["event_end"].to_numpy() if "event_end" in df.columns else None
    windows = generate_windows(
        total_bars=len(df),
        train_window_bars=config.validation.train_window_bars,
        test_window_bars=config.validation.test_window_bars,
        step_bars=config.validation.step_bars,
        purge_bars=config.validation.purge_bars,
        embargo_bars=config.validation.embargo_bars,
        min_train_bars=config.validation.min_train_bars,
        event_end=event_end,
    )
    if not windows:
        raise RuntimeError("No valid walk-forward windows")

    from thesis.shared.constants import EXCLUDE_COLS

    feature_cols = sorted(c for c in df.columns if c not in EXCLUDE_COLS)
    return df, windows, feature_cols, {"is_regression": False}
