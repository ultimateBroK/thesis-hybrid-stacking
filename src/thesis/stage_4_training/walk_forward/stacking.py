"""Classical time-safe stacking walk-forward trainer."""

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
from thesis.stage_4_training.lgbm.utils import (
    _compute_class_weights,
    _save_feature_importance,
    _train_fixed,
)
from thesis.stage_4_training.walk_forward.artifacts import (
    _log_walk_forward_complete,
    _save_arch_copy,
    _save_oof_predictions,
    _save_training_history,
    _save_walk_forward_history,
)
from thesis.stage_4_training.walk_forward.lgbm import _prepare_static_wf_data
from thesis.stage_4_training.walk_forward.utils import (
    _CLASS_ORDER,
    _add_prediction_diagnostics,
    _align_probability_matrix,
    _probability_columns,
    _select_static_feature_cols,
    _window_diagnostics,
    fit_static_feature_pipeline,
)

logger = logging.getLogger("thesis.pipeline")

_BASE_MODEL_ALIASES = {
    "logistic_regression": "logreg",
    "random_forest": "rf",
    "lightgbm": "lgbm",
}
_MIN_SPLIT_ROWS = 4


def _split_base_meta(
    train_df: pl.DataFrame, meta_fraction: float
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Chronologically split an outer train window into base and meta folds."""
    if not 0.0 < meta_fraction < 0.5:
        raise ValueError("stacking_meta_fraction must be between 0 and 0.5")
    if len(train_df) < _MIN_SPLIT_ROWS:
        raise ValueError("Training window too small for base/meta stacking split")
    meta_rows = max(1, int(round(len(train_df) * meta_fraction)))
    base_rows = len(train_df) - meta_rows
    if base_rows < 2 or meta_rows < 1:
        raise ValueError("Training window too small after base/meta split")
    return train_df.slice(0, base_rows), train_df.slice(base_rows, meta_rows)


def _build_sklearn_base_model(name: str, config: Config) -> Any:
    """Build a scikit-learn base learner by configured name."""
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
    raise ValueError(f"Unsupported sklearn base model: {name!r}")


def _fit_predictable_classifier(model: Any, X: np.ndarray, y: np.ndarray) -> Any:
    """Fit a classifier, falling back to DummyClassifier for one-class folds."""
    if len(np.unique(y)) < 2:
        from sklearn.dummy import DummyClassifier

        model = DummyClassifier(strategy="most_frequent")
    model.fit(X, y)
    return model


def _fit_meta_model(config: Config, X_meta: np.ndarray, y_meta: np.ndarray) -> Any:
    """Fit the probability-level meta model."""
    if len(np.unique(y_meta)) < 2:
        from sklearn.dummy import DummyClassifier

        return DummyClassifier(strategy="most_frequent").fit(X_meta, y_meta)
    if config.model.stacking_meta_model == "logistic_regression":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=config.workflow.random_seed,
        ).fit(X_meta, y_meta)
    if config.model.stacking_meta_model == "lightgbm":
        return _train_fixed(
            X_meta,
            y_meta,
            X_meta,
            y_meta,
            _compute_class_weights(y_meta),
            config,
            [f"meta_{i}" for i in range(X_meta.shape[1])],
        )
    raise ValueError(
        f"Unsupported stacking_meta_model: {config.model.stacking_meta_model!r}"
    )


def _aligned_predict_proba(model: Any, X: np.ndarray) -> np.ndarray:
    """Predict probabilities aligned to the canonical class order [-1, 0, 1]."""
    proba = model.predict_proba(X)
    return _align_probability_matrix(proba, model.classes_)


def _stack_probability_features(
    base_outputs: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    """Concatenate base probability matrices into deterministic meta features."""
    if not base_outputs:
        raise ValueError("No base model outputs available for stacking")
    row_counts = {name: matrix.shape[0] for name, matrix in base_outputs.items()}
    if len(set(row_counts.values())) != 1:
        raise ValueError(f"Base probability row counts differ: {row_counts}")

    matrices: list[np.ndarray] = []
    names: list[str] = []
    suffixes = ["short", "hold", "long"]
    for name in sorted(base_outputs):
        matrix = np.asarray(base_outputs[name], dtype=np.float64)
        if matrix.shape[1] != len(_CLASS_ORDER):
            raise ValueError(f"{name} probability matrix must have 3 columns")
        matrices.append(matrix)
        names.extend(f"{name}_proba_{suffix}" for suffix in suffixes)
    return np.hstack(matrices), names


def _classification_summary(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    """Return accuracy, macro-F1 and per-class metrics for artifact JSON."""
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
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
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


def _train_and_predict_stacking_window(
    config: Config,
    w_idx: int,
    window: Any,
    df: pl.DataFrame,
    feature_cols: list[str],
) -> dict[str, Any] | None:
    """Train base/meta learners for one outer walk-forward window."""
    train_df = df.slice(
        window.train_start_idx, window.train_end_idx - window.train_start_idx
    )
    test_df = df.slice(
        window.test_start_idx, window.test_end_idx - window.test_start_idx
    )
    if train_df.is_empty() or test_df.is_empty():
        return None
    base_df, meta_df = _split_base_meta(train_df, config.model.stacking_meta_fraction)
    y_base = base_df["label"].to_numpy().astype(np.int32)
    y_meta = meta_df["label"].to_numpy().astype(np.int32)
    y_test = test_df["label"].to_numpy().astype(np.int32)

    static_cols = _select_static_feature_cols(config, train_df, feature_cols)
    feature_pipeline, selected_cols = fit_static_feature_pipeline(
        config, base_df, static_cols, y_base
    )
    X_base = feature_pipeline.transform(base_df.select(static_cols).to_pandas())
    X_meta = feature_pipeline.transform(meta_df.select(static_cols).to_pandas())
    X_test = feature_pipeline.transform(test_df.select(static_cols).to_pandas())

    base_models: dict[str, Any] = {}
    meta_train_outputs: dict[str, np.ndarray] = {}
    test_outputs: dict[str, np.ndarray] = {}
    base_test_preds: dict[str, np.ndarray] = {}

    for configured_name in config.model.stacking_base_models:
        short_name = _BASE_MODEL_ALIASES.get(configured_name, configured_name)
        if configured_name == "lightgbm":
            class_weights = (
                _compute_class_weights(y_base)
                if len(np.unique(y_base)) > 1
                else None
            )
            model = _train_fixed(
                X_base,
                y_base,
                X_meta,
                y_meta,
                class_weights,
                config,
                selected_cols,
            )
        else:
            model = _fit_predictable_classifier(
                _build_sklearn_base_model(configured_name, config), X_base, y_base
            )
        base_models[short_name] = model
        meta_train_outputs[short_name] = _aligned_predict_proba(model, X_meta)
        test_outputs[short_name] = _aligned_predict_proba(model, X_test)
        base_test_preds[short_name] = _CLASS_ORDER[
            np.argmax(test_outputs[short_name], axis=1)
        ]

    X_meta_stack, meta_feature_names = _stack_probability_features(meta_train_outputs)
    X_test_stack, _ = _stack_probability_features(test_outputs)
    meta_model = _fit_meta_model(config, X_meta_stack, y_meta)
    final_proba = _aligned_predict_proba(meta_model, X_test_stack)
    final_preds = _CLASS_ORDER[np.argmax(final_proba, axis=1)]

    diag = _window_diagnostics(
        w_idx + 1, train_df, test_df, train_df["label"].to_numpy(), y_test
    )
    diag["base_train_rows"] = len(base_df)
    diag["meta_train_rows"] = len(meta_df)
    diag["base_models"] = list(base_models)
    diag["meta_model"] = config.model.stacking_meta_model
    diag["meta_feature_names"] = meta_feature_names
    diag["base_model_accuracy"] = {
        name: float((preds == y_test).mean()) for name, preds in base_test_preds.items()
    }
    _add_prediction_diagnostics(diag, final_preds, y_test, final_proba)

    oof_chunk = pl.DataFrame(
        {
            "timestamp": test_df["timestamp"],
            "true_label": y_test,
            "pred_label": final_preds.astype(np.int32),
            **_probability_columns(final_proba, _CLASS_ORDER),
        }
    )
    return {
        "oof_chunk": oof_chunk,
        "bundle": {
            "architecture": "stacking",
            "feature_pipeline": feature_pipeline,
            "feature_cols": selected_cols,
            "static_cols": static_cols,
            "base_models": base_models,
            "meta_model": meta_model,
            "meta_feature_names": meta_feature_names,
            "class_order": _CLASS_ORDER.tolist(),
        },
        "lgbm_model": base_models.get("lgbm"),
        "feature_cols": selected_cols,
        "accuracy": diag["accuracy"],
        "diag": diag,
        "base_preds": {name: preds.tolist() for name, preds in base_test_preds.items()},
        "final_preds": final_preds.tolist(),
        "y_true": y_test.tolist(),
    }


def _save_model_comparison(
    config: Config,
    comparison_inputs: dict[str, dict[str, list[int]]],
) -> None:
    """Persist aggregate base-vs-stacking classification metrics."""
    out = {
        name: _classification_summary(values["true"], values["pred"])
        for name, values in comparison_inputs.items()
    }
    if config.paths.session_dir:
        path = Path(config.paths.session_dir) / "reports" / "model_comparison.json"
    else:
        path = Path("results/model_comparison.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))


def train_stacking_walk_forward(config: Config) -> None:
    """Train leakage-safe classical stacking with outer walk-forward validation."""
    if config.model.objective != "multiclass":
        raise ValueError(
            "Stacking architecture currently supports objective='multiclass' only"
        )
    df, windows, feature_cols, _is_regression = _prepare_static_wf_data(config)

    all_oof_preds: list[pl.DataFrame] = []
    window_diagnostics: list[dict[str, Any]] = []
    comparison_inputs: dict[str, dict[str, list[int]]] = {
        "hybrid_stacking": {"true": [], "pred": []}
    }
    last_bundle: dict[str, Any] | None = None
    last_lgbm_model = None
    last_feature_cols: list[str] = []
    last_window_accuracy: float | None = None
    last_window_index = 0
    stage_start = time.perf_counter()

    for w_idx, window in enumerate(windows):
        console.rule(
            f"[bold cyan]Stacking window {w_idx + 1}/{len(windows)}[/]", style="cyan"
        )
        result = _train_and_predict_stacking_window(
            config, w_idx, window, df, feature_cols
        )
        if result is None:
            continue
        all_oof_preds.append(result["oof_chunk"])
        window_diagnostics.append(result["diag"])
        last_bundle = result["bundle"]
        last_lgbm_model = result["lgbm_model"]
        last_feature_cols = result["feature_cols"]
        last_window_accuracy = result["accuracy"]
        last_window_index = w_idx + 1

        y_true = [int(x) for x in result["y_true"]]
        comparison_inputs["hybrid_stacking"]["true"].extend(y_true)
        comparison_inputs["hybrid_stacking"]["pred"].extend(
            [int(x) for x in result["final_preds"]]
        )
        for name, preds in result["base_preds"].items():
            comparison_inputs.setdefault(name, {"true": [], "pred": []})
            comparison_inputs[name]["true"].extend(y_true)
            comparison_inputs[name]["pred"].extend([int(x) for x in preds])

    if not all_oof_preds or last_bundle is None:
        raise RuntimeError("No stacking OOF predictions generated")

    import joblib

    oof_df = _save_oof_predictions(
        config,
        all_oof_preds=all_oof_preds,
        window_diagnostics=window_diagnostics,
    )
    model_path = Path(config.paths.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(last_bundle, model_path)
    if last_lgbm_model is not None:
        _save_feature_importance(last_lgbm_model, last_feature_cols, config)
    _save_model_comparison(config, comparison_inputs)

    per_window_accuracies = {
        str(d.get("window")): d.get("accuracy") for d in window_diagnostics
    }
    _save_training_history(
        config,
        {
            "architecture": "stacking",
            "stacking": {
                "artifact_strategy": "last_walk_forward_window",
                "validation_protocol": {
                    "outer_windows": "bar_based_walk_forward_with_purge_embargo",
                    "base_meta_split": "chronological_train_head_meta_tail",
                    "meta_fraction": config.model.stacking_meta_fraction,
                },
                "base_models": config.model.stacking_base_models,
                "meta_model": config.model.stacking_meta_model,
                "last_window_accuracy": last_window_accuracy,
                "n_features": len(last_feature_cols),
            },
            "deployment_note": (
                f"Stacking bundle saved from window {last_window_index}/{len(windows)} "
                "(last chronological walk-forward window). It has not seen future data."
            ),
            "per_window_accuracies": per_window_accuracies,
        },
    )
    _save_walk_forward_history(
        config,
        windows=windows,
        window_diagnostics=window_diagnostics,
        oof_len=len(oof_df),
        architecture="stacking",
    )
    _save_arch_copy(oof_df, "stacking", config)
    _log_walk_forward_complete(
        arch_name="stacking",
        windows_count=len(windows),
        oof_len=len(oof_df),
        stage_start=stage_start,
        prefix="Stacking walk-forward complete",
    )
