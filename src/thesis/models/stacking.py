"""Stacking trainer. Base learners feed meta learner.

Simplified: single-holdout only (80/20 chronological split).
No OOF folds, no calibration, no passthrough.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np
import polars as pl

from thesis.models.evaluate import (
    _add_prediction_diagnostics,
    _align_proba,
    _apply_confidence_threshold,
    _save_arch_copy,
    _save_feature_importance,
    _save_oof_predictions,
    _save_training_history,
    _save_walk_forward_history,
    _window_diagnostics,
    proba_columns,
)
from thesis.models.train import (
    WalkForwardWindow,
    fit_static_feature_pipeline,
    select_static_cols,
)
from thesis.shared.config import Config

logger = logging.getLogger("thesis")

_CLASS_ORDER = np.array([-1, 0, 1], dtype=np.int32)
_BASE_ALIASES = {
    "logistic_regression": "logreg",
    "random_forest": "rf",
    "lightgbm": "lgbm",
}
_BASE_COMPARE_NAMES = {
    "logreg": "Logistic Regression",
    "rf": "Random Forest",
    "lgbm": "LightGBM",
}


# ── LightGBM helpers (merged from lgbm/utils.py) ─────────────────────────────


def _wrap_np(X: np.ndarray, feature_cols: list[str]) -> Any:
    """Wrap numpy matrix as DataFrame. Preserve feature names."""
    import pandas as pd

    return pd.DataFrame(X, columns=feature_cols)


def _compute_class_weights(y: np.ndarray) -> dict[int, float]:
    """Balanced weights. Counter class skew."""
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def _filter_unseen_classes(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    y_train: np.ndarray,
    feature_cols: list[str],
) -> tuple[Any, np.ndarray] | None:
    """Drop val rows with unseen classes.

    LightGBM eval cannot encode labels train never saw.
    """
    seen = np.unique(y_train)
    mask = np.isin(y_val, seen)
    if not mask.any():
        logger.warning(
            "Validation has no overlapping classes with training "
            "— skipping early stopping",
        )
        return None
    if not mask.all():
        logger.warning(
            "Dropping %d unseen class rows from validation", int((~mask).sum())
        )
    return _wrap_np(X_val[mask], feature_cols), y_val[mask]


def _train_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weights: dict[int, float] | None,
    config: Config,
    feature_cols: list[str],
    sample_weight: np.ndarray | None = None,
) -> Any:
    """Fit LightGBM. Early stop when validation usable."""
    import lightgbm as lgb

    m = config.model
    is_regression = m.objective == "regression"

    start = time.perf_counter()

    if is_regression:
        model = lgb.LGBMRegressor(
            num_leaves=m.num_leaves,
            max_depth=m.max_depth,
            learning_rate=m.learning_rate,
            n_estimators=m.n_estimators,
            min_child_samples=m.min_child_samples,
            subsample=m.subsample,
            subsample_freq=m.subsample_freq,
            colsample_bytree=m.feature_fraction,
            reg_alpha=m.reg_alpha,
            reg_lambda=m.reg_lambda,
            objective="regression",
            random_state=config.workflow.random_seed,
            n_jobs=config.workflow.n_jobs,
            verbose=-1,
        )
    else:
        model = lgb.LGBMClassifier(
            num_leaves=m.num_leaves,
            max_depth=m.max_depth,
            learning_rate=m.learning_rate,
            n_estimators=m.n_estimators,
            min_child_samples=m.min_child_samples,
            subsample=m.subsample,
            subsample_freq=m.subsample_freq,
            colsample_bytree=m.feature_fraction,
            reg_alpha=m.reg_alpha,
            reg_lambda=m.reg_lambda,
            interaction_constraints=[],
            class_weight=class_weights,
            objective="multiclass",
            num_class=3,
            random_state=config.workflow.random_seed,
            n_jobs=config.workflow.n_jobs,
            verbose=-1,
            use_missing=False,
            zero_as_missing=False,
        )

    # Filter val to classes seen in train
    filtered = (
        None
        if is_regression
        else _filter_unseen_classes(X_train, X_val, y_val, y_train, feature_cols)
    )

    def _progress(env: Any) -> None:
        """Log every 50 rounds."""
        if env.iteration % 50 == 0 or env.iteration == env.end_iteration - 1:
            loss = (
                env.evaluation_result_list[0][2] if env.evaluation_result_list else 0.0
            )
            logger.info("    LGBM iter=%d val_loss=%.5f", env.iteration, loss)

    if filtered is None:
        model.fit(_wrap_np(X_train, feature_cols), y_train, sample_weight=sample_weight)
    else:
        X_val_df, y_val_eval = filtered
        model.fit(
            _wrap_np(X_train, feature_cols),
            y_train,
            sample_weight=sample_weight,
            eval_set=[(X_val_df, y_val_eval)],
            callbacks=[
                lgb.early_stopping(m.early_stopping_rounds, verbose=False),
                _progress,
            ],
        )

    logger.info(
        "    LGBM done: best_iter=%d (%.1fs)",
        model.best_iteration_,
        time.perf_counter() - start,
    )
    return model


# ── Base model registry ─────────────────────────────────────────────────────


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


# ── Meta model registry ─────────────────────────────────────────────────────


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
        if len(X) <= 4:
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


# ── Probability helpers ──────────────────────────────────────────────────────


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


# ── Train one outer window ──────────────────────────────────────────────────


def _train_stacking_window(
    config: Config,
    w_idx: int,
    window: WalkForwardWindow,
    df: pl.DataFrame,
    feature_cols: list[str],
    *,
    is_regression: bool = False,
) -> dict[str, Any] | None:
    """Train/predict one stacking window.

    Protocol: chronological 80/20 split inside train window.
      - base_df = train[:80%]  → fit base models
      - meta_df = train[80%:]  → base predict_proba → meta features
      - meta model trains on meta features
      - base models refit on full train
      - predict test set via base→meta pipeline
    """
    train_df = df.slice(window.train_start_idx, window.train_len)
    test_df = df.slice(window.test_start_idx, window.test_len)
    if train_df.is_empty() or test_df.is_empty():
        return None

    y_test = test_df["label"].to_numpy().astype(np.int32)
    meta_frac = config.model.stacking_meta_fraction
    base_rows = int(round(len(train_df) * (1 - meta_frac)))
    base_df, meta_df = (
        train_df.slice(0, base_rows),
        train_df.slice(base_rows, len(train_df) - base_rows),
    )
    y_base = base_df["label"].to_numpy().astype(np.int32)
    y_meta = meta_df["label"].to_numpy().astype(np.int32)

    logger.info(
        "Window %d: single-holdout base=%d meta=%d test=%d",
        w_idx + 1,
        len(base_df),
        len(meta_df),
        len(test_df),
    )

    static_cols = select_static_cols(config, train_df, feature_cols)
    pipeline, selected = fit_static_feature_pipeline(
        config, base_df, static_cols, y_base
    )

    X_base = pipeline.transform(base_df.select(static_cols).to_pandas())
    X_meta = pipeline.transform(meta_df.select(static_cols).to_pandas())
    X_test = pipeline.transform(test_df.select(static_cols).to_pandas())

    # ── Fit base models on base_df ──────────────────────────────────────────
    base_models: dict[str, Any] = {}
    for configured_name in config.model.stacking_base_models:
        short = _BASE_ALIASES.get(configured_name, configured_name)
        if configured_name == "lightgbm":
            _val_split = max(1, int(len(X_base) * 0.2))
            X_tr, y_tr = X_base[:-_val_split], y_base[:-_val_split]
            X_val, y_val = X_base[-_val_split:], y_base[-_val_split:]
            class_weights = (
                _compute_class_weights(y_tr) if len(np.unique(y_tr)) > 1 else None
            )
            model = _train_lgbm(
                X_tr, y_tr, X_val, y_val, class_weights, config, selected
            )
        else:
            model = _fit_safe(_build_base(configured_name, config), X_base, y_base)
        base_models[short] = model

    # ── Base predict meta_df → meta features ────────────────────────────────
    meta_train_outputs = {
        name: _aligned_proba(model, X_meta, selected)
        for name, model in base_models.items()
    }
    test_outputs = {
        name: _aligned_proba(model, X_test, selected)
        for name, model in base_models.items()
    }

    # ── Refit base models on full train for deployment ──────────────────────
    y_full_train = train_df["label"].to_numpy().astype(np.int32)
    X_full_train = pipeline.transform(train_df.select(static_cols).to_pandas())
    for configured_name in config.model.stacking_base_models:
        short = _BASE_ALIASES.get(configured_name, configured_name)
        if configured_name == "lightgbm":
            _val_split = max(1, int(len(X_full_train) * 0.2))
            X_tr, y_tr = X_full_train[:-_val_split], y_full_train[:-_val_split]
            X_val, y_val = X_full_train[-_val_split:], y_full_train[-_val_split:]
            class_weights = (
                _compute_class_weights(y_tr) if len(np.unique(y_tr)) > 1 else None
            )
            base_models[short] = _train_lgbm(
                X_tr, y_tr, X_val, y_val, class_weights, config, selected
            )
        else:
            base_models[short] = _fit_safe(
                _build_base(configured_name, config), X_full_train, y_full_train
            )

    # ── Soft-voting baseline ────────────────────────────────────────────────
    soft_vote = np.mean(list(test_outputs.values()), axis=0)
    soft_vote_preds = _CLASS_ORDER[np.argmax(soft_vote, axis=1)]

    # ── Meta model ──────────────────────────────────────────────────────────
    X_meta_stack, meta_names = _stack_features(meta_train_outputs)
    X_test_stack, _ = _stack_features(test_outputs)
    meta_model = _build_meta(
        config.model.stacking_meta_model, config, X_meta_stack, y_meta
    )
    final_proba = _aligned_proba(meta_model, X_test_stack, meta_names)
    threshold = config.model.prediction_confidence_threshold
    final_preds = _apply_confidence_threshold(final_proba, threshold)

    # ── Diagnostics ─────────────────────────────────────────────────────────
    diag = _window_diagnostics(
        w_idx + 1, train_df, test_df, train_df["label"].to_numpy(), y_test
    )
    diag.update(
        {
            "base_train_rows": len(train_df),
            "meta_train_rows": len(y_meta),
            "base_models": list(base_models),
            "meta_model": config.model.stacking_meta_model,
            "stacking_mode": "single_holdout",
            "base_model_accuracy": {
                name: float((preds.argmax(axis=1) == y_test).mean())
                for name, preds in test_outputs.items()
            },
            "mean_base_prob_sum": float(
                np.mean(list(test_outputs.values())[0].sum(axis=1))
            ),
        }
    )

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


# ── Save ─────────────────────────────────────────────────────────────────────


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

    _save_training_history(
        config,
        {
            "architecture": "stacking",
            "stacking": {
                "artifact_strategy": "last_walk_forward_window",
                "validation_protocol": {
                    "outer_windows": "bar_based_walk_forward_with_purge_embargo",
                    "stacking_mode": "single_holdout",
                    "base_meta_split": "chronological",
                    "meta_fraction": config.model.stacking_meta_fraction,
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


# ── Entry point ──────────────────────────────────────────────────────────────


def train_stacking_walk_forward(config: Config) -> None:
    """Run stacking walk-forward training."""
    from thesis.models.train import run_walk_forward

    if config.model.objective != "multiclass":
        raise ValueError("Stacking architecture requires objective='multiclass'")

    run_walk_forward(
        config,
        prepare_fn=_prepare_for_stacking,
        window_fn=_train_stacking_window,
        save_fn=_save_results,
    )


def _load_labeled_data(config: Config) -> tuple[pl.DataFrame, bool]:
    """Load labels parquet. Returns (df, is_regression)."""
    path = Path(config.paths.labels)
    if not path.exists():
        raise FileNotFoundError(f"Labels not found: {path}")

    df = pl.read_parquet(path)
    logger.info("Loaded labels: %d rows", len(df))

    from thesis.models.train import compute_regression_target

    df, is_regression = compute_regression_target(df, config)
    return df, is_regression


def _prepare_for_stacking(
    config: Config,
) -> tuple[pl.DataFrame, list[WalkForwardWindow], list[str], dict[str, Any]]:
    """Prepare stacking data. Multiclass only."""
    from thesis.models.train import generate_windows

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
