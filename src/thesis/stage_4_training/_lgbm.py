"""Hybrid GRU + LightGBM model — training, tuning, and interpretation.

Merged from ``hybrid/train.py``, ``hybrid/lgbm.py``, and ``hybrid/interpret.py``
into a single module for walk-forward pipeline integration.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from thesis._shared.config import Config
from thesis._shared.constants import (
    DIST_SHIFT_CLIP_MAX,
    DIST_SHIFT_CLIP_MIN,
    EXCLUDE_COLS,
)
from thesis.stage_4_training._gru import (
    extract_hidden_states,
    prepare_sequences,
    save_gru_model,
    train_gru,
)
from thesis._shared.ui import console

logger = logging.getLogger("thesis.model")

# ---------------------------------------------------------------------------
# LightGBM utilities
# ---------------------------------------------------------------------------


def _wrap_np(X: np.ndarray, feature_cols: list[str]) -> Any:
    """Wrap a NumPy matrix as a pandas DataFrame.

    Args:
        X: Feature matrix of shape ``(n_samples, n_features)``.
        feature_cols: Feature names aligned to matrix columns.

    Returns:
        A pandas DataFrame preserving feature names for LightGBM.
    """
    import pandas as pd

    return pd.DataFrame(X, columns=feature_cols)


def _build_interaction_constraints(feature_cols: list[str]) -> list[list[int]]:
    """Interaction constraints for LightGBM feature groups.

    Currently disabled — returning empty list allows full cross-group
    interaction between GRU hidden states and static price-action features.
    This lets LightGBM discover the most informative feature combinations
    without artificial restrictions.
    """
    return []


def _compute_class_weights(y: np.ndarray) -> dict[int, float]:
    """Compute balanced class weights for multiclass labels.

    Args:
        y: Target labels.

    Returns:
        Mapping from class label to balanced class weight.
    """
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def _compute_distribution_shift_weights(
    y_train: np.ndarray,
    y_val: np.ndarray,
    clip_range: tuple[float, float] = (DIST_SHIFT_CLIP_MIN, DIST_SHIFT_CLIP_MAX),
) -> np.ndarray:
    """Compute per-sample weights for validation to correct distribution shift.

    Compares class frequencies between training and validation sets.
    Classes under-represented in validation relative to training are
    up-weighted so the validation metric better reflects the training
    distribution's priorities.

    Args:
        y_train: Training labels in ``{-1, 0, 1}``.
        y_val: Validation labels in ``{-1, 0, 1}``.
        clip_range: Min/max bounds for per-class weight ratios.

    Returns:
        Per-sample weight array aligned to ``y_val``.
    """
    classes = np.array([-1, 0, 1])
    train_counts = np.array([np.sum(y_train == c) for c in classes], dtype=np.float64)
    val_counts = np.array([np.sum(y_val == c) for c in classes], dtype=np.float64)

    train_freq = train_counts / train_counts.sum()
    val_freq = val_counts / val_counts.sum()

    # Avoid division by zero — classes absent from val get max weight
    val_freq_safe = np.where(val_freq > 0, val_freq, 1e-8)
    ratios = train_freq / val_freq_safe
    ratios = np.clip(ratios, clip_range[0], clip_range[1])

    # Map per-class weight to per-sample
    weight_map = {int(c): float(r) for c, r in zip(classes, ratios)}
    _sample_weights = np.array([weight_map[int(y)] for y in y_val], dtype=np.float64)

    logger.info(
        "Distribution-shift weights: SHORT=%d→%.2f HOLD=%d→%.2f LONG=%d→%.2f "
        "(train freq: [%.1f%%, %.1f%%, %.1f%%] val freq: [%.1f%%, %.1f%%, %.1f%%])",
        int(train_counts[0]),
        ratios[0],
        int(train_counts[1]),
        ratios[1],
        int(train_counts[2]),
        ratios[2],
        train_freq[0] * 100,
        train_freq[1] * 100,
        train_freq[2] * 100,
        val_freq[0] * 100,
        val_freq[1] * 100,
        val_freq[2] * 100,
    )

    # Disabled: distribution shift weights caused regression.
    # Keep logging above for future diagnostics; return uniform weights.
    logger.info(
        "Distribution-shift weights DISABLED — returning uniform weights (all 1.0)"
    )
    return np.ones(len(y_val))


def _filter_validation_to_seen_classes(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    y_train: np.ndarray,
    feature_cols: list[str],
) -> tuple[Any, np.ndarray] | None:
    """Drop validation rows whose class is absent from the training fold.

    LightGBM's sklearn wrapper label-encodes classes from ``y_train`` and
    cannot transform an ``eval_set`` containing unseen labels. Small
    walk-forward/stacking folds can miss the rare Hold class, so validation is
    filtered to classes actually learnable in that fold.

    Returns ``None`` when the validation set has **no** overlapping classes
    with training — early stopping should be skipped for that fold.

    Returns:
        ``(X_val_filtered, y_val_filtered)`` or ``None``.
    """
    seen = np.unique(y_train)
    mask = np.isin(y_val, seen)
    if not mask.all():
        logger.warning(
            "LightGBM validation contains %d row(s) from unseen train classes %s; "
            "dropping them from early-stopping eval_set",
            int((~mask).sum()),
            sorted(set(map(int, y_val[~mask]))),
        )
    if not mask.any():
        logger.warning(
            "Validation set has no overlapping classes with training "
            "— skipping early stopping"
        )
        return None
    return _wrap_np(X_val[mask], feature_cols), y_val[mask]


# ---------------------------------------------------------------------------
# LightGBM training — fixed hyperparameters
# ---------------------------------------------------------------------------


def _train_fixed(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weights: dict[int, float] | None,
    config: Config,
    feature_cols: list[str],
    sample_weight: np.ndarray | None = None,
) -> Any:
    """Train LightGBM with fixed hyperparameters.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels (multiclass) or continuous targets (regression).
        X_val: Validation feature matrix.
        y_val: Validation labels or targets.
        class_weights: Balanced class weights (None for regression).
        config: Resolved application configuration.
        feature_cols: Ordered feature names.
        sample_weight: Optional per-row training weights.

    Returns:
        Fitted LightGBM model (classifier or regressor).
    """
    import lightgbm as lgb

    m = config.model
    is_regression = m.objective == "regression"
    constraints = _build_interaction_constraints(feature_cols)
    gru_feature_count = sum(1 for name in feature_cols if name.startswith("gru_h"))
    static_feature_count = len(feature_cols) - gru_feature_count
    logger.info(
        "LightGBM: %s leaves=%d depth=%d lr=%.4f n_est=%d constraints=[%d GRU, %d static]",
        "regressor" if is_regression else "classifier",
        m.num_leaves,
        m.max_depth,
        m.learning_rate,
        m.n_estimators,
        gru_feature_count,
        static_feature_count,
    )

    start_time = time.perf_counter()

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
            interaction_constraints=constraints,
            class_weight=class_weights,
            objective="multiclass",
            num_class=3,
            random_state=config.workflow.random_seed,
            n_jobs=config.workflow.n_jobs,
            verbose=-1,
            use_missing=False,
            zero_as_missing=False,
        )

    # Rich progress bar over boosting iterations
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold magenta]LightGBM boosting"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TextColumn("[cyan]v_loss={task.fields[v_loss]:.4f}"),
        TimeElapsedColumn(),
        transient=False,
    )

    with progress:
        task = progress.add_task("iter", total=m.n_estimators, v_loss=0.0)

        if is_regression:
            filtered = _wrap_np(X_val, feature_cols), y_val
        else:
            filtered = _filter_validation_to_seen_classes(
                X_train, X_val, y_val, y_train, feature_cols
            )

        def _progress_cb(env: Any) -> None:
            """Update progress bar from LightGBM callback state.

            Args:
                env: LightGBM callback environment.
            """
            progress.update(
                task,
                advance=1,
                v_loss=env.evaluation_result_list[0][2]
                if env.evaluation_result_list
                else 0.0,
            )

        if filtered is None:
            logger.warning(
                "Validation set has no overlapping classes with training "
                "— skipping early stopping"
            )
            model.fit(
                _wrap_np(X_train, feature_cols),
                y_train,
                sample_weight=sample_weight,
            )
        else:
            X_val_df, y_val_eval = filtered
            model.fit(
                _wrap_np(X_train, feature_cols),
                y_train,
                sample_weight=sample_weight,
                eval_set=[(X_val_df, y_val_eval)],
                callbacks=[
                    lgb.early_stopping(m.early_stopping_rounds, verbose=False),
                    _progress_cb,
                ],
            )

    train_time = time.perf_counter() - start_time
    logger.info(
        "LightGBM done: best_iter=%d (%.1fs)",
        model.best_iteration_,
        train_time,
    )
    return model


def _save_feature_importance(
    model: Any, feature_cols: list[str], config: Config
) -> None:
    """Save sorted model feature importances to JSON.

    Args:
        model: Fitted model exposing ``feature_importances_``.
        feature_cols: Ordered feature names.
        config: Resolved application configuration.
    """
    try:
        imp = model.feature_importances_
        pairs = sorted(zip(feature_cols, imp), key=lambda x: x[1], reverse=True)
        if config.paths.session_dir:
            out_path = (
                Path(config.paths.session_dir) / "reports" / "feature_importance.json"
            )
        else:
            out_path = Path("results/feature_importance.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({name: float(val) for name, val in pairs}, f, indent=2)
        logger.info(
            "Feature importance saved (top 5: %s)",
            [p[0] for p in pairs[:5]],
        )
    except Exception as e:
        logger.warning("Feature importance save failed: %s", e)


# ---------------------------------------------------------------------------
# Hybrid matrix helpers
# ---------------------------------------------------------------------------


def _normalize_label(lbl: int) -> str:
    """Normalize a class label for probability column naming.

    Args:
        lbl: Integer class label.

    Returns:
        A string-safe label where negatives are prefixed with ``minus``.
    """
    if lbl < 0:
        return f"minus{abs(lbl)}"
    return str(lbl)


def _align_splits_with_sequences(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    train_hidden: np.ndarray,
    val_hidden: np.ndarray,
    test_hidden: np.ndarray,
    seq_len: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Align DataFrames with GRU sequence outputs.

    Args:
        train_df: Full training DataFrame.
        val_df: Full validation DataFrame.
        test_df: Full test DataFrame.
        train_hidden: GRU hidden states for training.
        val_hidden: GRU hidden states for validation.
        test_hidden: GRU hidden states for test.
        seq_len: GRU sequence length.

    Returns:
        Tuple of (train_aligned, val_aligned, test_aligned) DataFrames.
    """
    train_aligned = train_df.slice(seq_len - 1, len(train_hidden))
    val_aligned = val_df.slice(seq_len - 1, len(val_hidden))
    test_aligned = test_df.slice(seq_len - 1, len(test_hidden))
    logger.info(
        "Aligned: train=%d val=%d test=%d",
        len(train_aligned),
        len(val_aligned),
        len(test_aligned),
    )
    return train_aligned, val_aligned, test_aligned


def _build_hybrid_matrix(
    train_hidden: np.ndarray,
    val_hidden: np.ndarray,
    test_hidden: np.ndarray,
    train_aligned: pl.DataFrame,
    val_aligned: pl.DataFrame,
    test_aligned: pl.DataFrame,
    static_cols: list[str],
    hidden_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build hybrid feature matrices combining GRU hidden states with static features.

    Args:
        train_hidden: GRU hidden states for training.
        val_hidden: GRU hidden states for validation.
        test_hidden: GRU hidden states for test.
        train_aligned: Aligned training DataFrame.
        val_aligned: Aligned validation DataFrame.
        test_aligned: Aligned test DataFrame.
        static_cols: List of static feature column names.
        hidden_size: GRU hidden size (number of GRU features).

    Returns:
        Tuple of (X_train, X_val, X_test, feature_names).
    """
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
    return X_train, X_val, X_test, all_feature_cols


def _save_predictions(
    test_aligned: pl.DataFrame,
    y_test: np.ndarray,
    preds: np.ndarray,
    proba: np.ndarray,
    class_order: list,
    preds_path: Path,
) -> None:
    """Save predictions as Parquet and CSV files.

    Args:
        test_aligned: Aligned test DataFrame with timestamps.
        y_test: True labels.
        preds: Predicted labels.
        proba: Prediction probabilities.
        class_order: Class order mapping.
        preds_path: Destination path for Parquet file.
    """
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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def train_model(config: Config) -> None:
    """**Pipeline Stage 4 (of 6):** Train and evaluate the hybrid GRU + LightGBM model.

    This stage trains the GRU feature extractor, builds hybrid features,
    trains LightGBM, saves artifacts, and computes interpretation outputs.

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

    # Load splits
    train_df = pl.read_parquet(train_path)
    val_df = pl.read_parquet(val_path)
    test_df = pl.read_parquet(test_path)

    logger.info(
        "Splits: train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df)
    )

    # --- 1. Train GRU feature extractor ---
    console.print(
        Panel(
            "Stage 4.1: [bold]GRU Feature Extractor[/]", style="magenta", padding=(0, 2)
        )
    )
    (
        gru_model,
        _gru_classifier,
        train_hidden,
        val_hidden,
        gru_history,
        gru_mean,
        gru_std,
        gru_cols,
    ) = train_gru(config, train_df, val_df)

    # Save GRU model (single source of truth: paths.gru_model)
    gru_path = Path(config.paths.gru_model)
    gru_path.parent.mkdir(parents=True, exist_ok=True)
    save_gru_model(gru_model, config, gru_path, mean=gru_mean, std=gru_std)

    # Extract hidden states for test set (using dynamically filtered gru_cols)
    test_seq, _, _ = prepare_sequences(test_df, gru_cols, config.gru.sequence_length)
    test_hidden = extract_hidden_states(
        gru_model,
        test_seq,
        config.gru.batch_size,
        mean=gru_mean,
        std=gru_std,
    )

    # --- 2. Align DataFrames with GRU sequences ---
    seq_len = config.gru.sequence_length
    train_aligned, val_aligned, test_aligned = _align_splits_with_sequences(
        train_df,
        val_df,
        test_df,
        train_hidden,
        val_hidden,
        test_hidden,
        seq_len,
    )

    # --- 3. Build hybrid feature matrix ---
    static_cols = [c for c in train_aligned.columns if c not in EXCLUDE_COLS]
    hidden_size = config.gru.hidden_size
    X_train, X_val, X_test, all_feature_cols = _build_hybrid_matrix(
        train_hidden,
        val_hidden,
        test_hidden,
        train_aligned,
        val_aligned,
        test_aligned,
        static_cols,
        hidden_size,
    )

    y_train = train_aligned["label"].to_numpy().astype(np.int32)
    y_val = val_aligned["label"].to_numpy().astype(np.int32)
    y_test = test_aligned["label"].to_numpy().astype(np.int32)
    train_weights = (
        train_aligned["sample_weight"].to_numpy().astype(np.float64)
        if "sample_weight" in train_aligned.columns
        else None
    )

    logger.info(
        "Features: %d total (%d GRU + %d static)",
        len(all_feature_cols),
        hidden_size,
        len(static_cols),
    )

    # --- 4. Train LightGBM ---
    console.print(
        Panel("Stage 4.2: [bold]LightGBM[/] (Fixed)", style="magenta", padding=(0, 2))
    )
    class_weights = _compute_class_weights(y_train)

    model = _train_fixed(
        X_train,
        y_train,
        X_val,
        y_val,
        class_weights,
        config,
        all_feature_cols,
        sample_weight=train_weights,
    )

    # Save LightGBM model
    model_path = Path(config.paths.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info("Saved model: %s", model_path)

    is_regression = config.model.objective == "regression"

    # Save training history
    models_dir = model_path.parent
    history_path = models_dir / "training_history.json"
    lgbm_info: dict = {
        "best_iteration": int(model.best_iteration_)
        if hasattr(model, "best_iteration_")
        else None,
        "n_features": len(all_feature_cols),
        "objective": config.model.objective,
        "n_classes": int(model.n_classes_) if hasattr(model, "n_classes_") else None,
    }
    training_history = {
        "gru": gru_history,
        "lightgbm": lgbm_info,
    }
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2)

    # --- 5. Generate test predictions ---
    console.print(
        Panel(
            "Stage 4.3: [bold]Predictions & Evaluation[/]",
            style="magenta",
            padding=(0, 2),
        )
    )

    if is_regression:
        raw_preds = model.predict(_wrap_np(X_test, all_feature_cols))
        # Threshold at 0: pred > 0 → Long (1), pred < 0 → Short (-1)
        preds = np.where(raw_preds > 0, 1, np.where(raw_preds < 0, -1, 0))
        proba = None  # No probability matrix for regression
    else:
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

    console.print(table)
    logger.info("Test accuracy: %.4f", acc)

    if is_regression:
        # Save predictions with raw values and thresholded labels
        preds_path = Path(config.paths.predictions)
        preds_path.parent.mkdir(parents=True, exist_ok=True)
        preds_df = pl.DataFrame(
            {
                "timestamp": test_aligned["timestamp"],
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
        _save_predictions(test_aligned, y_test, preds, proba, class_order, preds_path)

    # --- 6. Feature importance ---
    _save_feature_importance(model, all_feature_cols, config)

    # Final summary panel
    stage_time = time.perf_counter() - stage_start
    console.print(
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
