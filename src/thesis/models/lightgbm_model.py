"""LightGBM model training with Optuna hyperparameter tuning."""

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.utils.class_weight import compute_class_weight

from thesis.config.loader import Config

logger = logging.getLogger("thesis.models")


def train_lightgbm(config: Config) -> None:
    """Train the LightGBM base model and save stacking artifacts.

    This stage loads train/validation parquet files, trains LightGBM with either
    fixed parameters or Optuna tuning, persists the model, and writes validation
    probabilities aligned by timestamp for the stacking stage.

    Args:
        config: Loaded application configuration.
    """
    logger.info("Loading training data...")

    train_path = Path(config.paths.train_data)
    val_path = Path(config.paths.val_data)

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("Training/validation data not found. Run split stage.")

    # Load data
    train_df = pl.read_parquet(train_path)
    val_df = pl.read_parquet(val_path)

    # Get feature columns (exclude timestamp, label, price columns)
    exclude_cols = [
        "timestamp",
        "label",
        "tp_price",
        "sl_price",
        "touched_bar",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "avg_spread",
        "tick_count",
    ]
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    logger.info(f"Training with {len(feature_cols)} features")

    # Prepare data
    X_train = train_df.select(feature_cols).to_numpy()
    y_train = train_df["label"].to_numpy()
    X_val = val_df.select(feature_cols).to_numpy()
    y_val = val_df["label"].to_numpy()

    # Handle class imbalance
    class_weights = None
    if config.models["tree"].use_class_weights:
        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        class_weights = {
            int(class_id): float(weight) for class_id, weight in zip(classes, weights)
        }
        logger.info(f"Class weights: {class_weights}")

    # Import LightGBM
    try:
        import lightgbm as lgb  # noqa: F401
    except ImportError:
        raise ImportError("LightGBM not installed. Run: pip install lightgbm")

    # Train model
    if config.models["tree"].use_optuna:
        model = _train_with_optuna(
            X_train,
            y_train,
            X_val,
            y_val,
            class_weights,
            config,
            train_df=train_df,  # Pass train_df for Walk-Forward CV
            feature_cols=feature_cols,  # Pass feature names for DataFrames
        )
    else:
        model = _train_fixed_params(
            X_train, y_train, X_val, y_val, class_weights, config
        )

    # Save model
    model_path = Path(config.models["tree"].model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info(f"Saved model: {model_path}")

    # Generate validation predictions for stacking (aligned with LSTM)
    logger.info("Generating validation predictions for stacking...")

    # Use DataFrame to preserve feature names and avoid warnings
    X_val_df = pd.DataFrame(X_val, columns=feature_cols)
    val_probs = model.predict_proba(X_val_df)

    # Convert to dense array if sparse
    if hasattr(val_probs, "toarray"):
        val_probs = val_probs.toarray()

    # Save predictions with timestamps for alignment
    preds_df = pl.DataFrame(
        {
            "timestamp": val_df["timestamp"],
            "true_label": y_val,
            "pred_proba_class_1": val_probs[:, 2],  # Long
            "pred_proba_class_0": val_probs[:, 1],  # Hold
            "pred_proba_class_minus1": val_probs[:, 0],  # Short
        }
    )

    preds_path = Path(config.models["tree"].predictions_path)
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    preds_df.write_parquet(preds_path)
    logger.info(f"Saved validation predictions: {preds_path}")


def _train_fixed_params(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weights: dict[int, float] | None,
    config: Config,
) -> Any:
    """Train LightGBM with fixed configuration values.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features used for early stopping.
        y_val: Validation labels.
        class_weights: Optional class rebalancing weights.
        config: Loaded application configuration.

    Returns:
        Trained ``lightgbm.LGBMClassifier`` instance.
    """
    import lightgbm as lgb

    cfg = config.models["tree"]

    model = lgb.LGBMClassifier(
        num_leaves=cfg.num_leaves,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        n_estimators=cfg.n_estimators,
        min_child_samples=cfg.min_child_samples,
        subsample=cfg.subsample,
        subsample_freq=cfg.subsample_freq,
        colsample_bytree=cfg.colsample_bytree,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        class_weight=class_weights,
        objective="multiclass",
        num_class=3,
        random_state=config.workflow.random_seed,
        n_jobs=config.workflow.n_jobs,
        verbose=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
    )

    return model


def _train_with_optuna(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weights: dict[int, float] | None,
    config: Config,
    train_df: pl.DataFrame | None = None,
    feature_cols: list[str] | None = None,
) -> Any:
    """Train LightGBM with Optuna hyperparameter optimization.

    Uses walk-forward cross-validation when available, otherwise falls back to
    the fixed train/validation split.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        class_weights: Optional class rebalancing weights.
        config: Loaded application configuration.
        train_df: Optional training frame with timestamps for walk-forward CV.
        feature_cols: Optional feature names for DataFrame-based fitting.

    Returns:
        Trained ``lightgbm.LGBMClassifier`` instance.
    """
    import lightgbm as lgb
    import optuna
    from sklearn.metrics import accuracy_score, f1_score
    from thesis.models.cross_validation import (
        ExpandingWindowCV,
        SlidingWindowCV,
        create_cv_splitter,
    )
    import pandas as pd

    cfg = config.models["tree"]

    # Check if we should use Walk-Forward CV (from splitting config)
    use_wf_cv = getattr(config.splitting, "use_walk_forward_cv", True)

    cv_splitter: SlidingWindowCV | ExpandingWindowCV | None = None
    if use_wf_cv and train_df is not None:
        candidate_splitter = create_cv_splitter(config)
        if isinstance(candidate_splitter, (SlidingWindowCV, ExpandingWindowCV)):
            cv_splitter = candidate_splitter
            logger.info("Using Walk-Forward Cross-Validation for hyperparameter tuning")
        else:
            logger.warning(
                "Walk-forward CV was requested but splitter is not walk-forward. "
                "Falling back to fixed train/val tuning."
            )
    else:
        logger.info("Using fixed train/val split for hyperparameter tuning")

    def objective(trial: Any) -> float:
        """Evaluate one Optuna trial and return its scalar score."""
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "class_weight": class_weights,
            "objective": "multiclass",
            "num_class": 3,
            "random_state": config.workflow.random_seed,
            "n_jobs": -1,
            "verbose": -1,
        }

        if cv_splitter is not None and train_df is not None:
            # Walk-Forward CV: Evaluate on multiple time windows
            fold_scores = []
            fold_f1_scores = []

            # Convert numpy arrays to pandas DataFrames with feature names to avoid warnings
            X_train_df = (
                pd.DataFrame(X_train, columns=feature_cols) if feature_cols else None
            )

            for fold_idx, (train_idx, val_idx, _window_name) in enumerate(
                cv_splitter.split(train_df)
            ):
                # Use DataFrames with feature names instead of numpy arrays
                if X_train_df is not None:
                    X_tr_fold = X_train_df.iloc[train_idx]
                    X_val_fold = X_train_df.iloc[val_idx]
                else:
                    X_tr_fold = X_train[train_idx]
                    X_val_fold = X_train[val_idx]

                y_tr_fold = y_train[train_idx]
                y_val_fold = y_train[val_idx]

                # Skip if not enough samples
                if len(X_tr_fold) < 100 or len(X_val_fold) < 50:
                    continue

                model = lgb.LGBMClassifier(**params)
                model.fit(
                    X_tr_fold,
                    y_tr_fold,
                    eval_set=[(X_val_fold, y_val_fold)],
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                )

                preds = model.predict(X_val_fold)
                acc = accuracy_score(y_val_fold, preds)
                f1 = f1_score(y_val_fold, preds, average="weighted")

                fold_scores.append(acc)
                fold_f1_scores.append(f1)

                trial.report(acc, fold_idx)

                # Prune if performance is poor
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if len(fold_scores) == 0:
                return 0.0

            # Return mean accuracy across folds (more robust than single validation)
            mean_acc = float(np.mean(fold_scores))
            mean_f1 = float(np.mean(fold_f1_scores))

            # Combine accuracy and F1 for balanced metric
            score = 0.6 * mean_acc + 0.4 * mean_f1

        else:
            # Fixed train/val split (fallback)
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )

            preds = model.predict(X_val)
            score = accuracy_score(y_val, preds)

        return score

    # Optuna study with pruning (seeded sampler for reproducibility)
    sampler = optuna.samplers.TPESampler(seed=config.workflow.random_seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=cfg.optuna_trials, timeout=cfg.optuna_timeout)

    logger.info(f"Best params: {study.best_params}")
    logger.info(f"Best validation score: {study.best_value:.4f}")

    # Train final model with best params on full training set
    best_params = study.best_params.copy()
    best_params["class_weight"] = class_weights
    best_params["objective"] = "multiclass"
    best_params["num_class"] = 3
    best_params["random_state"] = config.workflow.random_seed
    best_params["n_jobs"] = -1
    best_params["verbose"] = -1

    model = lgb.LGBMClassifier(**best_params)

    # Train on training set only so that validation predictions passed to
    # the stacking meta-learner are truly out-of-sample.
    logger.info("Training final model on training set only (OOS for stacking)...")
    model.fit(X_train, y_train)

    return model


def generate_lightgbm_oof(config: Config) -> None:
    """Generate out-of-fold predictions for LightGBM on the training set.

    Uses SlidingWindowCV to split the training data into folds, trains a
    LightGBM clone on each fold's training portion, and predicts on the
    fold's validation portion.  The combined fold-validation predictions
    form the OOF matrix used to train the stacking meta-learner, ensuring
    no information leaks from the validation or test periods.

    Args:
        config: Loaded application configuration.
    """
    import lightgbm as lgb
    from thesis.models.cross_validation import create_cv_splitter

    train_path = Path(config.paths.train_data)
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")

    logger.info("Loading training data for OOF generation...")
    train_df = pl.read_parquet(train_path)

    # Load the trained LightGBM model to get its actual hyperparameters
    model_path = Path(config.models["tree"].model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained LightGBM model not found at {model_path}. "
            "Run train_lightgbm stage first."
        )
    trained_model = joblib.load(model_path)
    base_params = trained_model.get_params()
    logger.info(
        "Using hyperparameters from trained model: "
        f"{ {k: v for k, v in base_params.items() if not k.startswith('_')} }"
    )

    # Feature columns (same exclusion as train_lightgbm)
    exclude_cols = {
        "timestamp",
        "label",
        "tp_price",
        "sl_price",
        "touched_bar",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "avg_spread",
        "tick_count",
    }
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    X_all = train_df.select(feature_cols).to_numpy()
    y_all = train_df["label"].to_numpy()
    timestamps_all = train_df["timestamp"].to_numpy()

    # Class weights (same as main training)
    class_weights = None
    if config.models["tree"].use_class_weights:
        classes = np.unique(y_all)
        weights = compute_class_weight("balanced", classes=classes, y=y_all)
        class_weights = {int(cls): float(w) for cls, w in zip(classes, weights)}

    # Create CV splitter from config
    cv_splitter = create_cv_splitter(config)

    # OOF prediction accumulator
    oof_preds = np.full((len(X_all), 3), np.nan)

    folds = list(cv_splitter.split(train_df))
    logger.info(f"Generating LightGBM OOF predictions across {len(folds)} folds...")

    cfg = config.models["tree"]

    for fold_idx, fold in enumerate(folds):
        if len(fold) == 3:
            train_idx, val_idx, fold_name = fold
        else:
            train_idx, val_idx = fold
            fold_name = f"fold_{fold_idx + 1}"
        X_fold_train = X_all[train_idx]
        y_fold_train = y_all[train_idx]
        X_fold_val = X_all[val_idx]

        # Clone the trained model's hyperparameters (not config defaults)
        clone_params = base_params.copy()
        clone_params.update(
            {
                "class_weight": class_weights,
                "random_state": config.workflow.random_seed,
                "n_jobs": config.workflow.n_jobs,
                "verbose": -1,
            }
        )
        model_clone = lgb.LGBMClassifier(**clone_params)

        # Use DataFrame to preserve feature names
        X_fold_train_df = pd.DataFrame(X_fold_train, columns=feature_cols)
        X_fold_val_df = pd.DataFrame(X_fold_val, columns=feature_cols)

        model_clone.fit(
            X_fold_train_df,
            y_fold_train,
            callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
            eval_set=[(X_fold_val_df, y_all[val_idx])],
        )

        fold_preds = np.asarray(model_clone.predict_proba(X_fold_val_df))

        oof_preds[val_idx] = fold_preds
        logger.info(
            f"  Fold {fold_idx + 1}/{len(folds)} ({fold_name}): "
            f"{len(train_idx)} train, {len(val_idx)} val"
        )

    # Remove rows that were never predicted (not covered by any fold)
    valid_mask = ~np.isnan(oof_preds[:, 0])
    n_valid = np.sum(valid_mask)
    n_total = len(oof_preds)
    logger.info(
        f"OOF coverage: {n_valid}/{n_total} samples ({n_valid / n_total * 100:.1f}%)"
    )

    if n_valid == 0:
        raise ValueError(
            "No OOF predictions generated — check CV window configuration."
        )

    # Save OOF predictions
    valid_timestamps = timestamps_all[valid_mask]
    valid_labels = y_all[valid_mask]
    valid_preds = oof_preds[valid_mask]

    oof_df = pl.DataFrame(
        {
            "timestamp": valid_timestamps,
            "true_label": valid_labels,
            "pred_proba_class_minus1": valid_preds[:, 0],
            "pred_proba_class_0": valid_preds[:, 1],
            "pred_proba_class_1": valid_preds[:, 2],
        }
    )

    oof_path = Path(config.models["tree"].oof_predictions_path)
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    oof_df.write_parquet(oof_path)
    logger.info(f"Saved LightGBM OOF predictions: {oof_path} ({len(oof_df)} samples)")
