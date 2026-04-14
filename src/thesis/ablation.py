"""Ablation study: LightGBM-only vs GRU-only vs Combined (GRU + LightGBM).

Runs three variants on the same data splits and compares trading performance.
GRU is trained **once** and shared across variants B and C to avoid redundant
~5 min training runs.

Variants:
    A. LightGBM-only  — static features only (no GRU hidden states)
    B. GRU-only       — direct softmax predictions (no LightGBM)
    C. Combined       — GRU hidden states + static features → LightGBM (current)

Usage:
    Called from pipeline or standalone:
        from thesis.ablation import run_ablation
        run_ablation(config)
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from thesis.config import Config
from thesis.gru_model import (
    extract_hidden_states,
    prepare_sequences,
    train_gru,
)

logger = logging.getLogger("thesis.ablation")

# Same exclude set as model.py — label-derived + raw price + GRU-input columns
_EXCLUDE_COLS = frozenset(
    [
        "timestamp",
        "label",
        "tp_price",
        "sl_price",
        "touched_bar",
        "open_right",
        "high_right",
        "low_right",
        "close_right",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "avg_spread",
        "tick_count",
        "dead_hour",
        "log_returns",
    ]
)


def _wrap_np(X: np.ndarray, feature_cols: list[str]) -> Any:
    """Wrap numpy array in DataFrame to preserve feature names for LightGBM."""
    import pandas as pd

    return pd.DataFrame(X, columns=feature_cols)


def _compute_class_weights(y: np.ndarray) -> dict[int, float]:
    """Compute balanced class weights."""
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def _train_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weights: dict[int, float],
    config: Config,
    feature_cols: list[str],
) -> Any:
    """Train LightGBM classifier with fixed hyperparameters."""
    import lightgbm as lgb

    m = config.model
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
        class_weight=class_weights,
        objective="multiclass",
        num_class=3,
        random_state=config.workflow.random_seed,
        n_jobs=config.workflow.n_jobs,
        verbose=-1,
    )
    model.fit(
        _wrap_np(X_train, feature_cols),
        y_train,
        eval_set=[(_wrap_np(X_val, feature_cols), y_val)],
        callbacks=[lgb.early_stopping(m.early_stopping_rounds, verbose=False)],
    )
    logger.info("LightGBM best iteration: %d", model.best_iteration_)
    return model


def _save_preds(
    timestamps: pl.Series,
    y_true: np.ndarray,
    preds: np.ndarray,
    proba: np.ndarray,
    path: Path,
) -> None:
    """Save prediction parquet with standard columns."""
    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "true_label": y_true.astype(np.int32),
            "pred_label": preds.astype(np.int32),
            "pred_proba_class_minus1": proba[:, 0],
            "pred_proba_class_0": proba[:, 1],
            "pred_proba_class_1": proba[:, 2],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    logger.info("Saved predictions: %s (%d rows)", path, len(df))


def _run_backtest_for(
    test_df: pl.DataFrame,
    preds_df: pl.DataFrame,
    config: Config,
    variant: str,
) -> dict:
    """Run backtest for a variant and return metrics dict."""
    from thesis.backtest import run_backtest_from_data

    test = test_df.with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
    preds = preds_df.with_columns(pl.col("timestamp").cast(pl.Datetime("us")))

    logger.info("%s: running backtest", variant)

    return run_backtest_from_data(test, preds, config)


def _log_comparison(comparison: dict) -> None:
    """Log formatted comparison table."""
    logger.info("=" * 70)
    logger.info("ABLATION COMPARISON")
    logger.info("=" * 70)

    header = f"{'Metric':<25} {'LGBM-only':>14} {'GRU-only':>14} {'Combined':>14}"
    logger.info(header)
    logger.info("-" * 70)

    metrics_keys = [
        ("num_trades", "Trades"),
        ("win_rate_pct", "Win Rate"),
        ("return_pct", "Return %"),
        ("sharpe_ratio", "Sharpe"),
        ("max_drawdown_pct", "Max DD %"),
        ("profit_factor", "Profit Factor"),
        ("total_pnl", "Total PnL"),
    ]

    for key, label in metrics_keys:
        row = f"{label:<25}"
        for variant in ["lgbm_only", "gru_only", "combined"]:
            val = comparison[variant]["metrics"].get(key, "N/A")
            total_trades = comparison[variant]["metrics"].get("num_trades", 0)
            if total_trades == 0 and key != "num_trades":
                row += f" {'—':>14}"
            elif isinstance(val, float):
                row += f" {val:>13.4f}"
            else:
                row += f" {str(val):>14}"
        logger.info(row)

    logger.info("-" * 70)
    for variant in ["lgbm_only", "gru_only", "combined"]:
        fc = comparison[variant]["feature_count"]
        logger.info("%s feature count: %d", variant, fc)


def _determine_best(comparison: dict) -> str:
    """Determine best variant by Sharpe ratio, breaking ties by trades."""
    scores: dict[str, tuple[float, int]] = {}
    for variant in ["lgbm_only", "gru_only", "combined"]:
        m = comparison[variant]["metrics"]
        sharpe = m.get("sharpe_ratio", 0)
        trades = m.get("num_trades", 0)
        scores[variant] = (sharpe, trades)

    best = max(scores, key=lambda k: scores[k])
    return best


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_ablation(config: Config) -> None:
    """Run all three ablation variants and save comparison results.

    GRU is trained **once** and shared across GRU-only and Combined variants
    to avoid ~10 min of redundant training.

    Args:
        config: Loaded application configuration.
    """
    import torch

    train_path = Path(config.paths.train_data)
    val_path = Path(config.paths.val_data)
    test_path = Path(config.paths.test_data)

    for p in (train_path, val_path, test_path):
        if not p.exists():
            raise FileNotFoundError(f"Split data not found: {p}")

    train_df = pl.read_parquet(train_path)
    val_df = pl.read_parquet(val_path)
    test_df = pl.read_parquet(test_path)

    logger.info(
        "Loaded splits — train: %d, val: %d, test: %d",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    session_dir = (
        Path(config.paths.session_dir) if config.paths.session_dir else Path("results")
    )

    logger.info("=" * 60)
    logger.info("ABLATION STUDY: 3 Variants")
    logger.info("=" * 60)

    # ==================================================================
    # Shared: Train GRU once (used by variants B and C)
    # ==================================================================
    logger.info("=== Training GRU (shared by variants B + C) ===")
    gru_model, gru_classifier, train_hidden, val_hidden = train_gru(
        config, train_df, val_df
    )

    gru_cols = ["log_returns", "rsi_14", "atr_14", "macd_hist"]
    test_seq, _, _ = prepare_sequences(test_df, gru_cols, config.gru.sequence_length)
    test_hidden = extract_hidden_states(gru_model, test_seq, config.gru.batch_size)

    seq_len = config.gru.sequence_length
    train_aligned = train_df.slice(seq_len - 1, len(train_hidden))
    val_aligned = val_df.slice(seq_len - 1, len(val_hidden))
    test_aligned = test_df.slice(seq_len - 1, len(test_hidden))

    y_train_aligned = train_aligned["label"].to_numpy().astype(np.int32)
    y_val_aligned = val_aligned["label"].to_numpy().astype(np.int32)
    y_test_aligned = test_aligned["label"].to_numpy().astype(np.int32)

    # GRU softmax probabilities via trained classifier head (shared by variant B)
    gru_model.eval()
    gru_classifier.eval()
    with torch.no_grad():
        device = next(gru_model.parameters()).device
        test_tensor = torch.tensor(test_seq, dtype=torch.float32, device=device)
        gru_hidden_test = gru_model(test_tensor)
        gru_logits = gru_classifier(gru_hidden_test)
        gru_softmax = torch.softmax(gru_logits, dim=1).cpu().numpy()

    static_cols = [c for c in train_aligned.columns if c not in _EXCLUDE_COLS]
    hidden_size = config.gru.hidden_size

    # ==================================================================
    # Variant A: LightGBM-only (no GRU)
    # ==================================================================
    logger.info("=== VARIANT A: LightGBM-only (static features) ===")
    logger.info("Static features (%d): %s", len(static_cols), static_cols)

    X_train_a = train_df.select(static_cols).to_numpy()
    X_val_a = val_df.select(static_cols).to_numpy()
    X_test_a = test_df.select(static_cols).to_numpy()
    y_train_a = train_df["label"].to_numpy().astype(np.int32)
    y_val_a = val_df["label"].to_numpy().astype(np.int32)
    y_test_a = test_df["label"].to_numpy().astype(np.int32)

    cw_a = _compute_class_weights(y_train_a)
    model_a = _train_lgbm(
        X_train_a, y_train_a, X_val_a, y_val_a, cw_a, config, static_cols
    )
    proba_a = model_a.predict_proba(_wrap_np(X_test_a, static_cols))
    preds_a = np.argmax(proba_a, axis=1) - 1
    logger.info("LightGBM-only accuracy: %.4f", (preds_a == y_test_a).mean())

    preds_path_a = session_dir / "predictions" / "ablation_lgbm_only.parquet"
    _save_preds(test_df["timestamp"], y_test_a, preds_a, proba_a, preds_path_a)
    preds_df_a = pl.read_parquet(preds_path_a)
    metrics_a = _run_backtest_for(test_df, preds_df_a, config, "LightGBM-only")
    result_a = {"metrics": metrics_a, "feature_count": len(static_cols)}

    # ==================================================================
    # Variant B: GRU-only (direct softmax, no LightGBM)
    # ==================================================================
    logger.info("=== VARIANT B: GRU-only (direct predictions) ===")
    gru_preds = np.argmax(gru_softmax, axis=1)
    trading_signals = gru_preds - 1  # Map 0,1,2 → -1,0,1
    logger.info("GRU-only accuracy: %.4f", (trading_signals == y_test_aligned).mean())

    preds_path_b = session_dir / "predictions" / "ablation_gru_only.parquet"
    _save_preds(
        test_aligned["timestamp"],
        y_test_aligned,
        trading_signals,
        gru_softmax,
        preds_path_b,
    )
    preds_df_b = pl.read_parquet(preds_path_b)
    metrics_b = _run_backtest_for(test_df, preds_df_b, config, "GRU-only")
    result_b = {"metrics": metrics_b, "feature_count": hidden_size}

    # ==================================================================
    # Variant C: Combined (GRU hidden + static → LightGBM)
    # ==================================================================
    logger.info("=== VARIANT C: Combined (GRU + LightGBM) ===")
    gru_feat_names = [f"gru_h{i}" for i in range(hidden_size)]
    all_feature_cols = gru_feat_names + static_cols

    X_train_c = np.concatenate(
        [train_hidden, train_aligned.select(static_cols).to_numpy()], axis=1
    )
    X_val_c = np.concatenate(
        [val_hidden, val_aligned.select(static_cols).to_numpy()], axis=1
    )
    X_test_c = np.concatenate(
        [test_hidden, test_aligned.select(static_cols).to_numpy()], axis=1
    )

    logger.info(
        "Hybrid features: %d (%d GRU + %d static)",
        len(all_feature_cols),
        hidden_size,
        len(static_cols),
    )

    cw_c = _compute_class_weights(y_train_aligned)
    model_c = _train_lgbm(
        X_train_c,
        y_train_aligned,
        X_val_c,
        y_val_aligned,
        cw_c,
        config,
        all_feature_cols,
    )
    proba_c = model_c.predict_proba(_wrap_np(X_test_c, all_feature_cols))
    preds_c = np.argmax(proba_c, axis=1) - 1
    logger.info("Combined accuracy: %.4f", (preds_c == y_test_aligned).mean())

    preds_path_c = session_dir / "predictions" / "ablation_combined.parquet"
    _save_preds(
        test_aligned["timestamp"], y_test_aligned, preds_c, proba_c, preds_path_c
    )
    preds_df_c = pl.read_parquet(preds_path_c)
    metrics_c = _run_backtest_for(test_df, preds_df_c, config, "Combined")
    result_c = {"metrics": metrics_c, "feature_count": len(all_feature_cols)}

    # ==================================================================
    # Compare and save
    # ==================================================================
    comparison = {
        "lgbm_only": result_a,
        "gru_only": result_b,
        "combined": result_c,
    }

    _log_comparison(comparison)

    best_variant = _determine_best(comparison)
    comparison["comparison_note"] = (
        f"Best variant: {best_variant}. "
        f"Combined achieves hybrid temporal+static feature learning. "
        f"LightGBM-only lacks temporal context. "
        f"GRU-only lacks rich static features."
    )

    out_path = session_dir / "reports" / "ablation_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    logger.info("Ablation results saved: %s", out_path)
