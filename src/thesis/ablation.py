"""Ablation study: LightGBM-only vs GRU-only vs Combined (GRU + LightGBM).

Runs three variants on the same data splits and compares trading performance.
GRU is trained once and shared across variants B and C to avoid redundant runs.

Variants:
    A. LightGBM-only  - static features only (no GRU hidden states)
    B. GRU-only       - direct softmax predictions (no LightGBM)
    C. Combined       - GRU hidden states + static features -> LightGBM

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
import pandas as pd
import polars as pl

from thesis.config import Config
from thesis.gru import extract_hidden_states, prepare_sequences, train_gru
from thesis.hybrid.lgbm import _EXCLUDE_COLS, _compute_class_weights, _wrap_np

logger = logging.getLogger("thesis.ablation")

_SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60


def _train_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weights: dict[int, float],
    config: Config,
    feature_cols: list[str],
) -> Any:
    """
    Train a LightGBM multiclass classifier with config hyperparameters.

    Creates and trains a LGBMClassifier with parameters from config.model,
    applies class weighting for imbalanced data, and uses early stopping
    based on validation set performance.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        X_val: Validation feature matrix.
        y_val: Validation labels.
        class_weights: Balanced class weights computed from training labels.
        config: Application configuration with model and workflow settings.
        feature_cols: List of feature column names for wrapping X arrays.

    Returns:
        Trained LGBMClassifier model with best iteration loaded.
    """
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
    class_order: list[int] | None = None,
) -> None:
    """
    Write prediction parquet with standardized probability columns.

    Converts raw model outputs (labels and probabilities) into a consistent
    DataFrame format with standardized column names for downstream processing.

    Args:
        timestamps: Series of timestamps corresponding to each prediction.
        y_true: Ground truth labels.
        preds: Predicted labels.
        proba: Prediction probabilities (shape: n_samples x n_classes).
        path: Output path for the parquet file.
        class_order: Optional ordering of class indices; defaults to [-1, 0, 1].
    """
    if class_order is None:
        class_order = [-1, 0, 1]

    col_mapping = {label: proba[:, idx] for idx, label in enumerate(class_order)}
    standardized_cols = {
        -1: "pred_proba_class_minus1",
        0: "pred_proba_class_0",
        1: "pred_proba_class_1",
    }
    proba_columns = {
        standardized_cols[label]: col_mapping[label]
        for label in class_order
        if label in standardized_cols
    }

    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "true_label": y_true.astype(np.int32),
            "pred_label": preds.astype(np.int32),
            **proba_columns,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    logger.info("Saved predictions: %s (%d rows)", path, len(df))


def _time_span_years(df: pl.DataFrame) -> float:
    """
    Compute covered time span in years from a split dataframe.

    Calculates the elapsed time between the first and last timestamp
    in the dataframe, expressed in years. Used for annualizing Sharpe ratio.

    Args:
        df: DataFrame with a 'timestamp' column.

    Returns:
        Time span in years, or 0.0 if insufficient data.
    """
    if "timestamp" not in df.columns or len(df) < 2:
        return 0.0

    start_ts = df["timestamp"].min()
    end_ts = df["timestamp"].max()
    if start_ts is None or end_ts is None:
        return 0.0

    seconds = (pd.Timestamp(end_ts) - pd.Timestamp(start_ts)).total_seconds()
    if seconds <= 0:
        return 0.0
    return float(seconds / _SECONDS_PER_YEAR)


def _compute_annualized_sharpe(
    trades_df: pd.DataFrame,
    test_df: pl.DataFrame,
) -> tuple[float, float]:
    """
    Compute annualized Sharpe from per-trade returns and trades per year.

    Calculates the Sharpe ratio using trade-level returns, annualized by
    the number of trades per year (derived from the test set time span).

    Args:
        trades_df: DataFrame with 'ReturnPct' column containing per-trade returns.
        test_df: DataFrame with 'timestamp' column to compute the time span.

    Returns:
        Tuple of (annualized_sharpe, trades_per_year). Returns (0.0, 0) if
        insufficient data or invalid standard deviation.
    """
    if trades_df.empty or "ReturnPct" not in trades_df.columns:
        return 0.0, 0.0

    returns = trades_df["ReturnPct"].to_numpy(dtype=float)
    returns = returns[np.isfinite(returns)]
    n_trades = int(len(returns))
    if n_trades == 0:
        return 0.0, 0.0

    years = _time_span_years(test_df)
    trades_per_year = (n_trades / years) if years > 0 else float(n_trades)

    if n_trades < 2:
        return 0.0, float(trades_per_year)

    std_ret = float(np.std(returns, ddof=1))
    if std_ret <= 1e-12:
        return 0.0, float(trades_per_year)

    mean_ret = float(np.mean(returns))
    sharpe = (mean_ret / std_ret) * np.sqrt(trades_per_year)
    return float(sharpe), float(trades_per_year)


def _run_backtest_for(
    test_df: pl.DataFrame,
    preds_df: pl.DataFrame,
    config: Config,
    variant: str,
) -> dict:
    """
    Run backtest and compute annualized Sharpe ratio.

    Executes the backtesting.py simulation on the test set using predictions
    from a specific variant (LGBM-only, GRU-only, or Combined), then
    overrides the Sharpe ratio with a properly annualized trade-based calculation.

    Args:
        test_df: Test set DataFrame with features and labels.
        preds_df: DataFrame with predictions and probabilities.
        config: Application configuration for backtest parameters.
        variant: Name identifier for this ablation variant (for logging).

    Returns:
        Dictionary with backtest metrics including annualized Sharpe.
    """
    from thesis.backtest.runners import _prepare_df, _run_bt
    from thesis.backtest.stats import _normalize_stats

    test = test_df.with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
    preds = preds_df.with_columns(pl.col("timestamp").cast(pl.Datetime("us")))

    logger.info("%s: running backtest", variant)

    pdf = _prepare_df(test, preds)
    stats, _ = _run_bt(pdf, config)
    metrics = _normalize_stats(stats, initial_capital=config.backtest.initial_capital)

    trades_df = stats["_trades"]
    annual_sharpe, trades_per_year = _compute_annualized_sharpe(trades_df, test)
    raw_sharpe = float(metrics.get("sharpe_ratio", 0) or 0)

    metrics["sharpe_ratio_raw"] = raw_sharpe
    metrics["sharpe_ratio"] = annual_sharpe
    metrics["annualized_sharpe_ratio"] = annual_sharpe
    metrics["trades_per_year"] = trades_per_year

    logger.info(
        "%s: Sharpe(raw)=%.4f, Sharpe(annualized)=%.4f, trades/year=%.2f",
        variant,
        raw_sharpe,
        annual_sharpe,
        trades_per_year,
    )

    return metrics


def _log_comparison(comparison: dict[str, dict]) -> None:
    """
    Log a formatted table comparing ablation metrics for each variant.

    Logs a human-readable table showing key metrics (trades, win rate, return,
    Sharpe, drawdown, profit factor) for all three variants: LGBM-only, GRU-only,
    and Combined.

    Args:
        comparison: Dictionary keyed by variant name ('lgbm_only', 'gru_only', 'combined'),
            each containing a 'metrics' dict and 'feature_count'.
    """
    logger.info("=" * 78)
    logger.info("ABLATION COMPARISON")
    logger.info("=" * 78)

    header = f"{'Metric':<25} {'LGBM-only':>16} {'GRU-only':>16} {'Combined':>16}"
    logger.info(header)
    logger.info("-" * 78)

    metrics_keys = [
        ("num_trades", "Trades"),
        ("trades_per_year", "Trades/Year"),
        ("win_rate_pct", "Win Rate"),
        ("return_pct", "Return %"),
        ("sharpe_ratio", "Sharpe (Ann.)"),
        ("max_drawdown_pct", "Max DD %"),
        ("profit_factor", "Profit Factor"),
        ("total_pnl", "Total PnL"),
    ]

    for key, label in metrics_keys:
        row = f"{label:<25}"
        for variant in ["lgbm_only", "gru_only", "combined"]:
            val = comparison[variant]["metrics"].get(key, "N/A")
            total_trades = comparison[variant]["metrics"].get("num_trades", 0)
            if total_trades == 0 and key not in {"num_trades", "trades_per_year"}:
                row += f" {'-':>16}"
            elif isinstance(val, float):
                row += f" {val:>15.4f}"
            else:
                row += f" {str(val):>16}"
        logger.info(row)

    logger.info("-" * 78)
    for variant in ["lgbm_only", "gru_only", "combined"]:
        fc = comparison[variant]["feature_count"]
        logger.info("%s feature count: %d", variant, fc)


def _determine_best(comparison: dict[str, dict]) -> str:
    """
    Select best variant by annualized Sharpe ratio, then number of trades.

    Compares the annualized Sharpe ratios across all three variants and returns
    the name of the best performer. If Sharpe ratios are equal, the variant
    with more trades is preferred (better statistical significance).

    Args:
        comparison: Dictionary keyed by variant name, each containing 'metrics'
            with 'sharpe_ratio' and 'num_trades'.

    Returns:
        Name of the best variant ('lgbm_only', 'gru_only', or 'combined').
    """
    scores: dict[str, tuple[float, int]] = {}
    for variant in ["lgbm_only", "gru_only", "combined"]:
        m = comparison[variant]["metrics"]
        sharpe_ann = float(m.get("sharpe_ratio", 0) or 0)
        trades = int(m.get("num_trades", 0) or 0)
        scores[variant] = (sharpe_ann, trades)
    return max(scores, key=lambda k: scores[k])


def _run_variant_a(
    test_aligned: pl.DataFrame,
    y_test_aligned: np.ndarray,
    train_aligned: pl.DataFrame,
    val_aligned: pl.DataFrame,
    y_train_aligned: np.ndarray,
    y_val_aligned: np.ndarray,
    static_cols: list[str],
    config: Config,
    session_dir: Path,
) -> dict:
    """Run variant A: LightGBM-only (static features)."""
    logger.info("=== VARIANT A: LightGBM-only (static features) ===")
    X_train_a = train_aligned.select(static_cols).to_numpy()
    X_val_a = val_aligned.select(static_cols).to_numpy()
    X_test_a = test_aligned.select(static_cols).to_numpy()

    cw_a = _compute_class_weights(y_train_aligned)
    model_a = _train_lgbm(
        X_train_a, y_train_aligned, X_val_a, y_val_aligned, cw_a, config, static_cols
    )
    proba_a = model_a.predict_proba(_wrap_np(X_test_a, static_cols))
    preds_a = model_a.classes_[np.argmax(proba_a, axis=1)]
    logger.info("LightGBM-only accuracy: %.4f", (preds_a == y_test_aligned).mean())

    preds_path_a = session_dir / "predictions" / "ablation_lgbm_only.parquet"
    _save_preds(
        test_aligned["timestamp"],
        y_test_aligned,
        preds_a,
        proba_a,
        preds_path_a,
        class_order=[int(x) for x in model_a.classes_.tolist()],
    )
    preds_df_a = pl.read_parquet(preds_path_a)
    metrics_a = _run_backtest_for(test_aligned, preds_df_a, config, "LightGBM-only")
    return {"metrics": metrics_a, "feature_count": len(static_cols)}


def _run_variant_b(
    test_aligned: pl.DataFrame,
    y_test_aligned: np.ndarray,
    gru_softmax: np.ndarray,
    hidden_size: int,
    config: Config,
    session_dir: Path,
) -> dict:
    """Run variant B: GRU-only (direct softmax predictions)."""
    logger.info("=== VARIANT B: GRU-only (direct predictions) ===")
    gru_preds = np.argmax(gru_softmax, axis=1)
    trading_signals = gru_preds - 1  # 0,1,2 -> -1,0,1
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
    metrics_b = _run_backtest_for(test_aligned, preds_df_b, config, "GRU-only")
    return {"metrics": metrics_b, "feature_count": hidden_size}


def _run_variant_c(
    test_aligned: pl.DataFrame,
    y_test_aligned: np.ndarray,
    train_hidden: np.ndarray,
    val_hidden: np.ndarray,
    test_hidden: np.ndarray,
    train_aligned: pl.DataFrame,
    val_aligned: pl.DataFrame,
    y_train_aligned: np.ndarray,
    y_val_aligned: np.ndarray,
    static_cols: list[str],
    hidden_size: int,
    config: Config,
    session_dir: Path,
) -> dict:
    """Run variant C: Combined (GRU hidden states + static features -> LightGBM)."""
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
    preds_c = model_c.classes_[np.argmax(proba_c, axis=1)]
    logger.info("Combined accuracy: %.4f", (preds_c == y_test_aligned).mean())

    preds_path_c = session_dir / "predictions" / "ablation_combined.parquet"
    _save_preds(
        test_aligned["timestamp"],
        y_test_aligned,
        preds_c,
        proba_c,
        preds_path_c,
        class_order=[int(x) for x in model_c.classes_.tolist()],
    )
    preds_df_c = pl.read_parquet(preds_path_c)
    metrics_c = _run_backtest_for(test_aligned, preds_df_c, config, "Combined")
    return {"metrics": metrics_c, "feature_count": len(all_feature_cols)}


def run_ablation(config: Config) -> None:
    """Run all ablation variants and save comparison results to JSON."""
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
        "Loaded splits - train: %d, val: %d, test: %d",
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

    # Shared: train GRU once (used by variants B + C)
    logger.info("=== Training GRU (shared by variants B + C) ===")
    gru_model, gru_classifier, train_hidden, val_hidden, _gru_history = train_gru(
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

    # GRU softmax probabilities for variant B
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

    # Run variants
    result_a = _run_variant_a(
        test_aligned,
        y_test_aligned,
        train_aligned,
        val_aligned,
        y_train_aligned,
        y_val_aligned,
        static_cols,
        config,
        session_dir,
    )
    result_b = _run_variant_b(
        test_aligned,
        y_test_aligned,
        gru_softmax,
        hidden_size,
        config,
        session_dir,
    )
    result_c = _run_variant_c(
        test_aligned,
        y_test_aligned,
        train_hidden,
        val_hidden,
        test_hidden,
        train_aligned,
        val_aligned,
        y_train_aligned,
        y_val_aligned,
        static_cols,
        hidden_size,
        config,
        session_dir,
    )

    comparison = {
        "lgbm_only": result_a,
        "gru_only": result_b,
        "combined": result_c,
    }

    _log_comparison(comparison)

    best_variant = _determine_best(comparison)
    comparison["comparison_note"] = (
        f"Best variant by annualized Sharpe: {best_variant}. "
        "Sharpe is annualized from per-trade ReturnPct using the observed "
        "trade frequency (trades/year) on the test window."
    )

    out_path = session_dir / "reports" / "ablation_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    logger.info("Ablation results saved: %s", out_path)
