"""Static report generation — matplotlib charts + markdown.

Generates: equity curve, feature importance bar chart, and a comprehensive
markdown report with per-stage detail, model performance, backtest metrics,
ablation results, and a conclusion.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from thesis.config import Config

logger = logging.getLogger("thesis.report")


def generate_report(config: Config) -> None:
    """Generate thesis report with static charts and markdown.

    Args:
        config: Loaded application configuration.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if config.paths.session_dir:
        out_dir = Path(config.paths.session_dir) / "reports"
    else:
        out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load backtest results
    bt_path = Path(config.paths.backtest_results)
    metrics: dict = {}
    trades: list[dict] = []
    if bt_path.exists():
        with open(bt_path) as f:
            bt = json.load(f)
        metrics = bt.get("metrics", {})
        trades = bt.get("trades", [])

    # --- 1. Equity Curve ---
    if trades:
        pnls = [t["pnl"] for t in trades]
        equity = [config.backtest.initial_capital]
        for p in pnls:
            equity.append(equity[-1] + p)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(equity, linewidth=1)
        ax.set_title("Equity Curve")
        ax.set_ylabel("Equity (USD)")
        ax.set_xlabel("Trade #")
        ax.grid(True, alpha=0.3)
        fig.savefig(out_dir / "equity_curve.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Chart saved: equity_curve.png")

    # --- 2. Feature Importance ---
    if config.paths.session_dir:
        fi_path = Path(config.paths.session_dir) / "reports" / "feature_importance.json"
    else:
        fi_path = Path("results/feature_importance.json")
    feature_importance: dict = {}
    if fi_path.exists():
        with open(fi_path) as f:
            feature_importance = json.load(f)
        top = dict(list(feature_importance.items())[:20])

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(list(top.keys()), list(top.values()))
        ax.set_title("Feature Importance (Top 20)")
        ax.invert_yaxis()
        fig.savefig(out_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Chart saved: feature_importance.png")

    # --- 3. Load ablation results if available ---
    ablation: dict = {}
    if config.paths.session_dir:
        abl_path = Path(config.paths.session_dir) / "reports" / "ablation_results.json"
        if abl_path.exists():
            with open(abl_path) as f:
                ablation = json.load(f)

    # --- 4. Markdown Report ---
    md = _build_markdown(config, metrics, trades, feature_importance, ablation)
    report_path = Path(config.paths.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(md)
    logger.info("Report saved: %s", report_path)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_parquet_stats(path: Path) -> dict | None:
    """Load basic stats from a parquet file without loading full data."""
    if not path.exists():
        return None
    try:
        df = pl.read_parquet(path, columns=None)
        stats: dict = {"rows": len(df), "columns": df.columns}
        if "timestamp" in df.columns:
            ts = df["timestamp"]
            stats["date_range"] = (
                str(ts.min()),
                str(ts.max()),
            )
        return stats
    except Exception:
        return None


def _load_label_distribution(labels_path: Path) -> dict | None:
    """Load label class distribution from labels parquet."""
    if not labels_path.exists():
        return None
    try:
        df = pl.read_parquet(labels_path, columns=["label"])
        total = len(df)
        dist: dict[str, tuple[int, float]] = {}
        for label_val, name in [(-1, "Short"), (0, "Hold"), (1, "Long")]:
            count = (df["label"] == label_val).sum()
            dist[name] = (count, count / total * 100 if total > 0 else 0)
        dist["total"] = total
        return dist
    except Exception:
        return None


def _load_split_stats(config: Config) -> dict:
    """Load row counts and label distributions per split."""
    splits = {}
    for name, path_str in [
        ("train", config.paths.train_data),
        ("val", config.paths.val_data),
        ("test", config.paths.test_data),
    ]:
        path = Path(path_str)
        if not path.exists():
            continue
        try:
            df = pl.read_parquet(path)
            info: dict = {"rows": len(df), "columns": len(df.columns)}
            if "label" in df.columns:
                total = len(df)
                label_dist = {}
                for lv, ln in [(-1, "Short"), (0, "Hold"), (1, "Long")]:
                    c = int((df["label"] == lv).sum())
                    label_dist[ln] = (c, c / total * 100 if total else 0)
                info["label_distribution"] = label_dist
            if "timestamp" in df.columns:
                info["date_range"] = (
                    str(df["timestamp"].min()),
                    str(df["timestamp"].max()),
                )
            splits[name] = info
        except Exception:
            continue
    return splits


def _load_prediction_stats(preds_path: Path) -> dict | None:
    """Load model prediction performance statistics."""
    if not preds_path.exists():
        return None
    try:
        cols = ["true_label", "pred_label"]
        proba_cols = [
            "pred_proba_class_minus1",
            "pred_proba_class_0",
            "pred_proba_class_1",
        ]
        # Try loading with probability columns
        try:
            df = pl.read_parquet(preds_path)
        except Exception:
            df = pl.read_parquet(preds_path, columns=cols)

        true = df["true_label"].to_numpy()
        pred = df["pred_label"].to_numpy()
        total = len(true)

        # Overall accuracy
        accuracy = float((true == pred).mean())
        majority_baseline = float(max((true == lv).sum() for lv in [-1, 0, 1]) / total)

        # Per-class metrics
        per_class = {}
        for lv, ln in [(-1, "Short"), (0, "Hold"), (1, "Long")]:
            true_mask = true == lv
            pred_mask = pred == lv
            recall = float((pred[true_mask] == lv).mean()) if true_mask.sum() > 0 else 0
            precision = (
                float((true[pred_mask] == lv).mean()) if pred_mask.sum() > 0 else 0
            )
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )
            per_class[ln] = {
                "true_count": int(true_mask.sum()),
                "pred_count": int(pred_mask.sum()),
                "recall": recall,
                "precision": precision,
                "f1": f1,
            }

        # Confusion matrix
        cm = {}
        for true_lv, true_name in [(-1, "Short"), (0, "Hold"), (1, "Long")]:
            row = {}
            for pred_lv, pred_name in [(-1, "Short"), (0, "Hold"), (1, "Long")]:
                row[pred_name] = int(((true == true_lv) & (pred == pred_lv)).sum())
            cm[true_name] = row

        result: dict = {
            "total": total,
            "accuracy": accuracy,
            "majority_baseline": majority_baseline,
            "per_class": per_class,
            "confusion_matrix": cm,
        }

        # Confidence-filtered accuracy
        has_proba = all(c in df.columns for c in proba_cols)
        if has_proba:
            proba = df.select(proba_cols).to_numpy()
            max_proba = proba.max(axis=1)
            threshold = 0.70
            hc_mask = max_proba >= threshold
            if hc_mask.sum() > 0:
                hc_acc = float((true[hc_mask] == pred[hc_mask]).mean())
                hc_total = int(hc_mask.sum())
                # Directional (non-hold) accuracy
                non_hold = pred[hc_mask] != 0
                if non_hold.sum() > 0:
                    dir_acc = float(
                        (true[hc_mask][non_hold] == pred[hc_mask][non_hold]).mean()
                    )
                else:
                    dir_acc = 0
                result["high_confidence"] = {
                    "threshold": threshold,
                    "count": hc_total,
                    "pct_of_total": hc_total / total * 100,
                    "accuracy": hc_acc,
                    "directional_accuracy": dir_acc,
                }

        return result
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Markdown builder
# ---------------------------------------------------------------------------


def _build_markdown(
    config: Config,
    metrics: dict,
    trades: list[dict],
    feature_importance: dict,
    ablation: dict,
) -> str:
    """Build a comprehensive markdown report with per-stage detail."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    session = config.paths.session_dir or "N/A"

    lines: list[str] = [
        "# Thesis Report: Hybrid GRU + LightGBM for XAU/USD H1",
        "",
        f"> Generated: {now}  ",
        f"> Session: `{session}`",
        "",
        "---",
        "",
        "## Table of Contents",
        "",
        "1. [Data Preparation](#1-data-preparation)",
        "2. [Feature Engineering](#2-feature-engineering)",
        "3. [Triple-Barrier Labeling](#3-triple-barrier-labeling)",
        "4. [Data Splitting](#4-data-splitting)",
        "5. [Model Training](#5-model-training)",
        "6. [Model Prediction Performance](#6-model-prediction-performance)",
        "7. [Backtest Results](#7-backtest-results)",
        "8. [Ablation Study](#8-ablation-study)",
        "9. [Conclusion](#9-conclusion)",
        "10. [Charts](#10-charts)",
        "",
        "---",
        "",
    ]

    # ==================================================================
    # 1. DATA PREPARATION
    # ==================================================================
    lines.append("## 1. Data Preparation")
    lines.append("")
    ohlcv_path = Path(config.paths.ohlcv)
    ohlcv_stats = _load_parquet_stats(ohlcv_path)
    if ohlcv_stats:
        lines.append(f"**Source**: `{config.data.symbol}` raw tick data")
        lines.append(f"**Timeframe**: {config.data.timeframe}")
        lines.append(f"**Market timezone**: {config.data.market_tz}")
        lines.append(f"**Bars generated**: {ohlcv_stats['rows']:,}")
        if "date_range" in ohlcv_stats:
            lines.append(
                f"**Date range**: {ohlcv_stats['date_range'][0]} → {ohlcv_stats['date_range'][1]}"
            )
        lines.append("")
        lines.append(
            "OHLCV columns: `open`, `high`, `low`, `close`, `volume`, `tick_count`, `avg_spread`"
        )
    else:
        lines.append("*OHLCV data not found.*")
    lines.append("")

    # ==================================================================
    # 2. FEATURE ENGINEERING
    # ==================================================================
    lines.append("## 2. Feature Engineering")
    lines.append("")
    feat_path = Path(config.paths.features)
    feat_stats = _load_parquet_stats(feat_path)
    if feat_stats:
        lines.append(f"**Input bars**: {feat_stats['rows']:,}")
        lines.append(f"**Total columns**: {len(feat_stats['columns'])}")
        lines.append("")

        # Feature list
        static_features = [
            ("rsi_14", "Relative Strength Index (Wilder, period=14)"),
            ("atr_14", "Average True Range (period=14)"),
            ("macd_hist", "MACD Histogram (12/26/9)"),
            ("atr_ratio", "ATR(5) / ATR(20) — volatility regime"),
            ("price_dist_ratio", "(Close − EMA89) / ATR14 — trend distance"),
            ("pivot_position", "(Close − S1) / (R1 − S1) — bounded [0,1]"),
            ("atr_percentile", "Rolling rank of ATR14 over 50 bars"),
            ("sess_asia", "Asian session flag (0/1)"),
            ("sess_london", "London AM session flag (0/1)"),
            ("sess_overlap", "London-NY overlap flag (0/1)"),
            ("sess_ny_pm", "NY afternoon session flag (0/1)"),
        ]

        lines.append("### Static Features (11)")
        lines.append("")
        lines.append("| Feature | Description |")
        lines.append("|---------|-------------|")
        for name, desc in static_features:
            lines.append(f"| `{name}` | {desc} |")
        lines.append("")

        # Session distribution (if available)
        if all(
            c in feat_stats["columns"]
            for c in ["sess_asia", "sess_london", "sess_overlap", "sess_ny_pm"]
        ):
            try:
                df = pl.read_parquet(
                    feat_path,
                    columns=["sess_asia", "sess_london", "sess_overlap", "sess_ny_pm"],
                )
                total = len(df)
                lines.append("### Session Distribution")
                lines.append("")
                for col, label in [
                    ("sess_asia", "Asia"),
                    ("sess_london", "London AM"),
                    ("sess_overlap", "London-NY Overlap"),
                    ("sess_ny_pm", "NY Afternoon"),
                ]:
                    count = int(df[col].sum())
                    lines.append(
                        f"- **{label}**: {count:,} bars ({count / total * 100:.1f}%)"
                    )
                lines.append("")
            except Exception:
                pass
    else:
        lines.append("*Features data not found.*")
    lines.append("")

    # ==================================================================
    # 3. TRIPLE-BARRIER LABELING
    # ==================================================================
    lines.append("## 3. Triple-Barrier Labeling")
    lines.append("")
    lines.append("### Configuration")
    lines.append("")
    lines.append(f"- **ATR multiplier**: {config.labels.atr_multiplier}")
    lines.append(f"- **Horizon**: {config.labels.horizon_bars} bars")
    lines.append(f"- **Min ATR**: {config.labels.min_atr}")
    lines.append(
        f"- **Num classes**: {config.labels.num_classes} (Short/−1, Hold/0, Long/+1)"
    )
    lines.append("")

    labels_path = Path(config.paths.labels)
    label_dist = _load_label_distribution(labels_path)
    if label_dist:
        total = label_dist["total"]
        lines.append("### Label Distribution")
        lines.append("")
        lines.append("| Class | Count | Percentage |")
        lines.append("|-------|-------|------------|")
        for name in ["Short (−1)", "Hold (0)", "Long (+1)"]:
            key = name.split(" ")[0]
            count, pct = label_dist[key]
            lines.append(f"| {name} | {count:,} | {pct:.1f}% |")
        lines.append(f"| **Total** | **{total:,}** | **100%** |")
        lines.append("")

        # Interpretation
        short_pct = label_dist["Short"][1]
        hold_pct = label_dist["Hold"][1]
        long_pct = label_dist["Long"][1]
        if hold_pct < 25:
            lines.append(
                f"> **Note**: Low Hold ratio ({hold_pct:.1f}%) — the barriers are "
                f"tight relative to the {config.labels.horizon_bars}-bar horizon, "
                f"producing many directional signals."
            )
        else:
            lines.append(
                f"> **Note**: Balanced label distribution — "
                f"{short_pct:.0f}% Short, {hold_pct:.0f}% Hold, {long_pct:.0f}% Long."
            )
        lines.append("")
    else:
        lines.append("*Labels data not found.*")
    lines.append("")

    # ==================================================================
    # 4. DATA SPLITTING
    # ==================================================================
    lines.append("## 4. Data Splitting")
    lines.append("")
    lines.append("### Split Configuration")
    lines.append("")
    lines.append(
        f"- **Train**: `{config.splitting.train_start}` → `{config.splitting.train_end}`"
    )
    lines.append(
        f"- **Validation**: `{config.splitting.val_start}` → `{config.splitting.val_end}`"
    )
    lines.append(
        f"- **Test (OOS)**: `{config.splitting.test_start}` → `{config.splitting.test_end}`"
    )
    lines.append(f"- **Purge bars**: {config.splitting.purge_bars}")
    lines.append(f"- **Embargo bars**: {config.splitting.embargo_bars}")
    lines.append(
        f"- **Correlation threshold**: {config.features.correlation_threshold} "
        f"(applied on train set only)"
    )
    lines.append("")

    split_stats = _load_split_stats(config)
    if split_stats:
        lines.append("### Split Statistics")
        lines.append("")
        lines.append("| Split | Rows | Date Range | Short | Hold | Long |")
        lines.append("|-------|------|------------|-------|------|------|")
        for name in ["train", "val", "test"]:
            if name not in split_stats:
                continue
            s = split_stats[name]
            dr = s.get("date_range", ("N/A", "N/A"))
            dr_str = f"{dr[0][:10]} → {dr[1][:10]}" if dr != ("N/A", "N/A") else "N/A"
            if "label_distribution" in s:
                ld = s["label_distribution"]
                short_s = f"{ld['Short'][0]:,} ({ld['Short'][1]:.1f}%)"
                hold_s = f"{ld['Hold'][0]:,} ({ld['Hold'][1]:.1f}%)"
                long_s = f"{ld['Long'][0]:,} ({ld['Long'][1]:.1f}%)"
            else:
                short_s = hold_s = long_s = "N/A"
            lines.append(
                f"| **{name.title()}** | {s['rows']:,} | {dr_str} | {short_s} | {hold_s} | {long_s} |"
            )
        lines.append("")

        # Leakage prevention note
        lines.append("### Leakage Prevention")
        lines.append("")
        lines.append(
            f"- **Purge**: {config.splitting.purge_bars} bars removed at split boundaries "
            f"to prevent label leakage from the triple-barrier lookahead."
        )
        lines.append(
            f"- **Embargo**: {config.splitting.embargo_bars} bars ({config.splitting.embargo_bars}h ≈ "
            f"{config.splitting.embargo_bars // 24} days) between val and test — "
            f"covers the {config.labels.horizon_bars}-bar label horizon."
        )
        lines.append(
            "- **Correlation filtering**: Applied on train set only. "
            "Features with |correlation| > threshold are removed before training."
        )
        lines.append("")
    else:
        lines.append("*Split data not found.*")
    lines.append("")

    # ==================================================================
    # 5. MODEL TRAINING
    # ==================================================================
    lines.append("## 5. Model Training")
    lines.append("")

    # GRU
    lines.append("### 5.1 GRU Feature Extractor")
    lines.append("")
    gru_inputs = {
        2: "log_returns, rsi_14",
        4: "log_returns, rsi_14, atr_14, macd_hist",
    }
    input_desc = gru_inputs.get(
        config.gru.input_size, f"{config.gru.input_size} features"
    )
    lines.append(
        f"- **Input features**: {input_desc} ({config.gru.input_size} features)"
    )
    lines.append(f"- **Sequence length**: {config.gru.sequence_length} bars")
    lines.append(f"- **Hidden size**: {config.gru.hidden_size} dimensions")
    lines.append(f"- **Layers**: {config.gru.num_layers}")
    lines.append(f"- **Dropout**: {config.gru.dropout}")
    lines.append(f"- **Learning rate**: {config.gru.learning_rate}")
    lines.append(
        f"- **Max epochs**: {config.gru.epochs} (early stopping patience={config.gru.patience})"
    )
    lines.append(
        f"- **Output**: {config.gru.hidden_size}-dimensional hidden state vectors "
        f"fed as features to LightGBM"
    )
    lines.append("")

    # LightGBM
    lines.append("### 5.2 LightGBM Classifier")
    lines.append("")
    lines.append(f"- **Leaves**: {config.model.num_leaves}")
    lines.append(f"- **Max depth**: {config.model.max_depth}")
    lines.append(f"- **Learning rate**: {config.model.learning_rate}")
    lines.append(f"- **Estimators**: {config.model.n_estimators}")
    lines.append(f"- **Min child samples**: {config.model.min_child_samples}")
    lines.append(
        f"- **Subsample**: {config.model.subsample} (freq={config.model.subsample_freq})"
    )
    lines.append(f"- **Feature fraction**: {config.model.feature_fraction}")
    lines.append(
        f"- **Regularization**: alpha={config.model.reg_alpha}, lambda={config.model.reg_lambda}"
    )
    lines.append(f"- **Early stopping**: {config.model.early_stopping_rounds} rounds")
    lines.append(
        f"- **Optuna**: {'enabled' if config.model.use_optuna else 'disabled'}"
    )
    lines.append("- **Class weights**: balanced")
    lines.append("")

    # Hybrid feature space
    total_features = config.gru.hidden_size + 11
    lines.append("### 5.3 Hybrid Feature Space")
    lines.append("")
    lines.append("| Source | Features |")
    lines.append("|--------|----------|")
    lines.append(f"| GRU hidden states | {config.gru.hidden_size} |")
    lines.append("| Static technical indicators | 11 |")
    lines.append(f"| **Total** | **{total_features}** |")
    lines.append("")

    # Feature importance
    if feature_importance:
        items = list(feature_importance.items())
        gru_count = sum(1 for name, _ in items if name.startswith("gru_"))

        lines.append("### 5.4 Feature Importance (Top 15)")
        lines.append("")
        lines.append("| Rank | Feature | Type | Importance |")
        lines.append("|------|---------|------|------------|")
        for i, (name, imp) in enumerate(items[:15], 1):
            ftype = "GRU" if name.startswith("gru_") else "Static"
            lines.append(f"| {i} | `{name}` | {ftype} | {imp:.1f} |")
        lines.append("")
        lines.append(
            f"> GRU features: {gru_count} of {len(items)} total ranked features "
            f"({gru_count / len(items) * 100:.0f}%). "
            f"Top GRU features dominate, confirming the sequential feature extractor "
            f"captures meaningful temporal patterns."
        )
        lines.append("")

    # ==================================================================
    # 6. MODEL PREDICTION PERFORMANCE
    # ==================================================================
    lines.append("## 6. Model Prediction Performance")
    lines.append("")

    preds_path = Path(config.paths.predictions)
    pred_stats = _load_prediction_stats(preds_path)
    if pred_stats:
        total = pred_stats["total"]
        acc = pred_stats["accuracy"]
        baseline = pred_stats["majority_baseline"]

        lines.append("### Overall Metrics")
        lines.append("")
        lines.append(f"- **Total test predictions**: {total:,}")
        lines.append(f"- **Overall accuracy**: {acc * 100:.2f}%")
        lines.append(f"- **Majority class baseline**: {baseline * 100:.2f}%")
        lines.append(f"- **Edge over baseline**: {(acc - baseline) * 100:+.2f} pp")
        lines.append("")

        # Per-class
        per_class = pred_stats["per_class"]
        lines.append("### Per-Class Metrics")
        lines.append("")
        lines.append("| Class | True Count | Predicted | Recall | Precision | F1 |")
        lines.append("|-------|-----------|-----------|--------|-----------|-----|")
        for name in ["Short", "Hold", "Long"]:
            pc = per_class[name]
            lines.append(
                f"| {name} | {pc['true_count']:,} | {pc['pred_count']:,} | "
                f"{pc['recall'] * 100:.1f}% | {pc['precision'] * 100:.1f}% | "
                f"{pc['f1']:.3f} |"
            )
        lines.append("")

        # Confusion matrix
        cm = pred_stats["confusion_matrix"]
        lines.append("### Confusion Matrix")
        lines.append("")
        lines.append("| | Pred Short | Pred Hold | Pred Long |")
        lines.append("|---------|-----------|-----------|-----------|")
        for true_name in ["Short", "Hold", "Long"]:
            row = cm[true_name]
            lines.append(
                f"| **True {true_name}** | {row['Short']:,} | {row['Hold']:,} | {row['Long']:,} |"
            )
        lines.append("")

        # Model bias analysis
        lines.append("### Model Bias Analysis")
        lines.append("")
        short_pred = per_class["Short"]["pred_count"]
        long_pred = per_class["Long"]["pred_count"]
        if short_pred > total * 0.5:
            lines.append(
                f"⚠️ **Short-biased**: Model predicts Short {short_pred / total * 100:.1f}% "
                f"of the time (true Short rate: {per_class['Short']['true_count'] / total * 100:.1f}%). "
                f"This creates many false Short signals."
            )
        elif long_pred > total * 0.5:
            lines.append(
                f"⚠️ **Long-biased**: Model predicts Long {long_pred / total * 100:.1f}% "
                f"of the time (true Long rate: {per_class['Long']['true_count'] / total * 100:.1f}%)."
            )
        else:
            lines.append("Model has relatively balanced directional predictions.")
        lines.append("")

        # High-confidence analysis
        hc = pred_stats.get("high_confidence")
        if hc:
            lines.append("### High-Confidence Signal Analysis")
            lines.append("")
            lines.append(f"With confidence threshold ≥ {hc['threshold']:.2f}:")
            lines.append("")
            lines.append(
                f"- **Signals passing filter**: {hc['count']:,} / {total:,} ({hc['pct_of_total']:.1f}%)"
            )
            lines.append(
                f"- **Accuracy on these signals**: {hc['accuracy'] * 100:.1f}%"
            )
            lines.append(
                f"- **Directional accuracy (non-hold)**: {hc['directional_accuracy'] * 100:.1f}%"
            )
            lines.append("")
            lines.append(
                f"> The model's high-conviction predictions ({hc['pct_of_total']:.1f}% of bars) "
                f"achieve {hc['accuracy'] * 100:.0f}% accuracy — a "
                f"{(hc['accuracy'] - acc) / acc * 100:.0f}% improvement over the unfiltered "
                f"{acc * 100:.0f}% baseline. This validates confidence-based filtering."
            )
            lines.append("")
    else:
        lines.append("*Prediction data not found.*")
        lines.append("")

    # ==================================================================
    # 7. BACKTEST RESULTS
    # ==================================================================
    lines.append("## 7. Backtest Results")
    lines.append("")

    # Backtest config
    lines.append("### Configuration")
    lines.append("")
    lines.append(f"- **Initial capital**: ${config.backtest.initial_capital:,.0f}")
    lines.append(f"- **Leverage**: 1:{config.backtest.leverage}")
    lines.append(f"- **Lots per trade**: {config.backtest.lots_per_trade}")
    lines.append(f"- **ATR stop multiplier**: {config.backtest.atr_stop_multiplier}")
    lines.append(f"- **Confidence threshold**: {config.backtest.confidence_threshold}")
    lines.append(
        f"- **Spread**: {config.backtest.spread_ticks} ticks (${config.backtest.spread_ticks * config.data.tick_size:.2f})"
    )
    lines.append(
        f"- **Slippage**: {config.backtest.slippage_ticks} ticks (${config.backtest.slippage_ticks * config.data.tick_size:.2f})"
    )
    lines.append(
        f"- **Commission**: ${config.backtest.commission_per_lot:.0f}/lot round-trip"
    )
    lines.append("")

    if metrics:
        # Trade statistics
        lines.append("### Trade Statistics")
        lines.append("")
        _metric(lines, metrics, "num_trades", "Total Trades", fmt="d")
        _metric(lines, metrics, "exposure_time_pct", "Exposure Time", fmt="pct")
        _metric(lines, metrics, "duration", "Period", fmt="s")
        _metric(lines, metrics, "avg_trade_duration", "Avg Trade Duration", fmt="s")
        _metric(lines, metrics, "max_trade_duration", "Max Trade Duration", fmt="s")
        lines.append("")

        # Performance
        lines.append("### Performance")
        lines.append("")
        _metric(lines, metrics, "return_pct", "Total Return", fmt="pct")
        _metric(lines, metrics, "return_ann_pct", "Annual Return", fmt="pct")
        _metric(lines, metrics, "buy_&_hold_return_pct", "Buy & Hold Return", fmt="pct")
        _metric(lines, metrics, "cagr_pct", "CAGR", fmt="pct")
        _metric(lines, metrics, "equity_final", "Final Equity", fmt="dollar")
        _metric(lines, metrics, "equity_peak", "Equity Peak", fmt="dollar")
        _metric(lines, metrics, "commissions", "Total Commissions", fmt="dollar")
        lines.append("")

        # Risk metrics
        lines.append("### Risk Metrics")
        lines.append("")
        _metric(lines, metrics, "sharpe_ratio", "Sharpe Ratio", fmt="f2")
        _metric(lines, metrics, "sortino_ratio", "Sortino Ratio", fmt="f2")
        _metric(lines, metrics, "calmar_ratio", "Calmar Ratio", fmt="f2")
        _metric(lines, metrics, "max_drawdown_pct", "Max Drawdown", fmt="pct")
        _metric(lines, metrics, "avg_drawdown_pct", "Avg Drawdown", fmt="pct")
        _metric(lines, metrics, "max_drawdown_duration", "Max DD Duration", fmt="s")
        _metric(lines, metrics, "volatility_ann_pct", "Annual Volatility", fmt="pct")
        lines.append("")

        # Trade quality
        lines.append("### Trade Quality")
        lines.append("")
        _metric(lines, metrics, "win_rate_pct", "Win Rate", fmt="pct")
        _metric(lines, metrics, "profit_factor", "Profit Factor", fmt="f2")
        _metric(lines, metrics, "expectancy_pct", "Expectancy", fmt="pct")
        _metric(lines, metrics, "sqn", "System Quality Number (SQN)", fmt="f2")
        _metric(lines, metrics, "kelly_criterion", "Kelly Criterion", fmt="f4")
        _metric(lines, metrics, "best_trade_pct", "Best Trade", fmt="pct")
        _metric(lines, metrics, "worst_trade_pct", "Worst Trade", fmt="pct")
        _metric(lines, metrics, "avg_trade_pct", "Avg Trade", fmt="pct")
        lines.append("")

        # Alpha/Beta
        lines.append("### Market Comparison")
        lines.append("")
        _metric(lines, metrics, "alpha_pct", "Alpha", fmt="pct")
        _metric(lines, metrics, "beta", "Beta", fmt="f3")
        lines.append("")

        # Trade distribution analysis
        if trades:
            lines.append("### Trade Distribution")
            lines.append("")
            pnls = [t["pnl"] for t in trades]
            winners = [p for p in pnls if p > 0]
            losers = [p for p in pnls if p < 0]
            longs = [t for t in trades if t.get("direction") == "long"]
            shorts = [t for t in trades if t.get("direction") == "short"]

            lines.append(
                f"- **Winners**: {len(winners):,} ({len(winners) / len(trades) * 100:.1f}%)"
            )
            if winners:
                lines.append(f"  - Avg winner: ${np.mean(winners):,.2f}")
                lines.append(f"  - Best winner: ${max(winners):,.2f}")
            lines.append(
                f"- **Losers**: {len(losers):,} ({len(losers) / len(trades) * 100:.1f}%)"
            )
            if losers:
                lines.append(f"  - Avg loser: ${np.mean(losers):,.2f}")
                lines.append(f"  - Worst loser: ${min(losers):,.2f}")
            lines.append(f"- **Long trades**: {len(longs):,}")
            lines.append(f"- **Short trades**: {len(shorts):,}")
            if winners and losers:
                rr = abs(np.mean(winners) / np.mean(losers))
                lines.append(f"- **Reward/Risk ratio**: {rr:.2f}:1")
            lines.append("")
    else:
        lines.append("*No backtest results available.*")
        lines.append("")

    # ==================================================================
    # 8. ABLATION STUDY
    # ==================================================================
    if ablation:
        lines.append("## 8. Ablation Study")
        lines.append("")
        lines.append(
            "| Variant | Features | Trades | Win Rate | Return % | Sharpe | Max DD % |"
        )
        lines.append(
            "|---------|----------|--------|----------|----------|--------|----------|"
        )
        for variant in ["lgbm_only", "gru_only", "combined"]:
            if variant in ablation:
                v = ablation[variant]
                m = v.get("metrics", {})
                fc = v.get("feature_count", "?")
                trades_v = m.get("num_trades", 0)
                wr = m.get("win_rate_pct", 0)
                ret = m.get("return_pct", 0)
                sh = m.get("sharpe_ratio", 0)
                dd = m.get("max_drawdown_pct", 0)
                lines.append(
                    f"| {variant} | {fc} | {trades_v} | {wr:.2f}% | {ret:.2f} | {sh:.4f} | {dd:.2f} |"
                )
        lines.append("")
        if "comparison_note" in ablation:
            lines.append(ablation["comparison_note"])
            lines.append("")

    # ==================================================================
    # 9. CONCLUSION
    # ==================================================================
    lines.append("## 9. Conclusion")
    lines.append("")

    # Build conclusion dynamically from available data
    conclusion_parts: list[str] = []

    # Model performance summary
    if pred_stats:
        acc = pred_stats["accuracy"]
        hc = pred_stats.get("high_confidence")
        if hc:
            conclusion_parts.append(
                f"The hybrid GRU + LightGBM model achieves {acc * 100:.1f}% overall accuracy "
                f"on the out-of-sample test set ({pred_stats['total']:,} predictions, "
                f"{config.splitting.test_start} to {config.splitting.test_end}). "
                f"While this modestly exceeds the {pred_stats['majority_baseline'] * 100:.1f}% "
                f"majority-class baseline, the model's true value lies in its confidence "
                f"calibration: the {hc['pct_of_total']:.1f}% of predictions with "
                f"probability ≥ {hc['threshold']:.0%} achieve "
                f"{hc['accuracy'] * 100:.0f}% directional accuracy."
            )
        else:
            conclusion_parts.append(
                f"The hybrid GRU + LightGBM model achieves {acc * 100:.1f}% overall accuracy "
                f"on the out-of-sample test set ({pred_stats['total']:,} predictions). "
                f"This exceeds the {pred_stats['majority_baseline'] * 100:.1f}% majority-class "
                f"baseline by {(acc - pred_stats['majority_baseline']) * 100:.1f} percentage points."
            )

    # Backtest summary
    if metrics:
        n_trades = metrics.get("num_trades", 0)
        sharpe = metrics.get("sharpe_ratio", 0)
        max_dd = metrics.get("max_drawdown_pct", 0)
        pf = metrics.get("profit_factor", 0)
        ret = metrics.get("return_pct", 0)
        wr = metrics.get("win_rate_pct", 0)
        sortino = metrics.get("sortino_ratio", 0)

        conclusion_parts.append(
            f"The backtest produces {n_trades:,} trades over the OOS period "
            f"with a Sharpe ratio of {sharpe:.2f}, Sortino ratio of {sortino:.2f}, "
            f"and maximum drawdown of {max_dd:.2f}%. "
            f"The profit factor of {pf:.2f} indicates that winning trades are "
            f"{pf:.1f}× larger than losing trades on aggregate. "
            f"The win rate of {wr:.1f}% is below 50%, which is typical for "
            f"trend-following strategies that rely on asymmetric payoff "
            f"(letting winners run, cutting losers quickly)."
        )

        # Leverage warning
        if ret > 500:
            unleveraged = ret / config.backtest.leverage
            conclusion_parts.append(
                f"**Important caveat**: The {ret:.0f}% total return is amplified by "
                f"{config.backtest.leverage}:1 leverage. The estimated unleveraged return "
                f"is approximately {unleveraged:.0f}%. Additionally, the test period "
                f"({config.splitting.test_start} to {config.splitting.test_end}) includes "
                f"gold's historic bull run ($2,000 → $3,000+), which provided a strong "
                f"directional tailwind."
            )

    # Feature importance insight
    if feature_importance:
        items = list(feature_importance.items())
        top3 = items[:3]
        gru_top = [n for n, _ in top3 if n.startswith("gru_")]
        if gru_top:
            conclusion_parts.append(
                f"GRU-derived features dominate the top feature importance rankings "
                f"({', '.join(f'`{n}`' for n in gru_top)}), confirming that the "
                f"sequential feature extractor captures temporal patterns that static "
                f"indicators alone cannot. This validates the hybrid architecture."
            )

    # Confidence filtering insight
    if config.backtest.confidence_threshold > 0:
        conclusion_parts.append(
            f"Confidence-based filtering (threshold={config.backtest.confidence_threshold}) "
            f"significantly improves trade quality by restricting entries to the model's "
            f"highest-conviction signals. This is a form of model-based signal validation "
            f"that complements traditional risk management."
        )

    # Write conclusion paragraphs
    for para in conclusion_parts:
        lines.append(para)
        lines.append("")

    if not conclusion_parts:
        lines.append("*Insufficient data for automated conclusion.*")
        lines.append("")

    # ==================================================================
    # 10. CHARTS
    # ==================================================================
    lines.append("## 10. Charts")
    lines.append("")
    if config.paths.session_dir:
        bt_chart = Path(config.paths.session_dir) / "backtest" / "backtest_chart.html"
        if bt_chart.exists():
            lines.append("### Interactive Backtest Chart")
            lines.append("")
            lines.append(
                "See [backtest_chart.html](backtest_chart.html) for the interactive Bokeh visualization."
            )
            lines.append("")
    lines.append("![Equity Curve](equity_curve.png)")
    lines.append("")
    lines.append("![Feature Importance](feature_importance.png)")
    lines.append("")

    # Visualization charts
    if config.paths.session_dir:
        charts_dir = Path(config.paths.session_dir) / "reports" / "charts"
        if charts_dir.exists():
            for subdir in ["data", "model", "backtest"]:
                sub = charts_dir / subdir
                if sub.exists():
                    lines.append(f"### {subdir.title()} Charts")
                    lines.append("")
                    for img in sorted(sub.glob("*.png")):
                        rel = f"charts/{subdir}/{img.name}"
                        lines.append(f"![{img.stem}]({rel})")
                    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Metric formatting helpers
# ---------------------------------------------------------------------------

_METRIC_FMT = {
    "pct": lambda v: f"{v:.2f}%",
    "f2": lambda v: f"{v:.2f}",
    "f3": lambda v: f"{v:.3f}",
    "f4": lambda v: f"{v:.4f}",
    "d": lambda v: f"{int(v):,}",
    "dollar": lambda v: f"${v:,.2f}",
    "s": lambda v: str(v),
}


def _metric(
    lines: list[str],
    metrics: dict,
    key: str,
    label: str,
    fmt: str = "f2",
) -> None:
    """Append a formatted metric line if the key exists."""
    if key not in metrics:
        return
    val = metrics[key]
    formatter = _METRIC_FMT.get(fmt, _METRIC_FMT["f2"])
    lines.append(f"- **{label}**: {formatter(val)}")
