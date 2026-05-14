"""Report generation orchestrator.

Loads data, computes metrics, renders charts, and persists markdown reports.
No I/O inside renderers — all data flows through ReportData payload.
"""

from __future__ import annotations

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any

import polars as pl
from polars.exceptions import ColumnNotFoundError, ComputeError

from thesis.shared.config import Config
from thesis.shared.ui import console
from thesis.stage_6_reporting import model_metrics
from thesis.stage_6_reporting.benchmarks import model_label
from thesis.stage_6_reporting.charts import (
    load_feature_importance,
    plot_equity_curve,
    plot_feature_importance,
)
from thesis.stage_6_reporting.comparison import (
    build_model_comparison_rows,
    write_model_comparison_artifacts,
)
from thesis.stage_6_reporting.sections import (
    render_baseline_comparison_section,
    render_data_quality_section,
    render_label_design_section,
    render_metric_zones_section,
    render_oof_vs_oos_section,
    render_validation_methodology_section,
)
from thesis.stage_6_reporting.tables import (
    accuracy_table,
    backtest_metrics_table,
    backtest_params_table,
    benchmark_comparison_table,
    config_table,
    exec_table,
    exec_verdict,
    feature_importance_table,
    issues_list,
)

logger = logging.getLogger("thesis.report")

HIGH_CONFIDENCE_THRESHOLD: float = 0.70
DIRECTIONAL_BASELINE: float = 0.5


def load_prediction_stats(preds_path: Path) -> dict | None:
    """Compute prediction quality statistics from predictions CSV."""
    if not preds_path.exists():
        return None
    try:
        df = pl.read_csv(preds_path)
        true = df["true_label"].to_numpy()
        pred = df["pred_label"].to_numpy()

        proba_cols = [
            "pred_proba_class_minus1",
            "pred_proba_class_0",
            "pred_proba_class_1",
        ]
        proba = (
            df.select(proba_cols).to_numpy()
            if all(c in df.columns for c in proba_cols)
            else None
        )

        raw_metrics = model_metrics.compute_all_classification_metrics(
            true, pred, y_proba=proba
        )
        per_class_metrics = raw_metrics["precision_recall_f1_per_class"]
        class_map = {-1: "Short", 0: "Hold", 1: "Long"}
        per_class_counts = {
            class_map[c]: {
                "true_count": int((true == c).sum()),
                "pred_count": int((pred == c).sum()),
                "precision": float(per_class_metrics[class_map[c]]["precision"]),
                "recall": float(per_class_metrics[class_map[c]]["recall"]),
                "f1": float(per_class_metrics[class_map[c]]["f1"]),
            }
            for c in (-1, 0, 1)
        }

        result: dict[str, Any] = {
            "total": int(raw_metrics["total"]),
            "accuracy": float(raw_metrics["accuracy"]),
            "balanced_accuracy": float(raw_metrics["balanced_accuracy"]),
            "directional_accuracy": float(raw_metrics["directional_accuracy"]),
            "directional_baseline": DIRECTIONAL_BASELINE,
            "majority_baseline": float(raw_metrics["majority_baseline_accuracy"]),
            "macro_f1": float(raw_metrics["macro_f1"]),
            "weighted_f1": float(raw_metrics["weighted_f1"]),
            "per_class": per_class_counts,
            "confusion_matrix": raw_metrics["confusion_matrix"],
            "direction_confusion_matrix": raw_metrics["direction_confusion_matrix"],
        }

        if proba is not None:
            max_proba = proba.max(axis=1)
            hc_mask = max_proba >= HIGH_CONFIDENCE_THRESHOLD
            hc_count = int(hc_mask.sum())
            hc_acc = float((true[hc_mask] == pred[hc_mask]).mean()) if hc_count else 0.0
            hc_non_hold = hc_mask & (pred != 0)
            hc_non_hold_count = int(hc_non_hold.sum())
            hc_dir_acc = (
                float((true[hc_non_hold] == pred[hc_non_hold]).mean())
                if hc_non_hold_count
                else 0.0
            )
            result["high_confidence"] = {
                "threshold": HIGH_CONFIDENCE_THRESHOLD,
                "count": hc_count,
                "pct_of_total": hc_count / len(true) if len(true) else 0.0,
                "accuracy": hc_acc,
                "directional_accuracy": hc_dir_acc,
            }
        return result
    except (ComputeError, ColumnNotFoundError, OSError):
        logger.warning(
            "Failed to load prediction statistics: %s", preds_path, exc_info=True
        )
        return None


def build_markdown(
    config: Config,
    metrics: dict,
    trades: list[dict],
    feature_importance: dict,
    pred_stats: dict | None,
) -> str:
    """Build concise metrics-first markdown report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    session = config.paths.session_dir or "N/A"
    L: list[str] = []

    L.append(f"# Thesis Report: {model_label(config)} — XAU/USD")
    L.append("")
    L.append(f"> Generated: {now} | Session: `{session}`")
    L.append("")

    L.append("## Executive Summary")
    L.append("")
    exec_table(L, metrics, pred_stats)
    exec_verdict(L, metrics, pred_stats)
    L.append("")

    L.append("## Methodology")
    L.append("")
    render_data_quality_section(L, config, heading="### Data & Quality")
    render_label_design_section(L, config, heading="### Label Design")
    render_validation_methodology_section(L, config, heading="### Validation Scheme")

    L.append("### Model Architecture")
    L.append("")
    feature_importance_table(L, feature_importance)
    L.append("")

    L.append("## Classification Results ★")
    L.append("")
    L.append(
        "*Classification metrics are the primary evaluation criterion for "
        "this thesis. Directional Accuracy and Macro F1 measure the model's "
        "ability to predict market direction (Short / Hold / Long).*"
    )
    L.append("")
    accuracy_table(L, pred_stats, config)
    L.append("")

    render_baseline_comparison_section(L, config, heading="### Baseline Comparison")

    L.append("## Application Demo: Backtest")
    L.append("")
    L.append(
        "*Backtest results are presented as an application demo to illustrate "
        "how classification signals *could* be translated into trades. "
        "They are **not** the primary evaluation criterion.*"
    )
    L.append("")
    backtest_params_table(L, config)
    backtest_metrics_table(L, metrics, config)
    render_metric_zones_section(L, metrics, trades, heading="")
    L.append("")

    L.append("### Benchmark Comparison")
    L.append("")
    benchmark_comparison_table(L, metrics, config)

    render_oof_vs_oos_section(L, config, heading="## Generalization Assessment")

    L.append("## Issues & Recommendations")
    L.append("")
    issues_list(L, metrics, trades, config, pred_stats)
    L.append("")

    L.append("## Appendix: Full Configuration")
    L.append("")
    config_table(L, config)
    L.append("")

    return "\n".join(L)


def build_model_evaluation_markdown(
    config: Config, pred_stats: dict | None, model_comparison_rows: list[dict[str, Any]]
) -> str:
    """Build compact evaluation-first markdown artifact."""
    lines: list[str] = ["# Model Evaluation", ""]
    lines.append(
        "This file is the primary ML evidence artifact. "
        "Backtest metrics are intentionally excluded."
    )
    lines.append("")
    lines.append(f"- Model: {model_label(config)}")
    lines.append(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    if not pred_stats:
        lines.append("*Prediction statistics unavailable.*")
        return "\n".join(lines)

    def pct(v, default=0.0):
        return f"{float(v if v is not None else default) * 100:.2f}%"

    def f4(v, default=0.0):
        return f"{float(v if v is not None else default):.4f}"

    lines.append("## Classification Metrics (Primary)")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Accuracy | {pct(pred_stats.get('accuracy'))} |")
    lines.append(
        f"| Directional Accuracy | {pct(pred_stats.get('directional_accuracy'))} |"
    )
    lines.append(f"| Macro F1 | {f4(pred_stats.get('macro_f1'))} |")
    lines.append(f"| Balanced Accuracy | {pct(pred_stats.get('balanced_accuracy'))} |")
    lines.append("")

    lines.append("## Per-Class Metrics")
    lines.append("")
    lines.append("| Class | Precision | Recall | F1 |")
    lines.append("|---|---:|---:|---:|")
    for class_name in ("Short", "Hold", "Long"):
        pc = pred_stats.get("per_class", {}).get(class_name, {})
        lines.append(
            f"| {class_name} | {pc.get('precision', 0.0):.4f}"
            f" | {pc.get('recall', 0.0):.4f}"
            f" | {pc.get('f1', 0.0):.4f} |"
        )
    lines.append("")

    reg_aux = pred_stats.get("regression_auxiliary")
    if reg_aux:
        lines.append("## Regression Auxiliary Metrics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---:|")
        lines.append(f"| MAE Return | {reg_aux.get('mae', float('nan')):.6f} |")
        lines.append(f"| RMSE Return | {reg_aux.get('rmse', float('nan')):.6f} |")
        lines.append(f"| R2 Return | {reg_aux.get('r_squared', float('nan')):.6f} |")
        lines.append("")

    lines.append("## Model Comparison")
    lines.append("")
    lines.append(
        "| Model | Directional Acc | Accuracy | Macro F1 | Long F1 | Short F1 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")

    def cell(row: dict, key: str, fmt: str = "pct") -> str:
        v = row.get(key)
        if v is None:
            return ""
        return f"{float(v) * 100:.2f}%" if fmt == "pct" else f"{float(v):.4f}"

    for row in model_comparison_rows:
        lines.append(
            f"| {row.get('model', '')} | {cell(row, 'directional_accuracy')}"
            f" | {cell(row, 'accuracy')} | {cell(row, 'macro_f1', 'f4')}"
            f" | {cell(row, 'long_f1', 'f4')} | {cell(row, 'short_f1', 'f4')} |"
        )
    lines.append("")
    return "\n".join(lines)


class ReportData:
    """Structured report payload — all data needed to render and persist reports.

    Computed once by compute_report_data, then passed to rendering
    and persistence functions. Keeps I/O out of renderers.
    """

    __slots__ = (
        "metrics",
        "trades",
        "feature_importance",
        "pred_stats",
        "model_comparison_rows",
        "out_dir",
        "report_path",
    )

    def __init__(
        self,
        *,
        metrics: dict,
        trades: list[dict],
        feature_importance: dict,
        pred_stats: dict | None,
        model_comparison_rows: list[dict],
        out_dir: Path,
        report_path: Path,
    ) -> None:
        """Initialise report data payload."""
        self.metrics = metrics
        self.trades = trades
        self.feature_importance = feature_importance
        self.pred_stats = pred_stats
        self.model_comparison_rows = model_comparison_rows
        self.out_dir = out_dir
        self.report_path = report_path


def setup_matplotlib() -> None:
    """Configure matplotlib for headless report chart rendering."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
        try:
            plt.style.use(style)
            break
        except Exception:
            continue

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        }
    )


def compute_report_data(config: Config) -> ReportData:
    """Load data, compute metrics, render charts — return structured payload.

    No report files are written. Callers use the returned ReportData
    to build markdown and persist artifacts.
    """
    setup_matplotlib()

    out_dir = (
        Path(config.paths.session_dir) / "reports"
        if config.paths.session_dir
        else Path("results")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(config.paths.report)

    bt_path = Path(config.paths.backtest_results)
    metrics: dict = {}
    trades: list[dict] = []
    if bt_path.exists():
        with console.status(f"[cyan]Loading backtest results[/] {bt_path}"):
            with open(bt_path) as f:
                bt = json.load(f)
            metrics = bt.get("metrics", {})
            trades = bt.get("trades", [])

    with console.status("[cyan]Rendering report charts[/]"):
        plot_equity_curve(trades, config, out_dir)
        feature_importance = load_feature_importance(config, out_dir)
        plot_feature_importance(feature_importance, out_dir)

    with console.status("[cyan]Building thesis markdown[/]"):
        pred_stats = load_prediction_stats(Path(config.paths.predictions))
        model_comparison_rows = build_model_comparison_rows(config, pred_stats)

    return ReportData(
        metrics=metrics,
        trades=trades,
        feature_importance=feature_importance,
        pred_stats=pred_stats,
        model_comparison_rows=model_comparison_rows,
        out_dir=out_dir,
        report_path=report_path,
    )


def generate_report(config: Config) -> None:
    """Generate thesis report with static charts and markdown."""
    data = compute_report_data(config)

    md = build_markdown(
        config, data.metrics, data.trades, data.feature_importance, data.pred_stats
    )
    model_eval_md = build_model_evaluation_markdown(
        config, data.pred_stats, data.model_comparison_rows
    )

    data.report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(data.report_path, "w") as f:
        f.write(md)
    logger.info("Report saved: %s", data.report_path)

    model_eval_path = data.out_dir / "model_evaluation.md"
    with model_eval_path.open("w") as f:
        f.write(model_eval_md)
    logger.info("Model evaluation saved: %s", model_eval_path)

    model_metrics_path = data.out_dir / "model_metrics.json"
    with model_metrics_path.open("w") as f:
        json.dump(data.pred_stats or {}, f, indent=2)
    logger.info("Model metrics saved: %s", model_metrics_path)

    model_cmp_csv = write_model_comparison_artifacts(
        data.out_dir, data.model_comparison_rows
    )
    logger.info("Model comparison saved: %s", model_cmp_csv)
