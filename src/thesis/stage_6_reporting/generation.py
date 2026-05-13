"""Report generation — orchestrator, markdown builder, and statistics."""

from __future__ import annotations

from datetime import datetime
import json
import logging
import math
from pathlib import Path
from typing import Any

import polars as pl
from polars.exceptions import ColumnNotFoundError, ComputeError

from thesis.shared.config import Config
from thesis.shared.ui import console
from thesis.shared.zones import get_metric_zone
from thesis.stage_6_reporting import model_metrics
from thesis.stage_6_reporting.benchmarks import _model_label
from thesis.stage_6_reporting.charts import (
    _load_feature_importance,
    _plot_equity_curve,
    _plot_feature_importance,
)
from thesis.stage_6_reporting.comparison import (
    _build_model_comparison_rows,
    _write_model_comparison_artifacts,
)
from thesis.stage_6_reporting.sections import (
    _render_baseline_comparison_section,
    _render_data_quality_section,
    _render_label_design_section,
    _render_metric_zones_section,
    _render_oof_vs_oos_section,
    _render_validation_methodology_section,
)
from thesis.stage_6_reporting.tables import (
    _accuracy_table,
    _backtest_metrics_table,
    _backtest_params_table,
    _benchmark_comparison_table,
    _config_table,
    _exec_table,
    _exec_verdict,
    _feature_importance_table,
    _issues_list,
)

logger = logging.getLogger("thesis.report")

_HIGH_CONFIDENCE_THRESHOLD: float = 0.70
_DIRECTIONAL_BASELINE: float = 0.5


def _load_prediction_stats(preds_path: Path) -> dict | None:
    """Compute prediction quality statistics from a predictions parquet file."""
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
            "directional_baseline": _DIRECTIONAL_BASELINE,
            "majority_baseline": float(raw_metrics["majority_baseline_accuracy"]),
            "macro_f1": float(raw_metrics["macro_f1"]),
            "weighted_f1": float(raw_metrics["weighted_f1"]),
            "per_class": per_class_counts,
            "confusion_matrix": raw_metrics["confusion_matrix"],
            "direction_confusion_matrix": raw_metrics["direction_confusion_matrix"],
        }

        if proba is not None:
            max_proba = proba.max(axis=1)
            hc_mask = max_proba >= _HIGH_CONFIDENCE_THRESHOLD
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
                "threshold": _HIGH_CONFIDENCE_THRESHOLD,
                "count": hc_count,
                "pct_of_total": (hc_count / len(true)) if len(true) else 0.0,
                "accuracy": hc_acc,
                "directional_accuracy": hc_dir_acc,
            }
        return result
    except (ComputeError, ColumnNotFoundError, OSError):
        logger.warning(
            "Failed to load prediction statistics: %s", preds_path, exc_info=True
        )
        return None


_ZONE_EMOJI = {
    "excellent": "✅",
    "good": "🟢",
    "moderate": "🟡",
    "poor": "🟠",
    "dangerous": "🔴",
}


def _zone(key: str, value: float) -> str:
    """Zone emoji for a metric value."""
    if value is None or (
        isinstance(value, float)
        and (math.isnan(value) if isinstance(value, float) else False)
    ):
        return "⚪"
    color, _, _ = get_metric_zone(key, value)
    return _ZONE_EMOJI.get(color, "⚪")


def _build_markdown(
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

    # Title & metadata
    L.append(f"# Thesis Report: {_model_label(config)} — XAU/USD")
    L.append("")
    L.append(f"> Generated: {now} | Session: `{session}`")
    L.append("")

    # 1. Executive Summary
    L.append("## Executive Summary")
    L.append("")
    _exec_table(L, metrics, pred_stats)
    _exec_verdict(L, metrics, pred_stats)
    L.append("")

    # 2. Methodology
    L.append("## Methodology")
    L.append("")
    _render_data_quality_section(L, config, heading="### Data & Quality")
    _render_label_design_section(L, config, heading="### Label Design")
    _render_validation_methodology_section(L, config, heading="### Validation Scheme")

    # Model Architecture subsection
    L.append("### Model Architecture")
    L.append("")
    _feature_importance_table(L, feature_importance)
    L.append("")

    # 3. Classification Results ★
    L.append("## Classification Results ★")
    L.append("")
    L.append(
        "*Classification metrics are the primary evaluation criterion for "
        "this thesis. Directional Accuracy and Macro F1 measure the model's "
        "ability to predict market direction (Short / Hold / Long).*"
    )
    L.append("")
    _accuracy_table(L, pred_stats, config)
    L.append("")

    _render_baseline_comparison_section(L, config, heading="### Baseline Comparison")

    # 4. Application Demo: Backtest
    L.append("## Application Demo: Backtest")
    L.append("")
    L.append(
        "*Backtest results are presented as an application demo to illustrate "
        "how classification signals *could* be translated into trades. "
        "They are **not** the primary evaluation criterion.*"
    )
    L.append("")
    _backtest_params_table(L, config)
    _backtest_metrics_table(L, metrics, config)
    _render_metric_zones_section(L, metrics, trades, heading="")
    L.append("")

    L.append("### Benchmark Comparison")
    L.append("")
    _benchmark_comparison_table(L, metrics, config)

    # 5. Generalization Assessment
    _render_oof_vs_oos_section(L, config, heading="## Generalization Assessment")

    # 6. Issues & Recommendations
    L.append("## Issues & Recommendations")
    L.append("")
    _issues_list(L, metrics, trades, config, pred_stats)
    L.append("")

    # 7. Appendix: Full Configuration
    L.append("## Appendix: Full Configuration")
    L.append("")
    _config_table(L, config)
    L.append("")

    return "\n".join(L)


def _build_model_evaluation_markdown(
    config: Config, pred_stats: dict | None, model_comparison_rows: list[dict[str, Any]]
) -> str:
    """Build compact evaluation-first markdown artifact."""
    lines: list[str] = ["# Model Evaluation", ""]
    lines.append(
        "This file is the primary ML evidence artifact."
        " Backtest metrics are intentionally excluded."
    )
    lines.append("")
    lines.append(f"- Model: {_model_label(config)}")
    lines.append(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    if not pred_stats:
        lines.append("*Prediction statistics unavailable.*")
        return "\n".join(lines)

    def _pct(v, default=0.0):
        return f"{float(v if v is not None else default) * 100:.2f}%"

    def _f4(v, default=0.0):
        return f"{float(v if v is not None else default):.4f}"

    lines.append("## Classification Metrics (Primary)")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Accuracy | {_pct(pred_stats.get('accuracy'))} |")
    da = _pct(pred_stats.get("directional_accuracy"))
    lines.append(f"| Directional Accuracy | {da} |")
    lines.append(f"| Macro F1 | {_f4(pred_stats.get('macro_f1'))} |")
    lines.append(f"| Balanced Accuracy | {_pct(pred_stats.get('balanced_accuracy'))} |")
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
        lines.append("|---|---|")
        lines.append(f"| MAE Return | {reg_aux.get('mae', float('nan')):.6f} |")
        lines.append(f"| RMSE Return | {reg_aux.get('rmse', float('nan')):.6f} |")
        lines.append(f"| R² Return | {reg_aux.get('r_squared', float('nan')):.6f} |")
        lines.append("")

    lines.append("## Model Comparison")
    lines.append("")
    lines.append(
        "| Model | Directional Acc | Accuracy | Macro F1 | Long F1 | Short F1 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")

    def _cell(row: dict, key: str, fmt: str = "pct") -> str:
        v = row.get(key)
        if v is None:
            return ""
        return f"{float(v) * 100:.2f}%" if fmt == "pct" else f"{float(v):.4f}"

    for row in model_comparison_rows:
        lines.append(
            f"| {row.get('model', '')} | {_cell(row, 'directional_accuracy')}"
            f" | {_cell(row, 'accuracy')} | {_cell(row, 'macro_f1', 'f4')}"
            f" | {_cell(row, 'long_f1', 'f4')} | {_cell(row, 'short_f1', 'f4')} |"
        )
    lines.append("")
    return "\n".join(lines)


class ReportData:
    """Structured report payload — all data needed to render & persist reports.

    Computed once by :func:`compute_report_data`, then passed to rendering
    and persistence functions.  Keeps I/O out of renderers.
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


def _setup_matplotlib() -> None:
    """Configure matplotlib for headless report chart rendering."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
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

    No report files are written.  Callers use the returned
    :class:`ReportData` to build markdown and persist artifacts.
    """
    _setup_matplotlib()

    out_dir = (
        Path(config.paths.session_dir) / "reports"
        if config.paths.session_dir
        else Path("results")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(config.paths.report)

    # Load backtest results
    bt_path = Path(config.paths.backtest_results)
    metrics: dict = {}
    trades: list[dict] = []
    if bt_path.exists():
        with console.status(f"[cyan]Loading backtest results[/] {bt_path}"):
            with open(bt_path) as f:
                bt = json.load(f)
            metrics = bt.get("metrics", {})
            trades = bt.get("trades", [])

    # Render charts
    with console.status("[cyan]Rendering report charts[/]"):
        _plot_equity_curve(trades, config, out_dir)
        feature_importance = _load_feature_importance(config, out_dir)
        _plot_feature_importance(feature_importance, out_dir)

    # Compute prediction statistics and model comparisons
    with console.status("[cyan]Building thesis markdown[/]"):
        pred_stats = _load_prediction_stats(Path(config.paths.predictions))
        model_comparison_rows = _build_model_comparison_rows(config, pred_stats)

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

    md = _build_markdown(
        config, data.metrics, data.trades, data.feature_importance, data.pred_stats
    )
    model_eval_md = _build_model_evaluation_markdown(
        config, data.pred_stats, data.model_comparison_rows
    )

    # Persist artifacts
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

    model_cmp_csv = _write_model_comparison_artifacts(
        data.out_dir, data.model_comparison_rows
    )
    logger.info("Model comparison saved: %s", model_cmp_csv)
