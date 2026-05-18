"""Static report figure exports (Matplotlib).

Renders PNG/SVG charts for thesis report from pipeline artifacts.
Dashboard uses pyecharts — this module is the separate static renderer.

Output: ``results/<session>/charts/``
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger("thesis.reporting.figures")

for _s in ("seaborn-v0_8-white", "seaborn-white", "ggplot"):
    try:
        plt.style.use(_s)
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


def _save(fig: plt.Figure, path: Path, fmt: str = "png", dpi: int = 180) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)


def _load_label_counts(labels_path: Path) -> dict[str, int]:
    import polars as pl

    try:
        df = pl.read_parquet(labels_path, columns=["label"])
        counts = {}
        for val, name in [(-1, "Short"), (0, "Hold"), (1, "Long")]:
            counts[name] = int((df["label"] == val).sum())
        return counts
    except Exception:
        logger.warning(
            "Failed to load label counts from %s", labels_path, exc_info=True
        )
        return {}


def export_label_distribution(
    label_counts: dict[str, int],
    output_path: Path,
    dpi: int = 180,
) -> None:
    """Bar chart: Short / Hold / Long counts."""
    names = ["Short", "Hold", "Long"]
    counts = [label_counts.get(n, 0) for n in names]
    total = sum(counts) or 1
    pcts = [c / total * 100 for c in counts]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        names, counts, color=["#e74c3c", "#95a5a6", "#2ecc71"], edgecolor="white"
    )
    for bar, cnt, pct in zip(bars, counts, pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.01,
            f"{cnt:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title("Label Distribution")
    ax.set_ylabel("Count")
    ax.set_ylim(0, max(counts) * 1.2 if max(counts) > 0 else 1)
    ax.grid(False)
    _save(fig, output_path, dpi=dpi)


def export_model_comparison(
    comparison_rows: list[dict[str, Any]],
    metric: str = "accuracy",
    output_path: Path = Path("model_comparison.png"),
    dpi: int = 180,
) -> None:
    """Grouped bar chart comparing models on one metric."""
    models = [r["model"] for r in comparison_rows]
    values = [
        float(r.get(metric, 0) or 0) * 100
        if float(r.get(metric, 0) or 0) <= 1.0
        else float(r.get(metric, 0) or 0)
        for r in comparison_rows
    ]
    title = (
        "Accuracy Comparison (%)" if metric == "accuracy" else "Macro F1 Comparison (%)"
    )
    colors = [
        "#bdc3c7" if v == 0 else "#2ecc71" if v == max(values) else "#3498db"
        for v in values
    ]

    width = max(8, len(models) * 2.2)
    fig, ax = plt.subplots(figsize=(width, 5))
    bars = ax.bar(range(len(models)), values, color=colors, edgecolor="white")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=0, ha="center", fontsize=10)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title(title)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_ylim(0, max(values) * 1.15 if max(values) > 0 else 1)
    ax.grid(False)
    _save(fig, output_path, dpi=dpi)


def export_confusion_matrix(
    cm: dict[str, dict[str, int]],
    labels: list[str] | None = None,
    output_path: Path = Path("confusion_matrix.png"),
    dpi: int = 180,
) -> None:
    """Heatmap confusion matrix with annotations."""
    if labels is None:
        labels = list(cm.keys())
    n = len(labels)
    mat = np.zeros((n, n), dtype=int)
    for i, r in enumerate(labels):
        for j, c in enumerate(labels):
            mat[i, j] = cm.get(r, {}).get(c, 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, cmap="Blues", interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    ax.grid(False)
    for i in range(n):
        for j in range(n):
            color = "white" if mat[i, j] > mat.max() / 2 else "black"
            ax.text(
                j, i, str(mat[i, j]), ha="center", va="center", color=color, fontsize=11
            )
    _save(fig, output_path, dpi=dpi)


def export_feature_importance(
    importance: dict[str, float],
    output_path: Path = Path("feature_importance.png"),
    top_n: int = 15,
    dpi: int = 180,
) -> None:
    """Horizontal bar chart: top N features."""
    if not importance:
        return
    ranked = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [n for n, _ in ranked]
    vals = [v for _, v in ranked]

    fig, ax = plt.subplots(figsize=(8, max(6, top_n * 0.35)))
    ax.barh(names, vals, color="#3498db", edgecolor="white")
    ax.invert_yaxis()
    ax.set_title(f"Feature Importance (Top {top_n})")
    ax.set_xlabel("Importance")
    ax.grid(False)
    _save(fig, output_path, dpi=dpi)


def export_equity_curve(
    trades: list[dict[str, Any]],
    initial_capital: float,
    output_path: Path = Path("equity_curve.png"),
    dpi: int = 180,
) -> None:
    """Equity curve from closed trades."""
    if not trades:
        return
    import pandas as pd

    times = [pd.to_datetime(trades[0]["entry_time"])]
    equity = [initial_capital]
    for t in trades:
        times.append(pd.to_datetime(t["exit_time"]))
        equity.append(equity[-1] + t["pnl"])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(times, equity, linewidth=1, color="#2c3e50")
    ax.fill_between(times, initial_capital, equity, alpha=0.1, color="#2ecc71")
    ax.axhline(initial_capital, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title("Equity Curve")
    ax.set_ylabel("Equity (USD)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    _save(fig, output_path, dpi=dpi)


def _export_label_dist_if_available(charts_dir: Path, config: Any, dpi: int) -> None:
    labels_path = Path(config.paths.labels) if hasattr(config, "paths") else None
    if not labels_path or not labels_path.exists():
        return
    label_counts = _load_label_counts(labels_path)
    if label_counts:
        export_label_distribution(
            label_counts, charts_dir / "01_label_distribution.png", dpi=dpi
        )


def _export_model_cmp_if_available(
    session_dir: Path, charts_dir: Path, dpi: int
) -> None:
    comparison_csv = session_dir / "reports" / "model_comparison.csv"
    if not comparison_csv.exists():
        return
    import pandas as pd

    df = pd.read_csv(comparison_csv)
    rows = df.to_dict("records")
    export_model_comparison(
        rows, "accuracy", charts_dir / "02_accuracy_comparison.png", dpi=dpi
    )
    export_model_comparison(
        rows, "macro_f1", charts_dir / "03_macro_f1_comparison.png", dpi=dpi
    )


def _export_cm_if_available(
    charts_dir: Path, artifacts: dict[str, Any], dpi: int
) -> None:
    cm = artifacts.get("confusion_matrix") if artifacts else None
    if cm:
        export_confusion_matrix(
            cm, output_path=charts_dir / "04_confusion_matrix.png", dpi=dpi
        )


def _export_fi_if_available(
    session_dir: Path, charts_dir: Path, top_n: int, dpi: int
) -> None:
    fi_path = session_dir / "reports" / "feature_importance.json"
    if not fi_path.exists():
        return
    try:
        importance = json.loads(fi_path.read_text())
        export_feature_importance(
            importance, charts_dir / "05_feature_importance.png", top_n=top_n, dpi=dpi
        )
    except (OSError, json.JSONDecodeError):
        logger.warning("Failed to load feature importance", exc_info=True)


def _export_equity_if_available(
    session_dir: Path, charts_dir: Path, config: Any, dpi: int
) -> None:
    bt_path = Path(config.paths.backtest_results) if hasattr(config, "paths") else None
    if not bt_path or not bt_path.exists():
        return
    try:
        bt = json.loads(bt_path.read_text())
        trades = bt.get("trades", [])
        cap = config.backtest.initial_capital
        export_equity_curve(trades, cap, charts_dir / "06_equity_curve.png", dpi=dpi)
    except (OSError, json.JSONDecodeError):
        logger.warning("Failed to load backtest for equity curve", exc_info=True)


def _write_chart_manifest(charts_dir: Path, dpi: int) -> None:
    manifest = {
        "charts": sorted(
            p.name for p in charts_dir.iterdir() if p.suffix in (".png", ".svg")
        ),
        "dpi": dpi,
    }
    (charts_dir / "chart_manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("Chart export complete: %s", charts_dir)


def export_all_figures(
    session_dir: Path,
    config: Any,
    artifacts: dict[str, Any] | None = None,
    dpi: int = 180,
    top_n_features: int = 15,
) -> Path:
    """Export all report charts to session_dir/reports/charts/."""
    charts_dir = session_dir / "reports" / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    _export_label_dist_if_available(charts_dir, config, dpi)
    _export_model_cmp_if_available(session_dir, charts_dir, dpi)
    _export_cm_if_available(charts_dir, artifacts or {}, dpi)
    _export_fi_if_available(session_dir, charts_dir, top_n_features, dpi)
    _export_equity_if_available(session_dir, charts_dir, config, dpi)
    _write_chart_manifest(charts_dir, dpi)

    return charts_dir
