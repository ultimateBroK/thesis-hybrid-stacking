"""Matplotlib/seaborn chart helpers for thesis reporting."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger("thesis.reporting.plots")

# ── style ────────────────────────────────────────────────────────────────
for _s in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
    try:
        plt.style.use(_s)
        break
    except Exception:
        continue


# ── equity curve ─────────────────────────────────────────────────────────


def plot_equity_curve(
    trades: list[dict[str, Any]],
    initial_capital: float,
    output_path: Path | str,
) -> None:
    """Save equity curve PNG from closed-trade list."""
    if not trades:
        return
    output_path = Path(output_path)
    times = [pd.to_datetime(trades[0]["entry_time"])]
    equity = [initial_capital]
    for t in trades:
        times.append(pd.to_datetime(t["exit_time"]))
        equity.append(equity[-1] + t["pnl"])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(times, equity, linewidth=1)
    ax.set_title("Equity Curve")
    ax.set_ylabel("Equity (USD)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


# ── feature importance ───────────────────────────────────────────────────


def plot_feature_importance(
    feature_importance: dict[str, float],
    output_path: Path | str,
    top_n: int = 20,
) -> None:
    """Save horizontal bar chart of top-N feature importances."""
    if not feature_importance:
        return
    output_path = Path(output_path)
    top = dict(
        sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(list(top.keys()), list(top.values()))
    ax.set_title(f"Feature Importance (Top {top_n})")
    ax.invert_yaxis()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def load_feature_importance(path: Path | str) -> dict[str, float]:
    """Load feature importance JSON from disk."""
    p = Path(path)
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


# ── confusion matrix ─────────────────────────────────────────────────────


def plot_confusion_matrix(
    y_true: np.ndarray | list,
    y_pred: np.ndarray | list,
    labels: list[str] | None = None,
    output_path: Path | str = "confusion_matrix.png",
    title: str = "Confusion Matrix",
) -> None:
    """Save confusion-matrix heatmap PNG."""
    output_path = Path(output_path)
    from sklearn.metrics import confusion_matrix as _cm

    cm = _cm(y_true, y_pred)

    if labels is None:
        labels = [str(i) for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)
