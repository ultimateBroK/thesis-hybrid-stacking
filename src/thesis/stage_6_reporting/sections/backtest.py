"""Backtest metric zones, baseline comparison, and verdict section renderers."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl
from polars.exceptions import ColumnNotFoundError, ComputeError

from thesis.shared.config import Config
from thesis.shared import baselines as baselines_mod
from thesis.stage_6_reporting.sections.assess import (
    PRIORITY_ICON,
    PRIORITY_ORDER,
    SEVERITY_ICON,
    SEVERITY_ORDER,
    assess_model_quality,
    assess_trading_edge,
    derive_recommendation,
    get_zone_info,
    identify_primary_issue,
)
from thesis.stage_6_reporting.md_format import _fmt_f2, _fmt_pct, _tbl_row

logger = logging.getLogger("thesis.report")


def compute_avg_win_loss_ratio(trades: list[dict]) -> float | None:
    """Average win / average loss from trades list."""
    wins = [t["pnl"] for t in trades if t["pnl"] > 0]
    losses = [t["pnl"] for t in trades if t["pnl"] < 0]
    if not wins or not losses:
        return None
    avg_win = sum(wins) / len(wins)
    avg_loss = abs(sum(losses) / len(losses))
    if avg_loss == 0:
        return None
    return avg_win / avg_loss


def render_metric_zones_section(
    L: list[str],
    metrics: dict,
    trades: list[dict] | None = None,
    heading: str | None = None,
) -> None:
    """Render metric quality zones section."""
    if heading is None:
        heading = "## Metric Quality Zones"
    L.append(heading)
    L.append("")
    L.append(
        "*Each metric is classified into quality zones based on "
        "industry-standard thresholds. "
        "🔴 = poor/dangerous, 🟡 = marginal, 🟢 = good.*"
    )
    L.append("")

    L.append(_tbl_row("Metric", "Value", "Zone & Rating", "Recommended"))
    L.append(_tbl_row("------", "-----", "------------", "-----------"))

    avg_wl: float | None = None
    if trades:
        avg_wl = compute_avg_win_loss_ratio(trades)

    metric_defs: list[tuple[str, str, callable, float | None]] = [
        ("return_pct", "Total Return", _fmt_pct, metrics.get("return_pct")),
        ("sharpe_ratio", "Sharpe Ratio", _fmt_f2, metrics.get("sharpe_ratio")),
        (
            "max_drawdown_pct",
            "Max Drawdown",
            lambda v: f"{abs(v):.1f}%",
            metrics.get("max_drawdown_pct"),
        ),
        ("win_rate_pct", "Win Rate", _fmt_pct, metrics.get("win_rate_pct")),
        ("profit_factor", "Profit Factor", _fmt_f2, metrics.get("profit_factor")),
        ("calmar_ratio", "Calmar Ratio", _fmt_f2, metrics.get("calmar_ratio")),
        ("sortino_ratio", "Sortino Ratio", _fmt_f2, metrics.get("sortino_ratio")),
        ("avg_win_loss_ratio", "Avg Win / Avg Loss", _fmt_f2, avg_wl),
        ("expectancy_pct", "Expectancy", _fmt_pct, metrics.get("expectancy_pct")),
    ]

    for key, label, fmt, val in metric_defs:
        if val is None:
            L.append(_tbl_row(label, "N/A", "⚪ N/A", "N/A"))
            continue
        emoji, zone_desc, rec = get_zone_info(key, val)
        L.append(_tbl_row(label, fmt(val), f"{emoji} {zone_desc}", rec))
    L.append("")


def render_baseline_comparison_section(
    L: list[str], config: Config, heading: str | None = None
) -> None:
    """Render baseline comparison section."""
    if heading is None:
        heading = "## Baseline Comparison"
    L.append(heading)
    L.append("")

    preds_path = Path(config.paths.predictions)
    if not preds_path.exists():
        L.append("*Predictions not available — baseline comparison skipped.*")
        L.append("")
        return

    try:
        df = pl.read_csv(preds_path)
    except (ComputeError, OSError):
        logger.warning("Failed to load predictions for baselines", exc_info=True)
        L.append("*Predictions file could not be read.*")
        L.append("")
        return

    if "true_label" not in df.columns:
        L.append("*true_label column missing — baseline comparison skipped.*")
        L.append("")
        return

    y_true = df["true_label"].to_numpy()
    y_returns: np.ndarray | None = None
    ohlcv_path = Path(config.paths.ohlcv)
    if ohlcv_path.exists():
        try:
            ohlcv = pl.read_parquet(ohlcv_path, columns=["close"])
            close = ohlcv["close"].to_numpy()
            if len(close) > 1:
                bar_returns = np.diff(close) / close[:-1]
                n = min(len(y_true), len(bar_returns))
                y_returns = bar_returns[-n:]
                y_true = y_true[-n:]
        except (ComputeError, ColumnNotFoundError, ValueError):
            logger.warning("Failed to load OHLCV for baseline returns", exc_info=True)

    if y_returns is None:
        y_returns = y_true.astype(np.float64)

    try:
        baselines = baselines_mod.run_all_baselines(
            y_true, y_returns, seed=config.workflow.random_seed
        )
    except (ValueError, TypeError):
        logger.warning("Failed to compute baselines", exc_info=True)
        L.append("*Baseline computation failed.*")
        L.append("")
        return

    L.append(
        "*Baseline strategies computed on the same prediction labels as "
        "reference. The model should outperform all baselines on "
        "directional accuracy and macro F1.*"
    )
    L.append("")
    L.append(_tbl_row("Strategy", "Accuracy", "Macro F1", "Dir. Accuracy"))
    L.append(_tbl_row("--------", "--------", "---------", "-------------"))
    for name, m in baselines.items():
        display = name.replace("_", " ").title()
        L.append(
            _tbl_row(
                display,
                f"{m['accuracy'] * 100:.1f}%",
                f"{m['macro_f1']:.3f}",
                f"{m['directional_accuracy'] * 100:.1f}%",
            )
        )
    L.append("")


def render_issues(
    L: list[str], issues: list[tuple[str, str]], recs: list[tuple[str, str]]
) -> None:
    """Render issues and recommendations lists."""
    L.append("### Issues")
    L.append("")
    if not issues:
        L.append("*No issues detected.*")
    else:
        sorted_issues = sorted(issues, key=lambda x: SEVERITY_ORDER.get(x[0], 9))
        for i, (severity, desc) in enumerate(sorted_issues, 1):
            icon = SEVERITY_ICON.get(severity, "⚪")
            L.append(f"{i}. {icon} {desc}")
    L.append("")

    L.append("### Recommendations")
    L.append("")
    if not recs:
        L.append("*No specific recommendations.*")
    else:
        sorted_recs = sorted(recs, key=lambda x: PRIORITY_ORDER.get(x[0], 9))
        for i, (priority, desc) in enumerate(sorted_recs, 1):
            icon = PRIORITY_ICON.get(priority, "⚪")
            L.append(f"{i}. {icon} {desc}")


def render_ml_quality_paragraph(L: list[str], pred_stats: dict) -> None:
    """Render ML quality assessment paragraph."""
    acc = pred_stats["accuracy"]
    baseline = pred_stats["majority_baseline"]
    dir_acc = pred_stats["directional_accuracy"]
    per_class = pred_stats["per_class"]
    macro_f1 = float(np.mean([per_class[name]["f1"] for name in per_class]))

    gap = acc - baseline
    if gap < 0:
        ml_quality = "weak"
        gate_msg = "Model is below majority baseline; predictive edge is not validated."
    elif acc > baseline + 0.05 and dir_acc > 0.55 and macro_f1 >= 0.45:
        ml_quality = "strong"
        gate_msg = "Model is above baseline with directional edge."
    elif dir_acc >= 0.50:
        ml_quality = "acceptable"
        gate_msg = "Model is slightly above baseline with marginal directional edge."
    else:
        ml_quality = "weak"
        gate_msg = "Model has no reliable directional edge."
    L.append(
        f"ML quality is **{ml_quality}**: exact accuracy {acc:.1%} vs "
        f"majority baseline {baseline:.1%}, directional accuracy {dir_acc:.1%}, "
        f"macro F1 {macro_f1:.3f}. {gate_msg} Backtest figures below are treated as an "
        "application demo, not the primary proof of model quality."
    )


def render_synthesized_verdict(L: list[str], pred_stats: dict, metrics: dict) -> None:
    """Render synthesized verdict from model quality and trading edge."""
    model_quality, ml_reason = assess_model_quality(pred_stats)
    if metrics:
        trading_edge, trade_reason = assess_trading_edge(metrics)
        recommendation = derive_recommendation(model_quality, trading_edge, metrics)
        L.append(
            f"**Verdict:** Model quality **{model_quality}** ({ml_reason}), "
            f"Trading edge **{trading_edge}** ({trade_reason}), "
            f"Recommendation: **{recommendation}**."
        )
    else:
        L.append(
            f"**Verdict:** Model quality **{model_quality}** ({ml_reason}). "
            "No backtest metrics available for trading assessment."
        )


def render_primary_issue(L: list[str], metrics: dict, pred_stats: dict | None) -> None:
    """Render primary issue paragraph."""
    if metrics:
        primary = identify_primary_issue(metrics, pred_stats)
        if primary:
            L.append(f"**Primary issue:** {primary}.")
    else:
        L.append("**Primary issue:** No backtest metrics — pipeline may have failed.")

    if not metrics:
        return
    ret = metrics.get("return_pct", 0)
    sharpe = metrics.get("sharpe_ratio", 0)
    n_trades = int(metrics.get("num_trades", 0))
    wr = metrics.get("win_rate_pct", 0)
    dd = abs(metrics.get("max_drawdown_pct", 0))
    L.append(
        f"Application demo returned {ret:.1f}% over {n_trades} trades "
        f"with Sharpe {sharpe:.2f}, win rate {wr:.1f}%, max drawdown {dd:.1f}%."
    )
