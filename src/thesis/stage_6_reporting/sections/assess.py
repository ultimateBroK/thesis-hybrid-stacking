"""Assessment helpers for backtest verdict logic."""

from __future__ import annotations

import numpy as np

from thesis.shared.zones import get_metric_zone

QUALITY_ACC_DELTA: float = 0.05
QUALITY_DIR_ACC_GOOD: float = 0.55
QUALITY_MACRO_F1_GOOD: float = 0.45
QUALITY_DIR_ACC_FAIR: float = 0.50

EDGE_PF_NEGATIVE: float = 1.0
EDGE_SHARPE_MARGINAL: float = 1.0
EDGE_PF_MARGINAL: float = 1.5

MIN_TRADES_DEPLOYABLE: int = 30

ISSUE_DD_CATASTROPHIC: float = 50.0
ISSUE_DD_ELEVATED: float = 30.0
ISSUE_DD_CFD_ELEVATED: float = 20.0
ISSUE_RET_SEVERE_LOSS: float = -50.0
ISSUE_RET_SUSPICIOUS: float = 500.0
ISSUE_WIN_RATE_VIABILITY: float = 40.0
ISSUE_TRADES_MARGINAL: int = 100
ISSUE_SHARPE_POOR: float = 0.5
ISSUE_PF_MARGINAL_EDGE: float = 1.2

SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}
PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}
SEVERITY_ICON = {"critical": "🔴", "warning": "🟡", "info": "✅"}
PRIORITY_ICON = {"high": "🔴", "medium": "🟡", "low": "🔵", "info": "✅"}


def get_zone_info(metric_name: str, value: float | None) -> tuple[str, str, str]:
    if value is None or (isinstance(value, float) and (value != value)):
        return ("⚪", "N/A", "N/A")
    color, label, rec = get_metric_zone(metric_name, value)
    emoji_map = {
        "excellent": "✅",
        "good": "🟢",
        "moderate": "🟡",
        "poor": "🟠",
        "dangerous": "🔴",
    }
    return (emoji_map.get(color, "⚪"), label, rec)


def assess_model_quality(pred_stats: dict) -> tuple[str, str]:
    acc = pred_stats["accuracy"]
    baseline = pred_stats["majority_baseline"]
    dir_acc = pred_stats["directional_accuracy"]
    per_class = pred_stats["per_class"]
    macro_f1 = float(np.mean([per_class[name]["f1"] for name in per_class]))
    gap = acc - baseline

    if gap < 0:
        return ("POOR", "acc below baseline")
    if (
        acc > baseline + QUALITY_ACC_DELTA
        and dir_acc > QUALITY_DIR_ACC_GOOD
        and macro_f1 >= QUALITY_MACRO_F1_GOOD
    ):
        return ("GOOD", "above baseline with directional edge")
    if dir_acc >= QUALITY_DIR_ACC_FAIR:
        return ("FAIR", "slightly above baseline, marginal edge")
    return ("POOR", "no reliable directional edge")


def assess_trading_edge(metrics: dict) -> tuple[str, str]:
    pf = metrics.get("profit_factor", 0)
    sharpe = metrics.get("sharpe_ratio", 0)

    if pf < EDGE_PF_NEGATIVE or sharpe < 0:
        reason = f"PF={pf:.2f}" if pf > 0 else f"PF<{EDGE_PF_NEGATIVE:.1f}"
        return ("NEGATIVE", reason)
    if sharpe < EDGE_SHARPE_MARGINAL or pf < EDGE_PF_MARGINAL:
        return ("MARGINAL", f"PF={pf:.2f}, Sharpe={sharpe:.2f}")
    return ("POSITIVE", f"PF={pf:.2f}, Sharpe={sharpe:.2f}")


def derive_recommendation(ml_quality: str, trading_edge: str, metrics: dict) -> str:
    n_trades = int(metrics.get("num_trades", 0)) if metrics else 0
    if ml_quality == "POOR" or trading_edge == "NEGATIVE":
        return "NOT DEPLOYABLE without fixes"
    if n_trades < MIN_TRADES_DEPLOYABLE:
        return "NOT DEPLOYABLE — insufficient trades for validation"
    if ml_quality == "FAIR" and trading_edge == "MARGINAL":
        return "DEPLOYABLE with caution — marginal edge"
    if ml_quality == "GOOD" and trading_edge == "POSITIVE":
        return "DEPLOYABLE"
    return "DEPLOYABLE with caution"


def identify_primary_issue(metrics: dict, pred_stats: dict | None) -> str | None:
    if not metrics:
        return None

    nt = int(metrics.get("num_trades", 0))
    sh = metrics.get("sharpe_ratio", 0)
    pf = metrics.get("profit_factor", 0)
    dd = abs(metrics.get("max_drawdown_pct", 0))
    ret = metrics.get("return_pct", 0)
    wr = metrics.get("win_rate_pct", 0)
    da = pred_stats.get("directional_accuracy", 0) if pred_stats else 0

    if nt == 0:
        return "Zero trades executed — model produces no actionable signals"
    if nt > 0 and nt < MIN_TRADES_DEPLOYABLE:
        return f"Only {nt} trades — statistically unreliable results"
    if sh < 0:
        return f"Sharpe {sh:.2f} is negative — strategy underperforms risk-free rate"
    if dd > ISSUE_DD_CATASTROPHIC:
        return f"Max drawdown {dd:.1f}% > {ISSUE_DD_CATASTROPHIC:.0f}% — catastrophic capital erosion"
    if pf < EDGE_PF_NEGATIVE:
        return f"Profit factor {pf:.2f} < {EDGE_PF_NEGATIVE:.1f} — strategy loses money on average"
    if da > 0 and da < QUALITY_DIR_ACC_FAIR:
        return f"Directional accuracy {da:.1%} < {QUALITY_DIR_ACC_FAIR:.0%} — predicts worse than random"
    if ret < ISSUE_RET_SEVERE_LOSS:
        return f"Return {ret:.0f}% — severe capital loss"
    if pf < ISSUE_PF_MARGINAL_EDGE and pf >= EDGE_PF_NEGATIVE:
        return f"Profit factor {pf:.2f} < {ISSUE_PF_MARGINAL_EDGE:.1f} — barely covers transaction costs"
    if sh < ISSUE_SHARPE_POOR and sh >= 0:
        return f"Sharpe {sh:.2f} < {ISSUE_SHARPE_POOR:.1f} — poor risk-adjusted returns"
    if dd > ISSUE_DD_ELEVATED and dd <= ISSUE_DD_CATASTROPHIC:
        return f"Max drawdown {dd:.1f}% exceeds {ISSUE_DD_ELEVATED:.0f}% threshold"
    if nt >= MIN_TRADES_DEPLOYABLE and nt < ISSUE_TRADES_MARGINAL:
        return f"Only {nt} trades — marginal sample size"
    if sh < EDGE_SHARPE_MARGINAL and sh >= ISSUE_SHARPE_POOR:
        return f"Sharpe {sh:.2f} < {EDGE_SHARPE_MARGINAL:.1f} — below professional threshold"
    if ret > ISSUE_RET_SUSPICIOUS:
        return f"Return {ret:.0f}% suspiciously high — verify for overfitting"
    if dd > ISSUE_DD_CFD_ELEVATED and dd <= ISSUE_DD_ELEVATED:
        return f"Max drawdown {dd:.1f}% > {ISSUE_DD_CFD_ELEVATED:.0f}% — elevated for CFD trading"
    if wr < ISSUE_WIN_RATE_VIABILITY and wr >= 0:
        return f"Win rate {wr:.1f}% < {ISSUE_WIN_RATE_VIABILITY:.0f}% — below trading viability"
    if da > 0 and da < QUALITY_DIR_ACC_GOOD:
        return (
            f"Directional accuracy {da:.1%} < {QUALITY_DIR_ACC_GOOD:.0%} — unreliable"
        )
    if pf < EDGE_PF_MARGINAL and pf >= ISSUE_PF_MARGINAL_EDGE:
        return f"Profit factor {pf:.2f} < {EDGE_PF_MARGINAL:.1f} — marginal edge"
    return None
