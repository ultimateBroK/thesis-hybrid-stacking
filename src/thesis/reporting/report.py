"""Report generation orchestrator.

Merged from legacy reporting modules: generation, tables, md_format,
comparison, sections (assess, backtest, data, oof), calibration, benchmarks.

Produces: final_report.md, model_evaluation.md, metrics.json,
confusion_matrix.png, model_comparison.csv, feature_importance.png.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from polars.exceptions import ColumnNotFoundError, ComputeError

from thesis.models import baselines as baselines_mod
from thesis.reporting.metrics import (
    compute_all_classification_metrics,
)
from thesis.reporting.metrics import (
    high_confidence_accuracy as _high_confidence_accuracy,
)
from thesis.reporting.plots import (
    load_feature_importance,
    plot_confusion_matrix,
    plot_equity_curve,
    plot_feature_importance,
)
from thesis.shared.config import Config
from thesis.shared.constants import H1_BARS_PER_YEAR
from thesis.shared.utils import console
from thesis.shared.zones import get_metric_zone

logger = logging.getLogger("thesis.report")

# ── Constants ───────────────────────────────────────────────────────────

HIGH_CONFIDENCE_THRESHOLD: float = 0.70
DIRECTIONAL_BASELINE: float = 0.5
BARS_PER_YEAR = H1_BARS_PER_YEAR
MIN_TRADES_DEPLOYABLE: int = 30
ECE_WELL_CALIBRATED: float = 0.05
ECE_MODERATELY_CALIBRATED: float = 0.15
SIGNIFICANCE_ALPHA: float = 0.05

SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}
PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}
SEVERITY_ICON = {"critical": "🔴", "warning": "🟡", "info": "✅"}
PRIORITY_ICON = {"high": "🔴", "medium": "🟡", "low": "🔵", "info": "✅"}

QUALITY_ACC_DELTA: float = 0.05
QUALITY_DIR_ACC_GOOD: float = 0.55
QUALITY_MACRO_F1_GOOD: float = 0.45
QUALITY_DIR_ACC_FAIR: float = 0.50
EDGE_PF_NEGATIVE: float = 1.0
EDGE_SHARPE_MARGINAL: float = 1.0
EDGE_PF_MARGINAL: float = 1.5

ISSUE_DD_CATASTROPHIC: float = 50.0
ISSUE_DD_ELEVATED: float = 30.0
ISSUE_DD_CFD_ELEVATED: float = 20.0
ISSUE_RET_SEVERE_LOSS: float = -50.0
ISSUE_RET_SUSPICIOUS: float = 500.0
ISSUE_WIN_RATE_VIABILITY: float = 40.0
ISSUE_TRADES_MARGINAL: int = 100
ISSUE_SHARPE_POOR: float = 0.5
ISSUE_PF_MARGINAL_EDGE: float = 1.2

# ── Markdown formatting helpers ─────────────────────────────────────────

_ZONE_EMOJI = {
    "excellent": "✅",
    "good": "🟢",
    "moderate": "🟡",
    "poor": "🟠",
    "dangerous": "🔴",
}


def _zone(key: str, value: float) -> str:
    import math

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "⚪"
    color, _, _ = get_metric_zone(key, value)
    return _ZONE_EMOJI.get(color, "⚪")


def _tbl_row(*cells: str) -> str:
    return "| " + " | ".join(cells) + " |"


def _fmt_pct(v: float) -> str:
    return f"{v:.1f}%"


def _fmt_f2(v: float) -> str:
    return f"{v:.2f}"


def _fmt_dollar(v: float) -> str:
    return f"${v:,.0f}"


def _get_zone_info(metric_name: str, value: float | None) -> tuple[str, str, str]:

    if value is None or (isinstance(value, float) and (value != value)):
        return ("⚪", "N/A", "N/A")
    color, label, rec = get_metric_zone(metric_name, value)
    return (_ZONE_EMOJI.get(color, "⚪"), label, rec)


# ── Calibration ────────────────────────────────────────────────────────


def _to_onehot(y_true: np.ndarray, classes: list[int]) -> np.ndarray:
    k = len(classes)
    idx_map = {c: i for i, c in enumerate(classes)}
    indices = np.array([idx_map[int(y)] for y in y_true], dtype=int)
    return np.eye(k)[indices]


def _expected_calibration_error(
    y_true_onehot: np.ndarray, y_proba: np.ndarray, n_bins: int = 10
) -> float:
    confidences = np.max(y_proba, axis=1)
    correct = (np.argmax(y_proba, axis=1) == np.argmax(y_true_onehot, axis=1)).astype(
        float
    )
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        count = mask.sum()
        if count == 0:
            continue
        ece += count * np.abs(confidences[mask].mean() - correct[mask].mean())
    return float(ece / len(y_true_onehot))


def _brier_score(y_true_onehot: np.ndarray, y_proba: np.ndarray) -> float:
    return float(np.mean((y_true_onehot - y_proba) ** 2))


def _log_loss(
    y_true: np.ndarray, y_proba: np.ndarray, classes: list[int] | None = None
) -> float:
    if classes is None:
        classes = [-1, 0, 1]
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    eps = 1e-15
    clipped = np.clip(y_proba, eps, 1.0 - eps)
    normalized = clipped / clipped.sum(axis=1, keepdims=True)
    indices = np.array([class_to_idx[int(y)] for y in y_true], dtype=int)
    return float(-np.mean(np.log(normalized[np.arange(len(y_true)), indices])))


def _compute_all_calibration_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    classes: list[int] | None = None,
) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_proba = np.asarray(y_proba, dtype=np.float64)
    if classes is None:
        classes = [-1, 0, 1]
    y_true_onehot = _to_onehot(y_true, classes)
    return {
        "ece": round(_expected_calibration_error(y_true_onehot, y_proba), 6),
        "brier_score": round(_brier_score(y_true_onehot, y_proba), 6),
        "log_loss": round(_log_loss(y_true, y_proba, classes=classes), 6),
        "high_confidence_accuracy": _high_confidence_accuracy(
            y_true, y_pred, y_proba, threshold=0.70
        ),
    }


def _calibration_summary_text(config: Config) -> str | None:
    preds_path = Path(config.paths.predictions)
    if not preds_path.exists():
        return None
    proba_cols = [
        "pred_proba_class_minus1",
        "pred_proba_class_0",
        "pred_proba_class_1",
    ]
    try:
        df = pl.read_csv(preds_path)
    except (ComputeError, OSError):
        logger.warning(
            "Failed to load predictions for calibration check: %s",
            preds_path,
            exc_info=True,
        )
        return None
    if not all(c in df.columns for c in proba_cols):
        return None
    if "true_label" not in df.columns:
        return None
    proba = df.select(proba_cols).to_numpy()
    true_labels = df["true_label"].to_numpy()
    pred_labels = df["pred_label"].to_numpy() if "pred_label" in df.columns else None
    calib = _compute_all_calibration_metrics(
        true_labels,
        pred_labels if pred_labels is not None else np.argmax(proba, axis=1) - 1,
        proba,
        classes=[-1, 0, 1],
    )
    ece = calib["ece"]
    brier = calib["brier_score"]
    logloss = calib["log_loss"]
    if ece < ECE_WELL_CALIBRATED:
        quality = "well-calibrated"
        note = (
            f"**Calibration**: ECE = {ece:.4f}. Confidence scores **{quality}** "
            f"(ECE < {ECE_WELL_CALIBRATED:.2f}). "
            "Probabilities match observed frequencies."
        )
    elif ece < ECE_MODERATELY_CALIBRATED:
        quality = "moderately calibrated"
        note = (
            f"**Calibration**: ECE = {ece:.4f}. Confidence scores **{quality}** "
            f"({ECE_WELL_CALIBRATED:.2f} <= ECE < {ECE_MODERATELY_CALIBRATED:.2f}). "
            "Probabilities somewhat aligned. Model over/under-confident in some bins."
        )
    else:
        quality = "poorly calibrated"
        note = (
            f"**Calibration**: ECE = {ece:.4f}. Confidence scores **{quality}** "
            f"(ECE >= {ECE_MODERATELY_CALIBRATED:.2f}). "
            "Probabilities do not reflect true likelihoods. "
            "Use temperature scaling or isotonic regression."
        )
    note += f" Brier score = {brier:.4f}, Log-loss = {logloss:.4f}."
    logger.info(
        "Calibration summary: ECE=%.4f, Brier=%.4f, LogLoss=%.4f (%s)",
        ece,
        brier,
        logloss,
        quality,
    )
    return note


# ── Benchmarks ──────────────────────────────────────────────────────────


def model_label(config: Config) -> str:
    """Return human-readable model label from config."""
    architecture = config.model.architecture
    if architecture in ("static", "lgbm"):
        return "LightGBM"
    if architecture == "stacking":
        return "Hybrid Stacking"
    return f"{architecture.title()} Model"


def _annualized_sharpe(
    returns: np.ndarray, bars_per_year: int = BARS_PER_YEAR
) -> float:
    std = float(np.std(returns, ddof=1))
    if std == 0 or np.isnan(std):
        return 0.0
    return float(np.mean(returns) / std * np.sqrt(bars_per_year))


def _max_drawdown_pct(equity: np.ndarray) -> float:
    if len(equity) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak * 100
    return float(abs(dd.min()))


def _equity_curve_from_bar_returns(
    returns: np.ndarray, initial_capital: float
) -> np.ndarray:
    equity = np.empty(len(returns) + 1)
    equity[0] = initial_capital
    for i, r in enumerate(returns):
        equity[i + 1] = equity[i] * (1.0 + r)
    return equity


def _compute_random_strategy(
    returns: np.ndarray, initial_capital: float, leverage: int, seed: int
) -> dict:
    rng = np.random.default_rng(seed)
    signals = rng.choice([-1, 1], size=len(returns))
    leveraged = returns * signals * leverage
    equity = _equity_curve_from_bar_returns(leveraged, initial_capital)
    ret = (equity[-1] / initial_capital - 1) * 100
    sharpe = _annualized_sharpe(leveraged)
    max_dd = _max_drawdown_pct(equity)
    active = leveraged[signals != 0]
    win_rate = float((active > 0).sum() / len(active) * 100) if len(active) > 0 else 0.0
    return {
        "return_pct": ret,
        "sharpe": sharpe,
        "max_dd_pct": max_dd,
        "win_rate_pct": win_rate,
        "num_trades": int((np.diff(signals) != 0).sum()) + 1,
    }


def _load_close_prices_for_benchmark(
    test_data_path: Path, hybrid_metrics: dict, config: Config
) -> np.ndarray | None:
    is_static = config.validation.method == "static"
    if test_data_path.exists() and is_static:
        try:
            df = pl.read_parquet(test_data_path, columns=["close"])
            return df["close"].to_numpy()
        except (ComputeError, OSError):
            logger.warning(
                "Failed to load static test data: %s", test_data_path, exc_info=True
            )
    elif test_data_path.exists() and not is_static:
        logger.warning(
            "Static test file found but workflow is walk-forward"
            " — ignoring stale test_data"
        )
    ohlcv_path = Path(config.paths.ohlcv)
    if not ohlcv_path.exists():
        logger.warning("No OHLCV available for benchmark fallback: %s", ohlcv_path)
        return None
    try:
        df = pl.read_parquet(ohlcv_path)
    except (ComputeError, OSError):
        logger.warning(
            "Failed to load OHLCV for benchmarks: %s", ohlcv_path, exc_info=True
        )
        return None
    ts_expr = pl.col("timestamp")
    ts_dtype = df.schema.get("timestamp")
    if ts_dtype == pl.Utf8:
        ts_expr = ts_expr.str.to_datetime()
        ts_dtype = df.select(ts_expr.alias("timestamp")).schema["timestamp"]
    if getattr(ts_dtype, "time_zone", None):
        ts_expr = ts_expr.dt.replace_time_zone(None)
    bt_start = hybrid_metrics.get("start")
    bt_end = hybrid_metrics.get("end")
    if bt_start and bt_end:
        start_dt = datetime.fromisoformat(str(bt_start)[:19])
        end_dt = datetime.fromisoformat(str(bt_end)[:19])
        df = df.filter((ts_expr >= start_dt) & (ts_expr <= end_dt))
    if len(df) < 2:
        logger.warning("OHLCV fallback: insufficient bars (%d)", len(df))
        return None
    logger.info("Benchmark using OHLCV fallback: %d bars", len(df))
    return df["close"].to_numpy()


def _compute_benchmark_comparison(
    test_data_path: Path, hybrid_metrics: dict, config: Config
) -> list[dict]:
    close = _load_close_prices_for_benchmark(test_data_path, hybrid_metrics, config)
    if close is None or len(close) < 2:
        return []
    initial = config.backtest.initial_capital
    leverage = config.backtest.leverage
    seed = config.workflow.random_seed
    bar_returns = np.diff(close) / close[:-1]
    bh_equity = _equity_curve_from_bar_returns(bar_returns, initial)
    bh_return = (bh_equity[-1] / initial - 1) * 100
    al_returns = (bar_returns * leverage).copy()
    al_equity = _equity_curve_from_bar_returns(al_returns, initial)
    al_return = (al_equity[-1] / initial - 1) * 100
    random_result = _compute_random_strategy(bar_returns, initial, leverage, seed)
    return [
        {
            "strategy": "Buy & Hold",
            "return_pct": bh_return,
            "sharpe": _annualized_sharpe(bar_returns),
            "max_dd_pct": _max_drawdown_pct(bh_equity),
            "win_rate_pct": float((bar_returns > 0).sum() / len(bar_returns) * 100)
            if len(bar_returns) > 0
            else 0.0,
            "num_trades": 1,
        },
        {
            "strategy": "Always Long",
            "return_pct": al_return,
            "sharpe": _annualized_sharpe(al_returns),
            "max_dd_pct": _max_drawdown_pct(al_equity),
            "win_rate_pct": float((al_returns > 0).sum() / len(al_returns) * 100)
            if len(al_returns) > 0
            else 0.0,
            "num_trades": 1,
        },
        {"strategy": "Random Signal", **random_result},
        {
            "strategy": model_label(config),
            "return_pct": hybrid_metrics.get("return_pct", 0),
            "sharpe": hybrid_metrics.get("sharpe_ratio", 0),
            "max_dd_pct": abs(hybrid_metrics.get("max_drawdown_pct", 0)),
            "win_rate_pct": hybrid_metrics.get("win_rate_pct", 0),
            "num_trades": int(hybrid_metrics.get("num_trades", 0)),
        },
    ]


# ── Comparison ──────────────────────────────────────────────────────────


def _parse_date(date_str: str) -> datetime | None:
    if not date_str:
        return None
    tz_aware_formats = ("%Y-%m-%d %H:%M:%S%z", "%Y-%m-%dT%H:%M:%S%z")
    for fmt in tz_aware_formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    naive_formats = ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S")
    truncated = date_str[:19]
    for fmt in naive_formats:
        try:
            return datetime.strptime(truncated, fmt)
        except ValueError:
            continue
    return None


def build_model_comparison_rows(
    config: Config, pred_stats: dict | None
) -> list[dict[str, Any]]:
    """Build thesis-level model comparison rows with available metrics."""
    rows: list[dict[str, Any]] = []
    if pred_stats:
        per_class = pred_stats.get("per_class", {})
        reg_aux = pred_stats.get("regression_auxiliary", {})
        rows.append(
            {
                "model": model_label(config),
                "directional_accuracy": pred_stats.get("directional_accuracy"),
                "accuracy": pred_stats.get("accuracy"),
                "macro_f1": pred_stats.get("macro_f1"),
                "long_f1": per_class.get("Long", {}).get("f1"),
                "short_f1": per_class.get("Short", {}).get("f1"),
                "mae_return": reg_aux.get("mae"),
                "rmse_return": reg_aux.get("rmse"),
                "r2_return": reg_aux.get("r_squared"),
                "source": "current_session",
            }
        )
    preds_path = Path(config.paths.predictions)
    if preds_path.exists():
        try:
            df = pl.read_csv(preds_path)
            y_true = df["true_label"].to_numpy()
            close_path = Path(config.paths.ohlcv)
            y_returns = np.zeros(len(y_true), dtype=np.float64)
            if close_path.exists():
                ohlcv = pl.read_parquet(close_path, columns=["close"])
                close = ohlcv["close"].to_numpy()
                if len(close) > 1:
                    bar_returns = np.diff(close) / close[:-1]
                    n = min(len(y_true), len(bar_returns))
                    y_returns = bar_returns[-n:]
                    y_true = y_true[-n:]
            baselines = baselines_mod.run_all(
                y_true, y_returns, seed=config.workflow.random_seed
            )
            for baseline_key, label in (
                ("naive_direction", "Naive Direction"),
                ("majority_class", "Majority Baseline"),
                ("random", "Random Baseline"),
            ):
                if baseline_key not in baselines:
                    continue
                m = baselines[baseline_key]
                rows.append(
                    {
                        "model": label,
                        "directional_accuracy": m.get("directional_accuracy"),
                        "accuracy": m.get("accuracy"),
                        "macro_f1": m.get("macro_f1"),
                        "long_f1": m.get("long_f1"),
                        "short_f1": m.get("short_f1"),
                        "mae_return": None,
                        "rmse_return": None,
                        "r2_return": None,
                        "source": "derived_baseline",
                    }
                )
        except (ColumnNotFoundError, ValueError):
            logger.warning(
                "Failed to build baseline rows for model comparison", exc_info=True
            )
    session_dir = config.paths.session_dir
    existing = {str(r["model"]).lower() for r in rows}
    name_map = {
        "logreg": "Logistic Regression",
        "rf": "Random Forest",
        "lgbm": "LightGBM",
        "hybrid_stacking": "Hybrid Stacking",
    }
    comparison_json = (
        Path(session_dir) / "reports" / "model_comparison.json" if session_dir else None
    )
    if comparison_json and comparison_json.exists():
        try:
            model_comparison = json.loads(comparison_json.read_text())
        except (OSError, json.JSONDecodeError):
            logger.warning("Failed to read model comparison JSON", exc_info=True)
            model_comparison = {}
        for key, metrics in model_comparison.items():
            model_name = name_map.get(key, str(key).replace("_", " ").title())
            if model_name.lower() in existing:
                continue
            per_class = metrics.get("per_class", {})
            dir_acc = metrics.get("directional_accuracy")
            if dir_acc is None and per_class:
                short_data = per_class.get("-1", {})
                long_data = per_class.get("1", {})
                short_correct = short_data.get("recall", 0) * short_data.get(
                    "support", 0
                )
                long_correct = long_data.get("recall", 0) * long_data.get("support", 0)
                total_dir = short_data.get("support", 0) + long_data.get("support", 0)
                if total_dir > 0:
                    dir_acc = (short_correct + long_correct) / total_dir
            rows.append(
                {
                    "model": model_name,
                    "directional_accuracy": dir_acc,
                    "accuracy": metrics.get("accuracy"),
                    "macro_f1": metrics.get("macro_f1"),
                    "long_f1": per_class.get("1", {}).get("f1"),
                    "short_f1": per_class.get("-1", {}).get("f1"),
                    "mae_return": None,
                    "rmse_return": None,
                    "r2_return": None,
                    "source": "walk_forward_model_comparison",
                }
            )
            existing.add(model_name.lower())
    for model_name in (
        "Logistic Regression",
        "Random Forest",
        "LightGBM",
        "Hybrid Stacking",
    ):
        if model_name.lower() in existing:
            continue
        rows.append(
            {
                "model": model_name,
                "directional_accuracy": None,
                "accuracy": None,
                "macro_f1": None,
                "long_f1": None,
                "short_f1": None,
                "mae_return": None,
                "rmse_return": None,
                "r2_return": None,
                "source": "pending_experiment",
            }
        )
    return rows


def write_model_comparison_artifacts(out_dir: Path, rows: list[dict[str, Any]]) -> Path:
    """Write model comparison table to CSV."""
    csv_path = out_dir / "model_comparison.csv"
    frame = pd.DataFrame(rows)
    frame.to_csv(csv_path, index=False)
    return csv_path


# ── Assessment helpers ──────────────────────────────────────────────────


def _assess_model_quality(pred_stats: dict) -> tuple[str, str]:
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
        return ("GOOD", "above baseline. directional edge")
    if dir_acc >= QUALITY_DIR_ACC_FAIR:
        return ("FAIR", "above baseline. marginal edge")
    return ("POOR", "no directional edge")


def _assess_trading_edge(metrics: dict) -> tuple[str, str]:
    pf = metrics.get("profit_factor", 0)
    sharpe = metrics.get("sharpe_ratio", 0)
    if pf < EDGE_PF_NEGATIVE or sharpe < 0:
        reason = f"PF={pf:.2f}" if pf > 0 else f"PF<{EDGE_PF_NEGATIVE:.1f}"
        return ("NEGATIVE", reason)
    if sharpe < EDGE_SHARPE_MARGINAL or pf < EDGE_PF_MARGINAL:
        return ("MARGINAL", f"PF={pf:.2f}, Sharpe={sharpe:.2f}")
    return ("POSITIVE", f"PF={pf:.2f}, Sharpe={sharpe:.2f}")


def _derive_recommendation(ml_quality: str, trading_edge: str, metrics: dict) -> str:
    n_trades = int(metrics.get("num_trades", 0)) if metrics else 0
    if ml_quality == "POOR" or trading_edge == "NEGATIVE":
        return "NOT DEPLOYABLE. Fix needed"
    if n_trades < MIN_TRADES_DEPLOYABLE:
        return "NOT DEPLOYABLE. Insufficient trades"
    if ml_quality == "FAIR" and trading_edge == "MARGINAL":
        return "DEPLOYABLE with caution. Marginal edge"
    if ml_quality == "GOOD" and trading_edge == "POSITIVE":
        return "DEPLOYABLE"
    return "DEPLOYABLE with caution"


def _identify_primary_issue(metrics: dict, pred_stats: dict | None) -> str | None:
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
        return "Zero trades executed. No actionable signals"
    if nt > 0 and nt < MIN_TRADES_DEPLOYABLE:
        return f"Only {nt} trades. Statistically unreliable"
    if sh < 0:
        return f"Sharpe {sh:.2f} negative. Underperforms risk-free rate"
    if dd > ISSUE_DD_CATASTROPHIC:
        return (
            f"Max drawdown {dd:.1f}% > {ISSUE_DD_CATASTROPHIC:.0f}%. "
            "Catastrophic capital erosion"
        )
    if pf < EDGE_PF_NEGATIVE:
        return f"Profit factor {pf:.2f} < {EDGE_PF_NEGATIVE:.1f}. Strategy loses money"
    if da > 0 and da < QUALITY_DIR_ACC_FAIR:
        return (
            f"Directional accuracy {da:.1%} < {QUALITY_DIR_ACC_FAIR:.0%}. "
            "Predicts worse than random"
        )
    if ret < ISSUE_RET_SEVERE_LOSS:
        return f"Return {ret:.0f}%. Severe capital loss"
    if pf < ISSUE_PF_MARGINAL_EDGE and pf >= EDGE_PF_NEGATIVE:
        return (
            f"Profit factor {pf:.2f} < {ISSUE_PF_MARGINAL_EDGE:.1f}. "
            "Barely covers costs"
        )
    if sh < ISSUE_SHARPE_POOR and sh >= 0:
        return f"Sharpe {sh:.2f} < {ISSUE_SHARPE_POOR:.1f}. Poor risk-adjusted returns"
    if dd > ISSUE_DD_ELEVATED and dd <= ISSUE_DD_CATASTROPHIC:
        return f"Max drawdown {dd:.1f}% > {ISSUE_DD_ELEVATED:.0f}%. Elevated"
    if nt >= MIN_TRADES_DEPLOYABLE and nt < ISSUE_TRADES_MARGINAL:
        return f"Only {nt} trades. Marginal sample size"
    if sh < EDGE_SHARPE_MARGINAL and sh >= ISSUE_SHARPE_POOR:
        return (
            f"Sharpe {sh:.2f} < {EDGE_SHARPE_MARGINAL:.1f}. "
            "Below professional threshold"
        )
    if ret > ISSUE_RET_SUSPICIOUS:
        return f"Return {ret:.0f}%. Suspiciously high. Verify overfitting"
    if dd > ISSUE_DD_CFD_ELEVATED and dd <= ISSUE_DD_ELEVATED:
        return (
            f"Max drawdown {dd:.1f}% > {ISSUE_DD_CFD_ELEVATED:.0f}%. Elevated for CFD"
        )
    if wr < ISSUE_WIN_RATE_VIABILITY and wr >= 0:
        return f"Win rate {wr:.1f}% < {ISSUE_WIN_RATE_VIABILITY:.0f}%. Below viability"
    if da > 0 and da < QUALITY_DIR_ACC_GOOD:
        return f"Directional accuracy {da:.1%} < {QUALITY_DIR_ACC_GOOD:.0%}. Unreliable"
    if pf < EDGE_PF_MARGINAL and pf >= ISSUE_PF_MARGINAL_EDGE:
        return f"Profit factor {pf:.2f} < {EDGE_PF_MARGINAL:.1f}. Marginal edge"
    return None


# ── Section renderers (inlined from sections/) ──────────────────────────


def _render_data_quality_section(
    L: list[str], config: Config, heading: str | None = None
) -> None:
    if heading is None:
        heading = "## Data Quality"
    L.append(heading)
    L.append("")
    dq_path = Path(config.paths.data_quality_json)
    if not dq_path.exists():
        L.append("Data quality JSON not found. Stage 1 may not have run.")
        L.append("")
        return
    try:
        with open(dq_path) as f:
            dq = json.load(f)
    except (OSError, json.JSONDecodeError):
        logger.warning("Failed to load data quality JSON: %s", dq_path, exc_info=True)
        L.append("Data quality JSON unreadable.")
        L.append("")
        return
    L.append("Data quality check: does data explain poor results?")
    L.append("")
    L.append(_tbl_row("Metric", "Value"))
    L.append(_tbl_row("------", "-----"))
    L.append(_tbl_row("Total Bars", f"{dq.get('total_bars', 0):,}"))
    L.append(_tbl_row("Deduped Timestamps", f"{dq.get('deduped_timestamps', 0):,}"))
    L.append(_tbl_row("Calendar Gaps (all)", f"{dq.get('calendar_gaps', 0):,}"))
    L.append(_tbl_row("  - Weekend / Holiday", f"{dq.get('weekend_gaps', 0):,}"))
    L.append(_tbl_row("  - Real Gaps", f"{dq.get('real_gaps', 0):,}"))
    L.append(
        _tbl_row("Estimated Missing Bars", f"{dq.get('estimated_missing_bars', 0):,}")
    )
    L.append(_tbl_row("Largest Gap", f"{dq.get('largest_gap_bars', 0)} bars"))
    L.append(_tbl_row("Data Start", str(dq.get("start_date", "N/A"))))
    L.append(_tbl_row("Data End", str(dq.get("end_date", "N/A"))))
    L.append("")


def _render_label_design_section(
    L: list[str], config: Config, heading: str | None = None
) -> None:
    if heading is None:
        heading = "## Label Design & Methodology"
    L.append(heading)
    L.append("")
    labels_cfg = config.labels
    L.append(
        "Labels: triple-barrier method. "
        "ATR-scaled TP/SL placed symmetrically. "
        "First barrier touched = class label."
    )
    L.append("")
    L.append(_tbl_row("Parameter", "Value"))
    L.append(_tbl_row("---------", "-----"))
    L.append(_tbl_row("ATR TP multiplier", f"{labels_cfg.atr_tp_multiplier}×"))
    L.append(_tbl_row("ATR SL multiplier", f"{labels_cfg.atr_sl_multiplier}×"))
    L.append(_tbl_row("Horizon", f"{labels_cfg.horizon_bars} bars"))
    L.append(_tbl_row("Classes", str(labels_cfg.num_classes)))
    L.append(_tbl_row("Class mapping", "Short (-1) / Hold (0) / Long (+1)"))
    L.append("")
    labels_path = Path(config.paths.labels)
    if labels_path.exists():
        try:
            df = pl.read_parquet(labels_path, columns=["label"])
            total = len(df)
            L.append("**Class distribution:**")
            L.append("")
            L.append(_tbl_row("Class", "Count", "Share"))
            L.append(_tbl_row("-----", "-----", "-----"))
            for label_val, name in [(-1, "Short"), (0, "Hold"), (1, "Long")]:
                count = int((df["label"] == label_val).sum())
                pct = count / total * 100 if total > 0 else 0.0
                L.append(_tbl_row(name, f"{count:,}", f"{pct:.1f}%"))
            L.append(_tbl_row("Total", f"{total:,}", ""))
            L.append("")
        except (ComputeError, OSError):
            logger.warning("Failed to load labels for distribution", exc_info=True)


def _render_validation_methodology_section(
    L: list[str], config: Config, heading: str | None = None
) -> None:
    if heading is None:
        heading = "## Validation Methodology"
    L.append(heading)
    L.append("")
    val_cfg = config.validation
    method_label = (
        "Walk-forward (sliding window)"
        if val_cfg.method == "sliding"
        else "Static train/val/test split"
    )

    def _bars_human(bars: int, bars_per_year: int = 8760) -> str:
        if bars >= bars_per_year:
            years = bars / bars_per_year
            return f"{years:.1f}y" if years != int(years) else f"{int(years)}y"
        months = bars / (bars_per_year / 12)
        if months >= 1:
            return f"{months:.1f}mo" if months != int(months) else f"{int(months)}mo"
        weeks = bars / (bars_per_year / 52)
        return f"{weeks:.1f}w" if weeks != int(weeks) else f"{int(weeks)}w"

    L.append("Walk-forward sliding-window CV. Prevents look-ahead bias.")
    L.append("")
    L.append(_tbl_row("Parameter", "Value"))
    L.append(_tbl_row("---------", "-----"))
    L.append(_tbl_row("Method", method_label))
    L.append(
        _tbl_row(
            "Train window",
            f"{val_cfg.train_window_bars:,} bars "
            f"(~{_bars_human(val_cfg.train_window_bars)})",
        )
    )
    L.append(
        _tbl_row(
            "Test window",
            f"{val_cfg.test_window_bars:,} bars "
            f"(~{_bars_human(val_cfg.test_window_bars)})",
        )
    )
    L.append(_tbl_row("Step", f"{val_cfg.step_bars:,} bars"))
    L.append(_tbl_row("Purge gap", f"{val_cfg.purge_bars} bars at train/test boundary"))
    L.append(_tbl_row("Embargo gap", f"{val_cfg.embargo_bars} bars after purge"))
    L.append(_tbl_row("Min train bars", f"{val_cfg.min_train_bars:,}"))
    L.append("")
    L.append(
        "Purge gap: prevents label leakage at train/test boundary. "
        "Embargo: extra buffer. Ensures strict temporal separation."
    )
    L.append("")


def _render_baseline_comparison_section(
    L: list[str], config: Config, heading: str | None = None
) -> None:
    if heading is None:
        heading = "## Baseline Comparison"
    L.append(heading)
    L.append("")
    preds_path = Path(config.paths.predictions)
    if not preds_path.exists():
        L.append("Predictions not available. Baseline comparison skipped.")
        L.append("")
        return
    try:
        df = pl.read_csv(preds_path)
    except (ComputeError, OSError):
        logger.warning("Failed to load predictions for baselines", exc_info=True)
        L.append("Predictions file unreadable.")
        L.append("")
        return
    if "true_label" not in df.columns:
        L.append("true_label column missing. Baseline comparison skipped.")
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
        bl = baselines_mod.run_all(y_true, y_returns, seed=config.workflow.random_seed)
    except (ValueError, TypeError):
        logger.warning("Failed to compute baselines", exc_info=True)
        L.append("Baseline computation failed.")
        L.append("")
        return
    L.append(
        "Baselines computed on same labels. "
        "Model must beat all on directional accuracy and macro F1."
    )
    L.append("")
    L.append(_tbl_row("Strategy", "Accuracy", "Macro F1", "Dir. Accuracy"))
    L.append(_tbl_row("--------", "--------", "---------", "-------------"))
    for name, m in bl.items():
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


def _render_oof_vs_oos_section(
    L: list[str], config: Config, heading: str | None = None
) -> None:
    if heading is None:
        heading = "## OOF vs OOS Generalization Check"
    session_dir = config.paths.session_dir
    if not session_dir:
        L.append(heading)
        L.append("")
        L.append("Session directory unavailable. OOF/OOS comparison skipped.")
        L.append("")
        return
    wf_path = Path(session_dir) / "reports" / "walk_forward_history.json"
    if not wf_path.exists():
        L.append(heading)
        L.append("")
        L.append("Walk-forward history unavailable. OOF/OOS comparison skipped.")
        L.append("")
        return
    try:
        wf = json.loads(wf_path.read_text())
    except (OSError, json.JSONDecodeError):
        logger.warning(
            "Failed to load walk-forward history: %s", wf_path, exc_info=True
        )
        return
    window_details = wf.get("window_details", [])
    if not window_details:
        L.append(heading)
        L.append("")
        L.append("No window details available. OOF/OOS comparison skipped.")
        L.append("")
        return
    total_test_rows = 0
    weighted_acc = 0.0
    class_support: dict[str, int] = {"-1": 0, "0": 0, "1": 0}
    weighted_class_f1: dict[str, float] = {"-1": 0.0, "0": 0.0, "1": 0.0}
    for wd in window_details:
        test_rows = wd.get("test_rows", 0)
        if test_rows <= 0:
            continue
        total_test_rows += test_rows
        acc = wd.get("accuracy")
        if acc is not None:
            weighted_acc += acc * test_rows
        per_class = wd.get("per_class", {})
        for cls_key in ("-1", "0", "1"):
            cls_f1 = per_class.get(cls_key, {}).get("f1", 0.0)
            support = per_class.get(cls_key, {}).get("support", 0)
            class_support[cls_key] += support
            weighted_class_f1[cls_key] += cls_f1 * support
    if total_test_rows == 0:
        oof_accuracy: float | None = None
        oof_macro_f1: float | None = None
        oof_class_f1: dict[str, float | None] = {"-1": None, "0": None, "1": None}
    else:
        oof_accuracy = weighted_acc / total_test_rows
        oof_class_f1 = {}
        for cls_key in ("-1", "0", "1"):
            sup = class_support.get(cls_key, 0)
            oof_class_f1[cls_key] = (
                weighted_class_f1[cls_key] / sup if sup > 0 else None
            )
        valid_class_f1s = [v for v in oof_class_f1.values() if v is not None]
        oof_macro_f1 = float(np.mean(valid_class_f1s)) if valid_class_f1s else None
    oos_accuracy: float | None = None
    oos_macro_f1: float | None = None
    oos_class_f1: dict[str, float | None] = {"-1": None, "0": None, "1": None}
    oos_start = ""
    oos_end = ""
    preds_path = Path(config.paths.predictions)
    if preds_path.exists():
        oos_start = config.backtest.oob_start_date or ""
        oos_end = config.backtest.oob_end_date or ""
        if not oos_start:
            bt_path = (
                Path(session_dir) / "backtest" / "backtest_results.json"
                if session_dir
                else None
            )
            if bt_path and bt_path.exists():
                try:
                    bt_data = json.loads(bt_path.read_text())
                    bt_metrics = bt_data.get("metrics", {})
                    bt_start = bt_metrics.get("start")
                    bt_end = bt_metrics.get("end")
                    if bt_start and bt_end:
                        start_s = _parse_date(str(bt_start)[:19])
                        end_s = _parse_date(str(bt_end)[:19])
                        if start_s and end_s:
                            total_span = end_s - start_s
                            mid_point = start_s + timedelta(
                                seconds=total_span.total_seconds() / 2
                            )
                            oos_start = mid_point.strftime("%Y-%m-%d %H:%M:%S")
                            oos_end = str(bt_end)[:19]
                except (OSError, json.JSONDecodeError):
                    logger.warning(
                        "Failed to load backtest results for OOS range", exc_info=True
                    )
        if oos_start and oos_end:
            try:
                df = pl.read_csv(preds_path)
                if "true_label" not in df.columns or "pred_label" not in df.columns:
                    logger.warning("Predictions missing true_label/pred_label columns")
                else:
                    ts_expr = pl.col("timestamp")
                    ts_dtype = df.schema.get("timestamp")
                    if ts_dtype != pl.Datetime:
                        try:
                            ts_expr = ts_expr.str.strptime(pl.Datetime)
                        except (ComputeError, ValueError):
                            ts_expr = ts_expr.cast(pl.Datetime)
                    ts_dtype = df.schema.get("timestamp")
                    if getattr(ts_dtype, "time_zone", None):
                        ts_expr = ts_expr.dt.replace_time_zone(None)
                    start_dt = _parse_date(oos_start)
                    end_dt = _parse_date(oos_end)
                    if start_dt is not None and end_dt is not None:
                        end_dt = end_dt.replace(hour=23, minute=59, second=59)
                        oos_df = df.filter((ts_expr >= start_dt) & (ts_expr <= end_dt))
                        if len(oos_df) > 0:
                            true = oos_df["true_label"].to_numpy()
                            pred = oos_df["pred_label"].to_numpy()
                            oos_accuracy = float((true == pred).mean())
                            per_class_metrics: dict[str, dict] = {}
                            for lv, cls_key in [(-1, "-1"), (0, "0"), (1, "1")]:
                                true_mask = true == lv
                                pred_mask = pred == lv
                                recall = (
                                    float((pred[true_mask] == lv).mean())
                                    if true_mask.sum() > 0
                                    else 0.0
                                )
                                precision = (
                                    float((true[pred_mask] == lv).mean())
                                    if pred_mask.sum() > 0
                                    else 0.0
                                )
                                f1 = (
                                    2 * precision * recall / (precision + recall)
                                    if (precision + recall) > 0
                                    else 0.0
                                )
                                per_class_metrics[cls_key] = {
                                    "f1": f1,
                                    "support": int(true_mask.sum()),
                                }
                            oos_macro_f1 = float(
                                np.mean(
                                    [
                                        per_class_metrics[k]["f1"]
                                        for k in ("-1", "0", "1")
                                    ]
                                )
                            )
                            oos_class_f1 = {
                                k: per_class_metrics[k]["f1"] for k in ("-1", "0", "1")
                            }
            except (ColumnNotFoundError, ValueError, ComputeError):
                logger.warning(
                    "Failed to compute OOS prediction metrics", exc_info=True
                )
    oos_all_none = (
        oos_accuracy is None
        and oos_macro_f1 is None
        and all(v is None for v in oos_class_f1.values())
    )
    if oos_all_none and oof_accuracy is None and oof_macro_f1 is None:
        L.append(heading)
        L.append("")
        L.append("Insufficient data. No test predictions available.")
        L.append("")
        return
    L.append(heading)
    L.append("")
    oos_label = (
        f"OOS ({oos_start[:4]}–{oos_end[:4]})" if oos_start and oos_end else "OOS"
    )
    L.append(
        f"OOF: aggregated across walk-forward windows. "
        f"OOS: {'later half of backtest period' if oos_start else 'test period'}"
        f"{' (' + oos_start[:10] + ' to ' + oos_end[:10] + ')' if oos_start else ''}. "
        "Gap = overfitting signal. Tight = generalizes."
    )
    L.append("")
    L.append(_tbl_row("Metric", "OOF (Walk-Forward)", oos_label, "Delta"))
    L.append(_tbl_row("------", "-------------------", "----------------", "-----"))

    def _metric_row(name: str, oof_val: float | None, oos_val: float | None) -> None:
        oof_str = f"{oof_val * 100:.1f}%" if oof_val is not None else "N/A"
        oos_str = f"{oos_val * 100:.1f}%" if oos_val is not None else "N/A"
        if oof_val is not None and oos_val is not None:
            delta = oos_val - oof_val
            delta_str = f"{delta * 100:+.1f}pp"
        else:
            delta_str = "N/A"
        L.append(_tbl_row(name, oof_str, oos_str, delta_str))

    _metric_row("Accuracy", oof_accuracy, oos_accuracy)
    _metric_row("Macro F1", oof_macro_f1, oos_macro_f1)
    for cls_key, cls_name in [("-1", "Short"), ("0", "Flat"), ("1", "Long")]:
        _metric_row(
            f"F1 ({cls_name})", oof_class_f1.get(cls_key), oos_class_f1.get(cls_key)
        )
    L.append("")
    if oof_accuracy is not None and oos_accuracy is not None:
        gap = abs(oos_accuracy - oof_accuracy)
        if gap < 0.02:
            note = "OOF-OOS tight (<2pp). Model generalizes."
        elif gap < 0.05:
            note = "OOF-OOS gap moderate (2-5pp). Monitor overfitting."
        else:
            note = (
                "OOF-OOS gap large (>=5pp). Possible overfitting. "
                "Review feature stability."
            )
        L.append(f"**Interpretation:** {note}")
        L.append("")
    logger.info(
        "OOF vs OOS comparison: OOF acc=%.4f, OOS acc=%.4f",
        oof_accuracy or 0.0,
        oos_accuracy or 0.0,
    )


def _render_metric_zones_section(
    L: list[str], metrics: dict, trades: list[dict] | None = None, heading: str = ""
) -> None:
    if heading:
        L.append(heading)
        L.append("")
    L.append(
        "Metric classified into quality zones. "
        "🔴 = poor/dangerous, 🟡 = marginal, 🟢 = good."
    )
    L.append("")
    L.append(_tbl_row("Metric", "Value", "Zone & Rating", "Recommended"))
    L.append(_tbl_row("------", "-----", "------------", "-----------"))
    avg_wl: float | None = None
    if trades:
        wins = [t["pnl"] for t in trades if t["pnl"] > 0]
        losses = [t["pnl"] for t in trades if t["pnl"] < 0]
        if wins and losses:
            avg_wl = (sum(wins) / len(wins)) / abs(sum(losses) / len(losses)) or None
    metric_defs: list[tuple[str, str, Any, float | None]] = [
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
        emoji, zone_desc, rec = _get_zone_info(key, val)
        L.append(_tbl_row(label, fmt(val), f"{emoji} {zone_desc}", rec))
    L.append("")


# ── Table builders ──────────────────────────────────────────────────────


def _exec_table(L: list[str], metrics: dict, pred_stats: dict | None) -> None:
    if not pred_stats and not metrics:
        L.append("No metrics available.")
        return
    L.append(_tbl_row("Metric", "Value", "Zone"))
    L.append(_tbl_row("------", "-----", "----"))
    if pred_stats:
        acc = pred_stats["accuracy"]
        maj_bl = pred_stats["majority_baseline"]
        acc_gap = acc - maj_bl
        dir_acc = pred_stats["directional_accuracy"]
        per_class = pred_stats["per_class"]
        macro_f1 = float(np.mean([per_class[name]["f1"] for name in per_class]))
        L.append(
            _tbl_row("Exact Accuracy", f"{acc * 100:.1f}%", _zone("accuracy", acc))
        )
        L.append(_tbl_row("Majority Baseline", f"{maj_bl * 100:.1f}%", ""))
        L.append(_tbl_row("Acc - Baseline", f"{acc_gap * 100:+.1f}pp", ""))
        L.append(
            _tbl_row(
                "Directional Acc.",
                f"{dir_acc * 100:.1f}%",
                _zone("directional_accuracy", dir_acc),
            )
        )
        L.append(_tbl_row("Macro F1", f"{macro_f1:.3f}", ""))
    if metrics:
        rows = [
            ("Demo Return", "return_pct", metrics.get("return_pct", 0), _fmt_pct),
            (
                "Demo Max DD",
                "max_drawdown_pct",
                metrics.get("max_drawdown_pct", 0),
                _fmt_pct,
            ),
            (
                "Demo Trades",
                "num_trades",
                float(metrics.get("num_trades", 0)),
                lambda v: f"{int(v):,}",
            ),
        ]
        for label, key, val, fmt in rows:
            L.append(_tbl_row(label, fmt(val), _zone(key, val)))


def _exec_verdict(L: list[str], metrics: dict, pred_stats: dict | None) -> None:
    if not pred_stats:
        if not metrics:
            return
        L.append("Prediction metrics unavailable. Only application demo ran.")
        return
    # ML quality paragraph
    acc = pred_stats["accuracy"]
    baseline = pred_stats["majority_baseline"]
    dir_acc = pred_stats["directional_accuracy"]
    per_class = pred_stats["per_class"]
    macro_f1 = float(np.mean([per_class[name]["f1"] for name in per_class]))
    gap = acc - baseline
    if gap < 0:
        ml_quality = "weak"
        gate_msg = "Below baseline. Predictive edge not validated."
    elif acc > baseline + 0.05 and dir_acc > 0.55 and macro_f1 >= 0.45:
        ml_quality = "strong"
        gate_msg = "Above baseline. Directional edge present."
    elif dir_acc >= 0.50:
        ml_quality = "acceptable"
        gate_msg = "Slightly above baseline. Marginal directional edge."
    else:
        ml_quality = "weak"
        gate_msg = "No reliable directional edge."
    L.append(
        f"ML quality **{ml_quality}**: exact {acc:.1%} vs "
        f"baseline {baseline:.1%}, dir {dir_acc:.1%}, "
        f"macro F1 {macro_f1:.3f}. {gate_msg} "
        "Backtest = application demo. Not primary proof."
    )
    # Synthesized verdict
    model_quality, ml_reason = _assess_model_quality(pred_stats)
    if metrics:
        trading_edge, trade_reason = _assess_trading_edge(metrics)
        recommendation = _derive_recommendation(model_quality, trading_edge, metrics)
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
    # Primary issue
    if metrics:
        primary = _identify_primary_issue(metrics, pred_stats)
        if primary:
            L.append(f"**Primary issue:** {primary}.")
    else:
        L.append("**Primary issue:** No backtest metrics. Pipeline may have failed.")


def _config_table(L: list[str], config: Config) -> None:
    rows = [
        ("Data", "symbol", str(config.data.symbol)),
        ("Data", "timeframe", config.data.timeframe),
        ("Validation", "method", config.validation.method),
    ]
    if config.validation.method == "sliding":
        rows.extend(
            [
                ("Validation", "window type", "bar-based walk-forward"),
                (
                    "Validation",
                    "train/test/step bars",
                    f"{config.validation.train_window_bars}/"
                    f"{config.validation.test_window_bars}/"
                    f"{config.validation.step_bars}",
                ),
                (
                    "Validation",
                    "purge/embargo bars",
                    f"{config.validation.purge_bars}/{config.validation.embargo_bars}",
                ),
                ("Validation", "min_train_bars", str(config.validation.min_train_bars)),
            ]
        )
    else:
        rows.extend(
            [
                (
                    "Data Range",
                    "start -> end",
                    f"{config.data_range.start} -> {config.data_range.end}",
                ),
                (
                    "Walk-Forward",
                    "train_window",
                    f"{config.validation.train_window_bars} bars",
                ),
                (
                    "Walk-Forward",
                    "test_window",
                    f"{config.validation.test_window_bars} bars",
                ),
                (
                    "Walk-Forward",
                    "purge/embargo",
                    f"{config.validation.purge_bars}/{config.validation.embargo_bars}",
                ),
            ]
        )
    rows.extend(
        [
            (
                "Labels",
                "atr_mult / horizon",
                f"{config.labels.atr_tp_multiplier}"
                f"/{config.labels.atr_sl_multiplier}"
                f" / {config.labels.horizon_bars}",
            ),
            ("Stacking", "base/meta", "LogReg + RF + LGBM / Logistic Regression"),
            (
                "Stacking",
                "base/meta split",
                f"{int((1 - config.model.stacking_meta_fraction) * 100)}/"
                f"{int(config.model.stacking_meta_fraction * 100)} chronological",
            ),
            (
                "LGBM",
                "leaves/depth/lr",
                f"{config.model.num_leaves}/{config.model.max_depth}/{config.model.learning_rate}",
            ),
            (
                "LGBM",
                "estimators/subsample",
                f"{config.model.n_estimators}/{config.model.subsample}",
            ),
            ("LGBM", "feature_fraction", str(config.model.feature_fraction)),
            (
                "Backtest",
                "capital/leverage",
                f"${config.backtest.initial_capital:,.0f}/{config.backtest.leverage}:1",
            ),
            (
                "Backtest",
                "lots/conf_thr",
                f"{config.backtest.lots_per_trade}/{config.backtest.confidence_threshold}",
            ),
            (
                "Backtest",
                "stop/tp (ATR)",
                f"{config.backtest.atr_stop_multiplier}/{config.backtest.atr_tp_multiplier}",
            ),
            ("Seed", "random_seed", str(config.workflow.random_seed)),
        ]
    )
    L.append(_tbl_row("Section", "Parameter", "Value"))
    L.append(_tbl_row("-------", "---------", "-----"))
    for section, param, val in rows:
        L.append(_tbl_row(section, param, val))


def _accuracy_table(
    L: list[str], pred_stats: dict | None, config: Config | None = None
) -> None:
    if not pred_stats:
        L.append("*Prediction data not found.*")
        return
    total = pred_stats["total"]
    acc = pred_stats["accuracy"]
    dir_acc = pred_stats.get("directional_accuracy", acc)
    dir_bl = pred_stats.get("directional_baseline", DIRECTIONAL_BASELINE)
    maj_bl = pred_stats.get("majority_baseline", 0)
    acc_gap = acc - maj_bl
    L.append(_tbl_row("Metric", "Value", "Zone"))
    L.append(_tbl_row("------", "-----", "----"))
    L.append(_tbl_row("Samples", f"{total:,}", ""))
    L.append(_tbl_row("Exact Accuracy", f"{acc * 100:.1f}%", _zone("accuracy", acc)))
    L.append(
        _tbl_row(
            "Directional Acc.",
            f"{dir_acc * 100:.1f}%",
            _zone("directional_accuracy", dir_acc),
        )
    )
    L.append(_tbl_row("Dir. Baseline", f"{dir_bl * 100:.1f}%", ""))
    L.append(_tbl_row("Majority Baseline", f"{maj_bl * 100:.1f}%", ""))
    L.append(_tbl_row("Acc - Baseline", f"{acc_gap * 100:+.1f}pp", ""))
    L.append("")
    per_class = pred_stats["per_class"]
    L.append(_tbl_row("Class", "Actual", "Pred", "Recall", "F1"))
    L.append(_tbl_row("-----", "------", "----", "------", "--"))
    for name in ("Short", "Hold", "Long"):
        pc = per_class[name]
        L.append(
            _tbl_row(
                name,
                f"{pc['true_count']:,}",
                f"{pc['pred_count']:,}",
                f"{pc['recall'] * 100:.1f}%",
                f"{pc['f1']:.3f}",
            )
        )
    L.append("")
    hc = pred_stats.get("high_confidence")
    if hc:
        L.append(
            f"High-confidence (>={hc['threshold']:.0%}): "
            f"{hc['count']:,} samples ({hc['pct_of_total'] * 100:.2f}%), "
            f"accuracy {hc['accuracy'] * 100:.1f}%, "
            f"dir. acc. {hc['directional_accuracy'] * 100:.1f}%"
        )
        L.append("")
    if config is not None:
        calib_note = _calibration_summary_text(config)
        if calib_note:
            L.append(calib_note)
            L.append("")


def _feature_importance_table(L: list[str], feature_importance: dict) -> None:
    if not feature_importance:
        return
    items = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    model_count = sum(1 for n, _ in items if n.startswith("model_"))
    max_imp = max(imp for _, imp in items) if items else 0

    def fmt_imp(v: float) -> str:
        if max_imp < 1:
            return f"{v:.3f}"
        if max_imp < 100:
            return f"{v:.1f}"
        return f"{v:.0f}"

    L.append(_tbl_row("Rank", "Feature", "Source", "Score"))
    L.append(_tbl_row("----", "-------", "------", "-----"))
    for i, (name, imp) in enumerate(items, 1):
        src = "Model" if name.startswith("model_") else "Technical"
        L.append(_tbl_row(str(i), f"`{name}`", src, fmt_imp(imp)))
    L.append(
        f"Top-10: {model_count}/{len(items)} model-derived features "
        f"({model_count / len(items) * 100:.0f}%)"
    )
    L.append("")


def _backtest_params_table(L: list[str], config: Config) -> None:
    bc = config.backtest
    L.append(_tbl_row("Parameter", "Value"))
    L.append(_tbl_row("---------", "-----"))
    L.append(_tbl_row("Initial Capital", _fmt_dollar(bc.initial_capital)))
    L.append(_tbl_row("Leverage", f"{bc.leverage}:1"))
    L.append(_tbl_row("Lots/Trade", str(bc.lots_per_trade)))
    L.append(_tbl_row("ATR Stop", f"{bc.atr_stop_multiplier}x"))
    if bc.atr_tp_multiplier > 0:
        L.append(_tbl_row("ATR TP", f"{bc.atr_tp_multiplier}x"))
    L.append(_tbl_row("Confidence Thr.", str(bc.confidence_threshold)))
    L.append(_tbl_row("Spread", f"${bc.spread_ticks * config.data.tick_size:.2f}"))
    L.append(_tbl_row("Commission/lot", _fmt_dollar(bc.commission_per_lot)))
    L.append("")


def _backtest_metrics_table(L: list[str], metrics: dict, config: Config) -> None:
    if not metrics:
        L.append("*No backtest results available.*")
        return
    rows = [
        ("Return", "return_pct", _fmt_pct),
        ("Sharpe", "sharpe_ratio", _fmt_f2),
        ("Max DD", "max_drawdown_pct", _fmt_pct),
        ("Win Rate", "win_rate_pct", _fmt_pct),
        ("Profit Factor", "profit_factor", _fmt_f2),
        ("Trades", "num_trades", lambda v: f"{int(v):,}"),
    ]
    L.append(_tbl_row("Metric", "Value", "Zone"))
    L.append(_tbl_row("------", "-----", "----"))
    for label, key, fmt in rows:
        val = metrics.get(key)
        if val is None:
            continue
        L.append(_tbl_row(label, fmt(val), _zone(key, val)))
    L.append("")
    initial = config.backtest.initial_capital
    eq_final = metrics.get("equity_final", 0)
    L.append(
        f"Initial balance: {_fmt_dollar(initial)} | Final equity: ${eq_final:,.0f}"
    )
    L.append("")


def _benchmark_comparison_table(L: list[str], metrics: dict, config: Config) -> None:
    test_path = Path(config.paths.test_data)
    benchmarks = _compute_benchmark_comparison(test_path, metrics, config)
    if not benchmarks:
        L.append("Test data unavailable. Benchmark comparison skipped.")
        L.append("")
        return
    L.append(
        "Benchmarks: rough directional references. Not CFD trading-cost-equivalent."
    )
    L.append(
        f"Note: Benchmarks exclude transaction costs. Not directly comparable to "
        f"{model_label(config)} (which incurs costs)."
    )
    L.append("")
    L.append(_tbl_row("Strategy", "Return", "Sharpe", "Max DD", "Win Rate", "Trades"))
    L.append(_tbl_row("--------", "------", "------", "-------", "--------", "------"))
    for b in benchmarks:
        ret = _fmt_pct(b["return_pct"])
        sharpe = _fmt_f2(b["sharpe"])
        dd = _fmt_pct(b["max_dd_pct"])
        wr = (
            "—"
            if np.isnan(b.get("win_rate_pct", float("nan")))
            else _fmt_pct(b["win_rate_pct"])
        )
        trades = str(b.get("num_trades", "—"))
        L.append(_tbl_row(b["strategy"], ret, sharpe, dd, wr, trades))
    valid_returns = [
        b for b in benchmarks if not np.isnan(b.get("return_pct", float("nan")))
    ]
    if valid_returns:
        best_ret = max(valid_returns, key=lambda x: x["return_pct"])
        best_sharpe = max(benchmarks, key=lambda x: x.get("sharpe", -999))
        best_dd = min(benchmarks, key=lambda x: x.get("max_dd_pct", 999))
        L.append("")
        L.append(
            f"Best return: **{best_ret['strategy']}** | "
            f"Best Sharpe: **{best_sharpe['strategy']}** | "
            f"Lowest DD: **{best_dd['strategy']}**"
        )
    L.append("")


def _issues_list(
    L: list[str],
    metrics: dict,
    trades: list[dict],
    config: Config,
    pred_stats: dict | None,
) -> None:
    issues: list[tuple[str, str]] = []
    recs: list[tuple[str, str]] = []
    if not metrics:
        issues.append(("critical", "No backtest metrics. Pipeline may have failed."))
        _render_issues(L, issues, recs)
        return
    primary = _identify_primary_issue(metrics, pred_stats)
    if primary:
        issues.append(("critical", primary))
    else:
        issues.append(("info", "No critical issues."))
    recs.append(("info", "Use walk-forward validation for production readiness."))
    _render_issues(L, issues, recs)


def _render_issues(
    L: list[str], issues: list[tuple[str, str]], recs: list[tuple[str, str]]
) -> None:
    L.append("### Issues")
    L.append("")
    if not issues:
        L.append("No issues detected.")
    else:
        sorted_issues = sorted(issues, key=lambda x: SEVERITY_ORDER.get(x[0], 9))
        for i, (severity, desc) in enumerate(sorted_issues, 1):
            icon = SEVERITY_ICON.get(severity, "⚪")
            L.append(f"{i}. {icon} {desc}")
    L.append("")
    L.append("### Recommendations")
    L.append("")
    if not recs:
        L.append("No recommendations.")
    else:
        sorted_recs = sorted(recs, key=lambda x: PRIORITY_ORDER.get(x[0], 9))
        for i, (priority, desc) in enumerate(sorted_recs, 1):
            icon = PRIORITY_ICON.get(priority, "⚪")
            L.append(f"{i}. {icon} {desc}")


# ── Prediction stats loader ────────────────────────────────────────────


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
        raw_metrics = compute_all_classification_metrics(true, pred, y_proba=proba)
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


# ── Markdown builders ───────────────────────────────────────────────────


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
    _exec_table(L, metrics, pred_stats)
    _exec_verdict(L, metrics, pred_stats)
    L.append("")
    L.append("## Methodology")
    L.append("")
    _render_data_quality_section(L, config, heading="### Data & Quality")
    _render_label_design_section(L, config, heading="### Label Design")
    _render_validation_methodology_section(L, config, heading="### Validation Scheme")
    L.append("### Model Architecture")
    L.append("")
    _feature_importance_table(L, feature_importance)
    L.append("")
    L.append("## Classification Results ★")
    L.append("")
    L.append(
        "Classification metrics = primary evaluation. "
        "Directional Accuracy and Macro F1 measure direction prediction."
    )
    L.append("")
    _accuracy_table(L, pred_stats, config)
    L.append("")
    _render_baseline_comparison_section(L, config, heading="### Baseline Comparison")
    L.append("## Application Demo: Backtest")
    L.append("")
    L.append("Backtest = application demo only. Not primary evaluation criterion.")
    L.append("")
    _backtest_params_table(L, config)
    _backtest_metrics_table(L, metrics, config)
    _render_metric_zones_section(L, metrics, trades, heading="")
    L.append("")
    L.append("### Benchmark Comparison")
    L.append("")
    _benchmark_comparison_table(L, metrics, config)
    _render_oof_vs_oos_section(L, config, heading="## Generalization Assessment")
    L.append("## Issues & Recommendations")
    L.append("")
    _issues_list(L, metrics, trades, config, pred_stats)
    L.append("")
    L.append("## Appendix: Full Configuration")
    L.append("")
    _config_table(L, config)
    L.append("")
    return "\n".join(L)


def _build_eval_summary(
    pred_stats: dict | None, model_comparison_rows: list[dict]
) -> str:
    notes: list[str] = []
    notes.append("## Evaluation Summary")
    notes.append("")
    stack_row = next(
        (r for r in model_comparison_rows if r.get("source") == "current_session"),
        {},
    )
    stack_acc = stack_row.get("accuracy")
    stack_dir = stack_row.get("directional_accuracy")
    stack_row.get("macro_f1")
    base_models = [
        r
        for r in model_comparison_rows
        if r.get("source") == "walk_forward_model_comparison"
    ]
    best_base = (
        max(base_models, key=lambda r: r.get("accuracy") or 0) if base_models else {}
    )
    baselines = {
        r["model"]: r
        for r in model_comparison_rows
        if r.get("source") == "derived_baseline"
    }
    random_row = baselines.get("Random Baseline", {})
    if best_base and stack_acc is not None:
        best_acc = best_base.get("accuracy", 0)
        gap = (stack_acc - best_acc) * 100
        best_name = best_base.get("model", "base model")
        if gap < -1:
            notes.append(
                f"⚠️ **Stacking underperforms**: Hybrid Stacking accuracy "
                f"({stack_acc * 100:.1f}%) trails best base model {best_name} "
                f"({best_acc * 100:.1f}%) by **{abs(gap):.1f}pp**. "
                f"Meta-learner degrades. Base models correlated. Redundant signals."
            )
        elif gap > 1:
            notes.append(
                f"✅ **Stacking improves over base models**: +{gap:.1f}pp vs "
                f"best base ({best_name}). Meta-learner combines complementary signals."
            )
        else:
            notes.append(
                f"➡️ **Stacking ≈ best base model**: marginal difference ({gap:+.1f}pp) "
                f"vs {best_name}. Adds complexity. No clear gain."
            )
        notes.append("")
    if stack_dir is not None:
        random_dir = random_row.get("directional_accuracy", 0.5)
        lift = (stack_dir - random_dir) * 100
        if lift > 5:
            notes.append(
                f"📊 **Directional edge**: "
                f"{stack_dir * 100:.1f}% vs {random_dir * 100:.1f}% "
                f"random baseline (+{lift:.1f}pp). "
                f"Captures directional signal beyond coin-flip."
            )
        elif lift > 0:
            notes.append(
                f"📊 **Weak directional edge**: {stack_dir * 100:.1f}% vs "
                f"{random_dir * 100:.1f}% random (+{lift:.1f}pp). "
                f"Marginal. May not survive transaction costs."
            )
        else:
            notes.append(
                f"❌ **No directional edge**: {stack_dir * 100:.1f}% ≤ random baseline "
                f"({random_dir * 100:.1f}%). Fails to predict market direction."
            )
        notes.append("")
    if pred_stats:
        per_class = pred_stats.get("per_class", {})
        hold_f1 = per_class.get("Hold", {}).get("f1", 0)
        hold_recall = per_class.get("Hold", {}).get("recall", 0)
        if hold_f1 < 0.20:
            notes.append(
                f"🔍 **Hold class weakness**: Hold F1 = {hold_f1:.4f}, "
                f"recall = {hold_recall:.1%}. Struggles to identify flat periods. "
                f"Often misclassifies as Long/Short. "
                f"Common with ATR-based labeling."
            )
            notes.append("")
    if pred_stats:
        hc = pred_stats.get("high_confidence", {})
        if hc and hc.get("count", 0) > 0:
            hc_acc = hc.get("accuracy", 0)
            if hc_acc < 0.20:
                notes.append(
                    f"🔴 **High-confidence paradox**: Only {hc.get('count', 0)} "
                    f"predictions > {hc.get('threshold', 0):.0%} confidence. "
                    f"Accuracy {hc_acc * 100:.1f}%. Worse than random. "
                    f"Confidence not calibrated to correctness."
                )
                notes.append("")
    if pred_stats:
        per_class = pred_stats.get("per_class", {})
        long_n = per_class.get("Long", {}).get("support", 0)
        short_n = per_class.get("Short", {}).get("support", 0)
        if long_n > 0 and short_n > 0:
            ls_ratio = long_n / short_n
            if ls_ratio > 2 or ls_ratio < 0.5:
                notes.append(
                    f"⚖️ **Label imbalance**: Long/Short ratio = {ls_ratio:.2f}. "
                    f"Class asymmetry biases model toward majority."
                )
                notes.append("")
    notes.append("### Overall Assessment")
    notes.append("")
    if stack_acc is not None and best_base:
        best_acc = best_base.get("accuracy", 0)
        if stack_acc < best_acc:
            notes.append(
                "Hybrid stacking underperforms best base model. Issues: "
                "(1) base models correlated, "
                "(2) meta-learner too simple, "
                "(3) 3-class ATR labels add noise to Hold. "
                "Recommendations: use LightGBM solo, add confidence scores, "
                "or switch to 2-class."
            )
        else:
            notes.append(
                "Hybrid stacking outperforms base models and baselines. "
                "Meta-learner combines complementary signals."
            )
    return "\n".join(notes)


def build_model_evaluation_markdown(
    config: Config, pred_stats: dict | None, model_comparison_rows: list[dict[str, Any]]
) -> str:
    """Build compact evaluation-first markdown artifact."""
    lines: list[str] = ["# Model Evaluation", ""]
    lines.append(
        "Primary ML evidence artifact. Backtest metrics intentionally excluded."
    )
    lines.append("")
    lines.append(f"- Model: {model_label(config)}")
    lines.append(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    if not pred_stats:
        lines.append("Prediction stats unavailable.")
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
    lines.append(f"| Weighted F1 | {f4(pred_stats.get('weighted_f1'))} |")
    lines.append(f"| Balanced Accuracy | {pct(pred_stats.get('balanced_accuracy'))} |")
    lines.append(f"| Total Predictions | {pred_stats.get('total', 0):,} |")
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
    cm = pred_stats.get("confusion_matrix")
    if cm:
        lines.append("## Confusion Matrix")
        lines.append("")
        classes = [c for c in ("Short", "Hold", "Long") if c in cm]
        lines.append("| True \\ Pred | " + " | ".join(classes) + " |")
        lines.append("|---|" + "|".join("---:" for _ in classes) + "|")
        for true_cls in classes:
            row = cm.get(true_cls, {})
            cells = [f"{row.get(pred_cls, 0):,}" for pred_cls in classes]
            lines.append(f"| {true_cls} | " + " | ".join(cells) + " |")
        lines.append("")
    dcm = pred_stats.get("direction_confusion_matrix")
    if dcm:
        lines.append("## Direction Confusion Matrix (Short/Long)")
        lines.append("")
        dir_classes = [c for c in ("Short", "Long") if c in dcm]
        lines.append("| True \\ Pred | " + " | ".join(dir_classes) + " |")
        lines.append("|---|" + "|".join("---:" for _ in dir_classes) + "|")
        for true_cls in dir_classes:
            row = dcm.get(true_cls, {})
            cells = [f"{row.get(pred_cls, 0):,}" for pred_cls in dir_classes]
            lines.append(f"| {true_cls} | " + " | ".join(cells) + " |")
        lines.append("")
    hc = pred_stats.get("high_confidence")
    if hc:
        lines.append("## High-Confidence Predictions")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| Confidence Threshold | {hc.get('threshold', 0):.2f} |")
        lines.append(f"| Count | {hc.get('count', 0):,} |")
        lines.append(f"| % of Total | {hc.get('pct_of_total', 0) * 100:.2f}% |")
        lines.append(f"| Accuracy | {pct(hc.get('accuracy'))} |")
        lines.append(
            f"| Directional Accuracy | {pct(hc.get('directional_accuracy'))} |"
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
    lines.append(_build_eval_summary(pred_stats, model_comparison_rows))
    return "\n".join(lines)


# ── Report data payload ────────────────────────────────────────────────


class ReportData:
    """Structured report payload — all data needed to render and persist reports."""

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


# ── Matplotlib setup ────────────────────────────────────────────────────


def _setup_matplotlib() -> None:
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


# ── Main orchestrator ──────────────────────────────────────────────────


def compute_report_data(config: Config) -> ReportData:
    """Load data, compute metrics, render charts — return structured payload."""
    _setup_matplotlib()
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
        # Use plots module for equity curve and feature importance
        fi_path = (
            Path(config.paths.session_dir) / "reports" / "feature_importance.json"
            if config.paths.session_dir
            else out_dir.parent / "feature_importance.json"
        )
        feature_importance = load_feature_importance(fi_path)
        plot_equity_curve(
            trades, config.backtest.initial_capital, out_dir / "equity_curve.png"
        )
        plot_feature_importance(feature_importance, out_dir / "feature_importance.png")
    with console.status("[cyan]Building thesis markdown[/]"):
        pred_stats = load_prediction_stats(Path(config.paths.predictions))
        model_comparison_rows = build_model_comparison_rows(config, pred_stats)
        # Render confusion matrix if predictions available
        if pred_stats and pred_stats.get("confusion_matrix"):
            preds_path = Path(config.paths.predictions)
            if preds_path.exists():
                try:
                    df = pl.read_csv(preds_path)
                    if "true_label" in df.columns and "pred_label" in df.columns:
                        plot_confusion_matrix(
                            df["true_label"].to_numpy(),
                            df["pred_label"].to_numpy(),
                            labels=["Short", "Hold", "Long"],
                            output_path=out_dir / "confusion_matrix.png",
                        )
                except (ComputeError, ColumnNotFoundError):
                    logger.warning("Failed to render confusion matrix", exc_info=True)
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
    model_metrics_path = data.out_dir / "metrics.json"
    with model_metrics_path.open("w") as f:
        json.dump(data.pred_stats or {}, f, indent=2)
    logger.info("Model metrics saved: %s", model_metrics_path)
    model_cmp_csv = write_model_comparison_artifacts(
        data.out_dir, data.model_comparison_rows
    )
    logger.info("Model comparison saved: %s", model_cmp_csv)
