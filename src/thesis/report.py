"""Report generation: markdown builder, statistics, charts, and orchestrator.

Merged from the former ``thesis.report`` package (``__init__``,
``main``, ``builder``, ``stats``).
"""

from __future__ import annotations

import json
import logging
import math
import tomllib
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from thesis.config import Config
from thesis.constants import H1_BARS_PER_YEAR
from thesis.zones import _get_metric_zone

logger = logging.getLogger("thesis.report")

# ---------------------------------------------------------------------------
# Stats helpers (formerly report/stats.py)
# ---------------------------------------------------------------------------


def _load_label_distribution(labels_path: Path) -> dict | None:
    """Compute class distribution from the labels parquet file.

    Args:
        labels_path: Path to the labels parquet file.

    Returns:
        A dictionary with class counts/percentages for ``Short``, ``Hold``, and
        ``Long``, plus ``total``; returns ``None`` when unavailable.
    """
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
        logger.warning(
            "Failed to load label distribution: %s", labels_path, exc_info=True
        )
        return None


def _load_prediction_stats(preds_path: Path) -> dict | None:
    """Compute prediction quality statistics from a predictions parquet file.

    Args:
        preds_path: Path to predictions parquet containing ``true_label``,
            ``pred_label``, and optional class-probability columns.

    Returns:
        A dictionary with overall accuracy, directional accuracy, baselines,
        per-class metrics, confusion matrix, and optional high-confidence
        stats; returns ``None`` if the file is unavailable or unreadable.
    """
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
            logger.warning(
                "Failed to load full predictions parquet; retrying required columns: %s",
                preds_path,
                exc_info=True,
            )
            df = pl.read_parquet(preds_path, columns=cols)

        true = df["true_label"].to_numpy()
        pred = df["pred_label"].to_numpy()
        total = len(true)

        # Overall accuracy: fraction of predictions matching true labels
        accuracy = float((true == pred).mean())
        # Majority baseline: accuracy if we always predict the most common class
        majority_baseline = float(max((true == lv).sum() for lv in [-1, 0, 1]) / total)

        # Directional accuracy: evaluate only on non-Hold predictions
        non_hold_mask = (true != 0) & (pred != 0)
        if non_hold_mask.sum() > 0:
            directional_correct = true[non_hold_mask] == pred[non_hold_mask]
            directional_accuracy = float(directional_correct.mean())
            directional_baseline = 0.5
        else:
            directional_accuracy = 0.0
            directional_baseline = 0.5

        # Per-class metrics
        per_class: dict = {}
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
        cm: dict = {}
        for true_lv, true_name in [(-1, "Short"), (0, "Hold"), (1, "Long")]:
            row: dict = {}
            for pred_lv, pred_name in [(-1, "Short"), (0, "Hold"), (1, "Long")]:
                row[pred_name] = int(((true == true_lv) & (pred == pred_lv)).sum())
            cm[true_name] = row

        result: dict = {
            "total": total,
            "accuracy": accuracy,
            "directional_accuracy": directional_accuracy,
            "directional_baseline": directional_baseline,
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
        logger.warning(
            "Failed to load prediction statistics: %s", preds_path, exc_info=True
        )
        return None


# ---------------------------------------------------------------------------
# Confusion cost matrix
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Benchmark comparison helpers (formerly report/stats.py)
# ---------------------------------------------------------------------------

_BARS_PER_YEAR = H1_BARS_PER_YEAR


def _annualized_sharpe(
    returns: np.ndarray, bars_per_year: int = _BARS_PER_YEAR
) -> float:
    """Compute annualized Sharpe ratio from bar returns."""
    std = float(np.std(returns, ddof=1))
    if std == 0 or np.isnan(std):
        return 0.0
    return float(np.mean(returns) / std * np.sqrt(bars_per_year))


def _max_drawdown_pct(equity: np.ndarray) -> float:
    """Compute maximum drawdown as a percentage from an equity curve."""
    if len(equity) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak * 100
    return float(abs(dd.min()))


def _build_equity_curve(
    returns: np.ndarray,
    initial_capital: float,
) -> np.ndarray:
    """Build equity curve from bar returns and initial capital."""
    equity = np.empty(len(returns) + 1)
    equity[0] = initial_capital
    for i, r in enumerate(returns):
        equity[i + 1] = equity[i] * (1.0 + r)
    return equity


def _compute_random_strategy(
    returns: np.ndarray,
    initial_capital: float,
    leverage: int,
    seed: int,
) -> dict:
    """Simulate a random long/short signal strategy."""
    rng = np.random.default_rng(seed)
    signals = rng.choice([-1, 1], size=len(returns))
    leveraged = returns * signals * leverage

    equity = _build_equity_curve(leveraged, initial_capital)
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
        "num_trades": int(np.abs(np.diff(signals)).sum() / 2 + 1),
    }


def _load_close_prices_for_benchmark(
    test_data_path: Path,
    hybrid_metrics: dict,
    config: Config,
) -> np.ndarray | None:
    """Load close prices for benchmark comparison.

    Walk-forward validation does not produce a static ``test.parquet``.
    Fall back to the full OHLCV dataset filtered by the backtest period
    recorded in the metrics.

    Args:
        test_data_path: Path to static test parquet (may not exist).
        hybrid_metrics: Backtest metrics containing ``start``/``end`` timestamps.
        config: Application configuration for resolving OHLCV path.

    Returns:
        1-D array of close prices, or ``None`` when no data is available.
    """
    # 1. Try static test split — only when validation method is actually "static"
    is_static = config.validation.method == "static"
    if test_data_path.exists() and is_static:
        try:
            df = pl.read_parquet(test_data_path, columns=["close"])
            return df["close"].to_numpy()
        except Exception:
            logger.warning(
                "Failed to load static test data for benchmarks: %s",
                test_data_path,
                exc_info=True,
            )
    elif test_data_path.exists() and not is_static:
        logger.warning(
            "Static test file found (%s) but workflow is walk-forward "
            "(method='%s') — ignoring stale test_data for benchmarks",
            test_data_path,
            config.validation.method,
        )

    # 2. Walk-forward fallback: load OHLCV and filter to backtest period
    ohlcv_path = Path(config.paths.ohlcv)
    if not ohlcv_path.exists():
        logger.warning("No OHLCV data available for benchmark fallback: %s", ohlcv_path)
        return None

    try:
        df = pl.read_parquet(ohlcv_path)
    except Exception:
        logger.warning(
            "Failed to load OHLCV for benchmarks: %s", ohlcv_path, exc_info=True
        )
        return None

    ts_col = df["timestamp"]
    if ts_col.dtype == pl.Utf8:
        ts_col = ts_col.str.to_datetime()

    bt_start = hybrid_metrics.get("start")
    bt_end = hybrid_metrics.get("end")

    if bt_start and bt_end:
        start_dt = pl.lit(str(bt_start)[:19]).str.to_datetime()
        end_dt = pl.lit(str(bt_end)[:19]).str.to_datetime()
        df = df.filter((ts_col >= start_dt) & (ts_col <= end_dt))

    if len(df) < 2:
        logger.warning("OHLCV fallback for benchmarks: insufficient bars (%d)", len(df))
        return None

    logger.info("Benchmark using OHLCV fallback: %d bars", len(df))
    return df["close"].to_numpy()


def compute_benchmark_comparison(
    test_data_path: Path,
    hybrid_metrics: dict,
    config: Config,
) -> list[dict]:
    """Compute benchmark comparison metrics for naive strategies vs hybrid model.

    Strategies computed:
        1. Buy & Hold — unleveraged, no costs.
        2. Always Long — leveraged, no timing.
        3. Random Signal — random long/short with leverage.
        4. Hybrid Model — actual backtest results.
    """
    close = _load_close_prices_for_benchmark(test_data_path, hybrid_metrics, config)
    if close is None or len(close) < 2:
        return []
    if len(close) < 2:
        return []

    initial = config.backtest.initial_capital
    leverage = config.backtest.leverage
    seed = config.workflow.random_seed

    bar_returns = np.diff(close) / close[:-1]

    # 1. Buy & Hold (unleveraged, no costs)
    bh_equity = _build_equity_curve(bar_returns, initial)
    bh_return = (bh_equity[-1] / initial - 1) * 100

    # 2. Always Long (leveraged, no timing/costs)
    al_returns = bar_returns * leverage
    al_equity = _build_equity_curve(al_returns, initial)
    al_return = (al_equity[-1] / initial - 1) * 100

    # 3. Random Signal
    random_result = _compute_random_strategy(bar_returns, initial, leverage, seed)

    results: list[dict] = [
        {
            "strategy": "Buy & Hold",
            "return_pct": bh_return,
            "sharpe": _annualized_sharpe(bar_returns),
            "max_dd_pct": _max_drawdown_pct(bh_equity),
            "win_rate_pct": float("nan"),
            "num_trades": 1,
        },
        {
            "strategy": "Always Long",
            "return_pct": al_return,
            "sharpe": _annualized_sharpe(al_returns),
            "max_dd_pct": _max_drawdown_pct(al_equity),
            "win_rate_pct": float("nan"),
            "num_trades": 1,
        },
        {
            "strategy": "Random Signal",
            **random_result,
        },
        {
            "strategy": _model_label(config),
            "return_pct": hybrid_metrics.get("return_pct", 0),
            "sharpe": hybrid_metrics.get("sharpe_ratio", 0),
            "max_dd_pct": abs(hybrid_metrics.get("max_drawdown_pct", 0)),
            "win_rate_pct": hybrid_metrics.get("win_rate_pct", 0),
            "num_trades": int(hybrid_metrics.get("num_trades", 0)),
        },
    ]

    return results


# ---------------------------------------------------------------------------
# Markdown builder (formerly report/builder.py)
# ---------------------------------------------------------------------------

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
    color, _, _ = _get_metric_zone(key, value)
    return _ZONE_EMOJI.get(color, "⚪")


def _fmt_pct(v: float) -> str:
    return f"{v:.1f}%"


def _fmt_f2(v: float) -> str:
    return f"{v:.2f}"


def _fmt_dollar(v: float) -> str:
    return f"${v:,.0f}"


def _tbl_row(*cells: str) -> str:
    return "| " + " | ".join(cells) + " |"


def _model_label(config: Config) -> str:
    """Human-readable model family label for reports."""
    architecture = config.model.architecture
    if architecture == "static":
        return "Static LightGBM"
    if architecture == "hybrid":
        return "Hybrid GRU + LightGBM"
    return f"{architecture.title()} Model"


def _static_vs_hybrid_comparison(L: list[str], config: Config) -> None:
    """Render hybrid-vs-static statistical comparison section.

    Loads walk-forward history from the current session and a sibling session
    of the opposite architecture, performs a paired t-test on per-window
    accuracy, and appends markdown lines to ``L``.

    Args:
        L: Output markdown lines.
        config: Loaded runtime configuration.
    """
    current_arch = config.model.architecture
    # Only meaningful for hybrid vs static
    if current_arch not in ("hybrid", "static"):
        return

    target_arch = "static" if current_arch == "hybrid" else "hybrid"
    current_session = config.paths.session_dir

    if not current_session:
        L.append("#### Hybrid vs Static Comparison")
        L.append("")
        L.append("*Comparison unavailable — no session directory configured.*")
        L.append("")
        return

    current_wf_path = Path(current_session) / "reports" / "walk_forward_history.json"
    if not current_wf_path.exists():
        L.append("#### Hybrid vs Static Comparison")
        L.append("")
        L.append(
            "*Comparison unavailable — walk-forward history not found for "
            f"current {current_arch} session.*"
        )
        L.append("")
        return

    # Find sibling session with opposite architecture
    results_dir = Path(current_session).parent
    sibling_session = _find_architecture_session(
        results_dir, target_arch, current_session
    )

    if sibling_session is None:
        L.append("#### Hybrid vs Static Comparison")
        L.append("")
        L.append("*Comparison unavailable — run both static and hybrid first.*")
        L.append(f"*No `{target_arch}` session found under `{results_dir}`.*")
        L.append("")
        return

    sibling_wf_path = sibling_session / "reports" / "walk_forward_history.json"
    if not sibling_wf_path.exists():
        L.append("#### Hybrid vs Static Comparison")
        L.append("")
        L.append(
            f"*Comparison unavailable — walk-forward history not found "
            f"for {target_arch} session `{sibling_session.name}`.*"
        )
        L.append("")
        return

    # Load both histories
    try:
        current_history = json.loads(current_wf_path.read_text())
        sibling_history = json.loads(sibling_wf_path.read_text())
    except Exception:
        logger.warning(
            "Failed to load walk-forward history for hybrid-vs-static comparison",
            exc_info=True,
        )
        L.append("#### Hybrid vs Static Comparison")
        L.append("")
        L.append("*Comparison unavailable — failed to load walk-forward history.*")
        L.append("")
        return

    current_windows = current_history.get("window_details", [])
    sibling_windows = sibling_history.get("window_details", [])

    if len(current_windows) < 3 or len(sibling_windows) < 3:
        L.append("#### Hybrid vs Static Comparison")
        L.append("")
        L.append(
            "*Comparison unavailable — need at least 3 windows in each "
            f"session (have {len(current_windows)}/{len(sibling_windows)}).*"
        )
        L.append("")
        return

    # Pair windows by matching test date ranges
    paired = _pair_windows_by_date(current_windows, sibling_windows)

    if len(paired) < 3:
        L.append("#### Hybrid vs Static Comparison")
        L.append("")
        L.append(
            f"*Comparison unavailable — only {len(paired)} overlapping "
            "test windows found (need ≥3).*"
        )
        L.append("")
        return

    current_accs = [p[0] for p in paired]
    sibling_accs = [p[1] for p in paired]

    # Paired t-test
    try:
        from scipy.stats import ttest_rel

        t_stat, p_value = ttest_rel(current_accs, sibling_accs)
    except Exception:
        logger.warning("ttest_rel failed", exc_info=True)
        L.append("#### Hybrid vs Static Comparison")
        L.append("")
        L.append("*Comparison unavailable — statistical test failed.*")
        L.append("")
        return

    current_mean = np.mean(current_accs)
    sibling_mean = np.mean(sibling_accs)
    delta_mean = current_mean - sibling_mean

    # Determine significance
    alpha = 0.05
    if p_value < alpha:
        if delta_mean > 0:
            result_line = (
                f"{_model_label(config)} **significantly outperforms** "
                f"{target_arch.title()} (p={p_value:.4f})"
            )
        else:
            result_line = (
                f"{target_arch.title()} **significantly outperforms** "
                f"{_model_label(config)} (p={p_value:.4f})"
            )
    else:
        result_line = (
            f"{_model_label(config)} is **not significantly different** from "
            f"{target_arch.title()} (p={p_value:.4f})"
        )

    L.append("#### Hybrid vs Static Comparison")
    L.append("")
    L.append(result_line)
    L.append("")
    L.append(_tbl_row("Metric", _model_label(config), target_arch.title(), "Delta"))
    L.append(_tbl_row("------", "------", "------", "------"))
    L.append(
        _tbl_row(
            "Mean Accuracy",
            f"{current_mean * 100:.1f}%",
            f"{sibling_mean * 100:.1f}%",
            f"{delta_mean * 100:+.1f}pp",
        )
    )
    L.append(
        _tbl_row(
            "Paired Windows",
            str(len(paired)),
            str(len(paired)),
            "",
        )
    )
    L.append(
        _tbl_row(
            "t-statistic",
            "",
            "",
            f"{t_stat:.4f}",
        )
    )
    L.append(
        _tbl_row(
            "p-value",
            "",
            "",
            f"{p_value:.4f}",
        )
    )
    L.append("")

    # Per-window delta table (first 10 windows)
    L.append("**Per-Window Accuracy Delta** (first 10 windows):")
    L.append("")
    L.append(
        _tbl_row(
            "Window",
            _model_label(config),
            target_arch.title(),
            "Delta",
        )
    )
    L.append(_tbl_row("------", "------", "------", "------"))
    for i, (c_acc, s_acc) in enumerate(paired[:10], 1):
        delta = c_acc - s_acc
        L.append(
            _tbl_row(
                str(i),
                f"{c_acc * 100:.1f}%",
                f"{s_acc * 100:.1f}%",
                f"{delta * 100:+.1f}pp",
            )
        )
    if len(paired) > 10:
        L.append(f"*... and {len(paired) - 10} more windows.*")
    L.append("")

    logger.info(
        "Hybrid vs Static comparison: %d paired windows, t=%.4f, p=%.4f, delta=%.4f",
        len(paired),
        t_stat,
        p_value,
        delta_mean,
    )


def _find_architecture_session(
    results_dir: Path, target_arch: str, exclude_session: str
) -> Path | None:
    """Find the most recent session directory with a given architecture.

    Args:
        results_dir: Directory containing session subdirectories.
        target_arch: Architecture to search for (``"static"`` or ``"hybrid"``).
        exclude_session: Session path to exclude (the current session).

    Returns:
        Path to the most recent matching session, or ``None``.
    """
    if not results_dir.exists():
        return None

    candidates: list[tuple[float, Path]] = []
    for session_dir in sorted(results_dir.iterdir()):
        if not session_dir.is_dir():
            continue
        session_str = str(session_dir)
        if session_str == str(exclude_session):
            continue

        snapshot = session_dir / "config" / "config_snapshot.toml"
        if not snapshot.exists():
            continue

        try:
            with open(snapshot, "rb") as f:
                data = tomllib.load(f)
            arch = data.get("model", {}).get("architecture", "")
            if arch == target_arch:
                # Use directory modification time for recency
                candidates.append((session_dir.stat().st_mtime, session_dir))
        except Exception:
            logger.debug(
                "Skipping session %s during architecture search",
                session_dir.name,
                exc_info=True,
            )
            continue

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _pair_windows_by_date(
    current_windows: list[dict],
    sibling_windows: list[dict],
) -> list[tuple[float, float]]:
    """Pair windows by overlapping test date ranges.

    Each window dict is expected to have ``accuracy``, ``test_dates``
    (with ``start``/``end`` keys), and ``window``.

    Args:
        current_windows: Window details from the current session.
        sibling_windows: Window details from the sibling session.

    Returns:
        List of ``(current_accuracy, sibling_accuracy)`` paired by
        best-overlapping test date range.
    """
    paired: list[tuple[float, float]] = []

    for cw in current_windows:
        if "accuracy" not in cw or cw["accuracy"] is None:
            continue
        cd = cw.get("test_dates", {})
        c_start = _parse_date(cd.get("start", ""))
        c_end = _parse_date(cd.get("end", ""))
        if c_start is None or c_end is None:
            continue

        best_sw = None
        best_overlap = timedelta.min
        for sw in sibling_windows:
            if "accuracy" not in sw or sw["accuracy"] is None:
                continue
            sd = sw.get("test_dates", {})
            s_start = _parse_date(sd.get("start", ""))
            s_end = _parse_date(sd.get("end", ""))
            if s_start is None or s_end is None:
                continue

            overlap_start = max(c_start, s_start)
            overlap_end = min(c_end, s_end)
            overlap = overlap_end - overlap_start
            if overlap > best_overlap:
                best_overlap = overlap
                best_sw = sw

        if best_sw is not None and best_overlap > timedelta(0):
            paired.append((cw["accuracy"], best_sw["accuracy"]))

    return paired


def _parse_date(date_str: str) -> datetime | None:
    """Parse a date string into a datetime, trying multiple formats."""
    if not date_str:
        return None
    for fmt in (
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S%z",
    ):
        try:
            return datetime.strptime(
                date_str[:19] if len(date_str) > 19 else date_str, fmt
            )
        except ValueError:
            continue
    return None


def _build_markdown(
    config: Config,
    metrics: dict,
    trades: list[dict],
    feature_importance: dict,
    pred_stats: dict | None,
) -> str:
    """Build concise metrics-first markdown report.

    Args:
        config: Loaded runtime configuration.
        metrics: Backtest metrics dictionary.
        trades: Backtest trades list.
        feature_importance: Feature importance values.
        pred_stats: Preloaded prediction statistics, if available.

    Returns:
        Rendered markdown report content.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    session = config.paths.session_dir or "N/A"
    L: list[str] = []

    # -- Header --
    L.append(f"# Thesis Report: {_model_label(config)} — XAU/USD")
    L.append("")
    L.append(f"> Generated: {now} | Session: `{session}`")
    L.append("")

    # -- Executive Summary --
    L.append("## Executive Summary")
    L.append("")
    _exec_table(L, metrics, pred_stats)
    _exec_verdict(L, metrics, pred_stats)
    L.append("")

    # -- Configuration --
    L.append("## Configuration")
    L.append("")
    _config_table(L, config)
    L.append("")

    # -- Model Performance --
    L.append("## Model Performance")
    L.append("")
    _accuracy_table(L, pred_stats, config)
    if config.model.architecture == "hybrid":
        _gru_summary(L, config)
    _feature_importance_table(L, feature_importance)
    L.append("")

    # -- Backtest Results --
    L.append("## Backtest Results")
    L.append("")
    _backtest_params_table(L, config)
    _backtest_metrics_table(L, metrics, config)
    L.append("")

    # -- Benchmark Comparison --
    L.append("## Benchmark Comparison")
    L.append("")
    _benchmark_comparison_table(L, metrics, config)

    # -- Hybrid vs Static Comparison --
    _static_vs_hybrid_comparison(L, config)

    # -- Issues & Recommendations --
    L.append("## Issues & Recommendations")
    L.append("")
    _issues_list(L, metrics, trades, config, pred_stats)
    L.append("")

    return "\n".join(L)


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _exec_table(L: list[str], metrics: dict, pred_stats: dict | None) -> None:
    """Key ML-first metrics with application-demo metrics second.

    Args:
        L: Output markdown lines.
        metrics: Backtest metrics dictionary.
        pred_stats: Preloaded prediction statistics.
    """
    if not pred_stats and not metrics:
        L.append("*No metrics available.*")
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


def _assess_model_quality(pred_stats: dict) -> tuple[str, str]:
    """Classify ML quality into POOR / FAIR / GOOD with a short reason.

    Args:
        pred_stats: Preloaded prediction statistics.

    Returns:
        (quality_label, reason_phrase) — e.g. ("POOR", "acc below baseline").
    """
    acc = pred_stats["accuracy"]
    baseline = pred_stats["majority_baseline"]
    dir_acc = pred_stats["directional_accuracy"]
    per_class = pred_stats["per_class"]
    macro_f1 = float(np.mean([per_class[name]["f1"] for name in per_class]))

    gap = acc - baseline
    if gap < 0:
        return ("POOR", "acc below baseline")
    if acc > baseline + 0.05 and dir_acc > 0.55 and macro_f1 >= 0.45:
        return ("GOOD", "above baseline with directional edge")
    if dir_acc >= 0.50:
        return ("FAIR", "slightly above baseline, marginal edge")
    return ("POOR", "no reliable directional edge")


def _assess_trading_edge(metrics: dict) -> tuple[str, str]:
    """Classify trading edge into NEGATIVE / MARGINAL / POSITIVE.

    Args:
        metrics: Backtest metrics dictionary.

    Returns:
        (edge_label, reason_phrase) — e.g. ("NEGATIVE", "PF<1.0").
    """
    pf = metrics.get("profit_factor", 0)
    sharpe = metrics.get("sharpe_ratio", 0)
    ret = metrics.get("return_pct", 0)

    if pf < 1.0 or sharpe < 0 or ret < 0:
        return ("NEGATIVE", f"PF={pf:.2f}" if pf > 0 else "PF<1.0")
    if sharpe < 1.0 or pf < 1.5:
        return ("MARGINAL", f"PF={pf:.2f}, Sharpe={sharpe:.2f}")
    return ("POSITIVE", f"PF={pf:.2f}, Sharpe={sharpe:.2f}")


def _derive_recommendation(ml_quality: str, trading_edge: str, metrics: dict) -> str:
    """Produce a deployment recommendation from model quality + trading edge.

    Args:
        ml_quality: "POOR", "FAIR", or "GOOD".
        trading_edge: "NEGATIVE", "MARGINAL", or "POSITIVE".
        metrics: Backtest metrics dictionary.

    Returns:
        Recommendation string, e.g. "NOT DEPLOYABLE without fixes".
    """
    n_trades = int(metrics.get("num_trades", 0)) if metrics else 0

    if ml_quality == "POOR" or trading_edge == "NEGATIVE":
        return "NOT DEPLOYABLE without fixes"
    if n_trades < 30:
        return "NOT DEPLOYABLE — insufficient trades for validation"
    if ml_quality == "FAIR" and trading_edge == "MARGINAL":
        return "DEPLOYABLE with caution — marginal edge"
    if ml_quality == "GOOD" and trading_edge == "POSITIVE":
        return "DEPLOYABLE"
    return "DEPLOYABLE with caution"


def _identify_primary_issue(metrics: dict, pred_stats: dict | None) -> str | None:
    """Return the single most critical issue description, or None.

    Issues are ranked by severity (critical > warning > info), then by
    impact (e.g. zero trades beats low win rate).

    Args:
        metrics: Backtest metrics dictionary.
        pred_stats: Preloaded prediction statistics.

    Returns:
        Most severe issue description, or ``None`` if none found.
    """
    n_trades = int(metrics.get("num_trades", 0)) if metrics else 0
    sharpe = metrics.get("sharpe_ratio", 0) if metrics else 0
    pf = metrics.get("profit_factor", 0) if metrics else 0
    dd = abs(metrics.get("max_drawdown_pct", 0)) if metrics else 0
    ret = metrics.get("return_pct", 0) if metrics else 0
    wr = metrics.get("win_rate_pct", 0) if metrics else 0
    dir_acc = pred_stats.get("directional_accuracy", 0) if pred_stats else 0

    # Ordered check: first match is most critical
    checks: list[tuple[bool, str]] = [
        (
            n_trades == 0,
            "Zero trades executed — model produces no actionable signals",
        ),
        (
            n_trades > 0 and n_trades < 30,
            f"Only {n_trades} trades — statistically unreliable results",
        ),
        (
            sharpe < 0,
            f"Sharpe {sharpe:.2f} is negative — strategy underperforms risk-free rate",
        ),
        (
            dd > 50,
            f"Max drawdown {dd:.1f}% > 50% — catastrophic capital erosion",
        ),
        (
            pf < 1.0,
            f"Profit factor {pf:.2f} < 1.0 — strategy loses money on average",
        ),
        (
            dir_acc > 0 and dir_acc < 0.50,
            f"Directional accuracy {dir_acc:.1%} < 50% — predicts worse than random",
        ),
        (
            ret < -50,
            f"Return {ret:.0f}% — severe capital loss",
        ),
        (
            pf < 1.2 and pf >= 1.0,
            f"Profit factor {pf:.2f} < 1.2 — barely covers transaction costs",
        ),
        (
            sharpe < 0.5 and sharpe >= 0,
            f"Sharpe {sharpe:.2f} < 0.5 — poor risk-adjusted returns",
        ),
        (
            dd > 30 and dd <= 50,
            f"Max drawdown {dd:.1f}% exceeds 30% threshold",
        ),
        (
            n_trades >= 30 and n_trades < 100,
            f"Only {n_trades} trades — marginal sample size",
        ),
        (
            sharpe < 1.0 and sharpe >= 0.5,
            f"Sharpe {sharpe:.2f} < 1.0 — below professional threshold",
        ),
        (
            ret > 500,
            f"Return {ret:.0f}% suspiciously high — verify for overfitting",
        ),
        (
            dd > 20 and dd <= 30,
            f"Max drawdown {dd:.1f}% > 20% — elevated for CFD trading",
        ),
        (
            wr < 40 and wr >= 0,
            f"Win rate {wr:.1f}% < 40% — below trading viability",
        ),
        (
            dir_acc > 0 and dir_acc < 0.55,
            f"Directional accuracy {dir_acc:.1%} < 55% — unreliable",
        ),
        (
            pf < 1.5 and pf >= 1.2,
            f"Profit factor {pf:.2f} < 1.5 — marginal edge",
        ),
    ]
    for condition, msg in checks:
        if condition:
            return msg
    return None


def _exec_verdict(L: list[str], metrics: dict, pred_stats: dict | None) -> None:
    """One-paragraph ML-first overall assessment with synthesized verdict.

    Produces:
    1. ML quality assessment paragraph.
    2. Synthesized verdict line (model quality + trading edge + recommendation).
    3. Primary issue identification.
    4. Application demo summary line (if metrics available).

    Args:
        L: Output markdown lines.
        metrics: Backtest metrics dictionary.
        pred_stats: Preloaded prediction statistics.
    """
    if not pred_stats:
        if not metrics:
            return
        L.append("Prediction metrics are unavailable; only the application demo ran.")
        return

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

    # ── Synthesized zone-based verdict ──
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

    # ── Primary issue ──
    if metrics:
        primary = _identify_primary_issue(metrics, pred_stats)
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
        f"with Sharpe {sharpe:.2f}, win rate {wr:.1f}%, "
        f"max drawdown {dd:.1f}%."
    )


def _config_table(L: list[str], config: Config) -> None:
    """Key hyperparameters in one table."""
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
                (
                    "Validation",
                    "min_train_bars",
                    str(config.validation.min_train_bars),
                ),
            ]
        )
    else:
        rows.extend(
            [
                (
                    "Split",
                    "train",
                    f"{config.splitting.train_start} → {config.splitting.train_end}",
                ),
                (
                    "Split",
                    "val",
                    f"{config.splitting.val_start} → {config.splitting.val_end}",
                ),
                (
                    "Split",
                    "test",
                    f"{config.splitting.test_start} → {config.splitting.test_end}",
                ),
                (
                    "Split",
                    "purge/embargo",
                    f"{config.splitting.purge_bars}/{config.splitting.embargo_bars}",
                ),
            ]
        )

    rows.extend(
        [
            (
                "Labels",
                "atr_mult / horizon",
                f"{config.labels.atr_tp_multiplier}/{config.labels.atr_sl_multiplier} / {config.labels.horizon_bars}",
            ),
            (
                "GRU",
                "hidden/layers/seq",
                f"{config.gru.hidden_size}/{config.gru.num_layers}/{config.gru.sequence_length}",
            ),
            (
                "GRU",
                "lr/dropout/epochs",
                f"{config.gru.learning_rate}/{config.gru.dropout}/{config.gru.epochs}",
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


def _compute_ece_numpy(
    proba: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error (ECE) from NumPy arrays.

    Partitions predictions into ``n_bins`` equal-width confidence bins
    and measures the absolute difference between average confidence
    and accuracy within each bin, weighted by bin size.

    Args:
        proba: Softmax probabilities with shape ``(N, C)``.
        labels: Ground-truth class indices with shape ``(N,)``.
        n_bins: Number of confidence bins (default 10).

    Returns:
        ECE value (0.0 = perfectly calibrated).
    """
    confidences = proba.max(axis=1)
    predictions = proba.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float64)

    ece = 0.0
    for i in range(n_bins):
        bin_lower = i / n_bins
        bin_upper = (i + 1) / n_bins
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = in_bin.sum()
        if bin_size > 0:
            bin_conf = confidences[in_bin].mean()
            bin_acc = accuracies[in_bin].mean()
            ece += (bin_size / len(proba)) * abs(bin_conf - bin_acc)

    return ece


def _calibration_summary_text(config: Config) -> str | None:
    """Compute a one-paragraph calibration reliability note.

    Reads predicted probabilities and true labels from the predictions
    parquet file, computes ECE, and returns a human-readable summary
    of whether confidence scores appear calibrated.

    Args:
        config: Loaded runtime configuration.

    Returns:
        Calibration summary string, or ``None`` if the predictions file
        is unavailable or missing probability columns.
    """
    preds_path = Path(config.paths.predictions)
    if not preds_path.exists():
        return None

    proba_cols = [
        "pred_proba_class_minus1",
        "pred_proba_class_0",
        "pred_proba_class_1",
    ]
    try:
        df = pl.read_parquet(preds_path)
    except Exception:
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
    labels = df["true_label"].to_numpy()

    # Map label values (-1, 0, 1) → class indices (0, 1, 2)
    class_indices = np.zeros(len(labels), dtype=np.int64)
    class_indices[labels == 0] = 1
    class_indices[labels == 1] = 2
    # class_indices[labels == -1] stays 0

    ece = _compute_ece_numpy(proba, class_indices)

    if ece < 0.05:
        quality = "well-calibrated"
        note = (
            f"**Calibration**: ECE = {ece:.4f} — confidence scores are **{quality}** "
            "(ECE < 0.05). Predicted probabilities closely match observed frequencies."
        )
    elif ece < 0.15:
        quality = "moderately calibrated"
        note = (
            f"**Calibration**: ECE = {ece:.4f} — confidence scores are **{quality}** "
            "(0.05 ≤ ECE < 0.15). Probabilities are somewhat aligned with outcomes; "
            "the model may be slightly over- or under-confident in some bins."
        )
    else:
        quality = "poorly calibrated"
        note = (
            f"**Calibration**: ECE = {ece:.4f} — confidence scores are **{quality}** "
            "(ECE ≥ 0.15). Predicted probabilities do not reliably reflect true "
            "likelihoods. Consider temperature scaling or isotonic regression."
        )

    logger.info("Calibration summary: ECE=%.4f (%s)", ece, quality)
    return note


def _accuracy_table(
    L: list[str], pred_stats: dict | None, config: Config | None = None
) -> None:
    """Model accuracy: exact + directional + per-class + calibration.

    Args:
        L: Output markdown lines.
        pred_stats: Preloaded prediction statistics.
        config: Optional application configuration for calibration check.
    """
    if not pred_stats:
        L.append("*Prediction data not found.*")
        return

    total = pred_stats["total"]
    acc = pred_stats["accuracy"]
    dir_acc = pred_stats.get("directional_accuracy", acc)
    dir_bl = pred_stats.get("directional_baseline", 0.5)
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

    # Per-class
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

    # Confidence filtering
    hc = pred_stats.get("high_confidence")
    if hc:
        hc_ratio = hc["count"] / total if total else 0.0
        L.append(
            f"High-confidence (≥{hc['threshold']:.0%}): "
            f"{hc['count']:,} samples ({hc_ratio * 100:.2f}%), "
            f"accuracy {hc['accuracy'] * 100:.1f}%, "
            f"dir. acc. {hc['directional_accuracy'] * 100:.1f}%"
        )
        L.append("")

    # Calibration reliability note (after confidence section)
    if config is not None:
        calib_note = _calibration_summary_text(config)
        if calib_note:
            L.append(calib_note)
            L.append("")


def _gru_summary(L: list[str], config: Config) -> None:
    """GRU architecture summary line (hybrid only — caller guards architecture)."""
    gru = config.gru
    L.append(
        f"GRU: input={gru.input_size}, hidden={gru.hidden_size}, "
        f"layers={gru.num_layers}, seq={gru.sequence_length}, "
        f"dropout={gru.dropout}, epochs≤{gru.epochs}, patience={gru.patience}"
    )
    L.append("")


def _feature_importance_table(L: list[str], feature_importance: dict) -> None:
    """Top-10 feature importance."""
    if not feature_importance:
        return
    items = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    gru_count = sum(1 for n, _ in items if n.startswith("gru_"))
    L.append(_tbl_row("Rank", "Feature", "Source", "Score"))
    L.append(_tbl_row("----", "-------", "------", "-----"))
    for i, (name, imp) in enumerate(items, 1):
        src = "GRU" if name.startswith("gru_") else "Technical"
        L.append(_tbl_row(str(i), f"`{name}`", src, f"{imp:.0f}"))
    L.append(
        f"Top-10: {gru_count}/{len(items)} GRU features ({gru_count / len(items) * 100:.0f}%)"
    )
    L.append("")


def _backtest_params_table(L: list[str], config: Config) -> None:
    """Backtest simulation parameters."""
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
    L.append(
        _tbl_row(
            "Spread",
            f"${bc.spread_ticks * config.data.tick_size:.2f}",
        )
    )
    L.append(_tbl_row("Commission/lot", _fmt_dollar(bc.commission_per_lot)))
    L.append("")


def _backtest_metrics_table(L: list[str], metrics: dict, config: Config) -> None:
    """Core backtest metrics with zone indicators."""
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
    """Compare benchmarks against the configured model architecture."""
    test_path = Path(config.paths.test_data)
    benchmarks = compute_benchmark_comparison(test_path, metrics, config)
    if not benchmarks:
        L.append("*Test data unavailable — benchmark comparison skipped.*")
        L.append("")
        return

    L.append(
        "*Benchmarks are rough directional references and are not "
        "trading-cost-equivalent to the CFD backtest strategy.*"
    )
    L.append(
        "*Note: Benchmarks exclude transaction costs (spread, slippage, "
        f"commission); not directly comparable to the {_model_label(config)} model "
        "which incurs all three.*"
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


_SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}
_PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}
_SEVERITY_ICON = {"critical": "🔴", "warning": "🟡", "info": "✅"}
_PRIORITY_ICON = {"high": "🔴", "medium": "🟡", "low": "🔵", "info": "✅"}


def _render_issues(
    L: list[str],
    issues: list[tuple[str, str]],
    recs: list[tuple[str, str]],
) -> None:
    """Render sorted issues and recommendations into markdown lines."""
    L.append("### Issues")
    L.append("")
    if not issues:
        L.append("*No issues detected.*")
    else:
        sorted_issues = sorted(issues, key=lambda x: _SEVERITY_ORDER.get(x[0], 9))
        for i, (severity, desc) in enumerate(sorted_issues, 1):
            icon = _SEVERITY_ICON.get(severity, "⚪")
            L.append(f"{i}. {icon} {desc}")
    L.append("")

    L.append("### Recommendations")
    L.append("")
    if not recs:
        L.append("*No specific recommendations.*")
    else:
        sorted_recs = sorted(recs, key=lambda x: _PRIORITY_ORDER.get(x[0], 9))
        for i, (priority, desc) in enumerate(sorted_recs, 1):
            icon = _PRIORITY_ICON.get(priority, "⚪")
            L.append(f"{i}. {icon} {desc}")


def _count_features(config: Config) -> int:
    """Count total features from the features parquet or GRU config."""
    features_path = Path(config.paths.features)
    if features_path.exists():
        try:
            import polars as pl  # noqa: F811

            df = pl.read_parquet(features_path)
            exclude = {"timestamp", "open", "high", "low", "close", "volume", "label"}
            return sum(1 for c in df.columns if c not in exclude)
        except Exception:
            logger.warning(
                "Failed to count features from features parquet: %s",
                features_path,
                exc_info=True,
            )
    return config.gru.hidden_size + len(config.features.static_feature_cols)


def _issues_list(
    L: list[str],
    metrics: dict,
    trades: list[dict],
    config: Config,
    pred_stats: dict | None,
) -> None:
    """High-signal issues and recommendations from report metrics.

    Only the most critical checks are included to keep the report focused.

    Args:
        L: Output markdown lines.
        metrics: Backtest metrics dictionary.
        trades: Backtest trades list.
        config: Loaded runtime configuration.
        pred_stats: Preloaded prediction statistics.
    """
    issues: list[tuple[str, str]] = []
    recs: list[tuple[str, str]] = []

    if not metrics:
        issues.append(("critical", "No backtest metrics — pipeline may have failed."))
        _render_issues(L, issues, recs)
        return

    sharpe = metrics.get("sharpe_ratio", 0)
    dd = abs(metrics.get("max_drawdown_pct", 0))
    pf = metrics.get("profit_factor", 0)
    n_trades = int(metrics.get("num_trades", 0))
    dir_acc = pred_stats.get("directional_accuracy", 0) if pred_stats else 0

    # — Core high-signal checks —

    if n_trades == 0:
        issues.append(
            (
                "critical",
                "Zero trades executed — model produces no actionable signals in test period.",
            )
        )

    if sharpe < 0:
        issues.append(
            (
                "critical",
                f"Sharpe {sharpe:.2f} is negative — strategy underperforms risk-free rate.",
            )
        )

    if dd > 50:
        issues.append(
            (
                "critical",
                f"Max drawdown {dd:.1f}% > 50% — catastrophic capital erosion.",
            )
        )

    if pf < 1.0:
        issues.append(
            (
                "critical",
                f"Profit factor {pf:.2f} < 1.0 — strategy loses money on average.",
            )
        )

    if dir_acc > 0 and dir_acc < 0.50:
        issues.append(
            (
                "critical",
                f"Directional accuracy {dir_acc:.1%} < 50% — model predicts worse than random.",
            )
        )

    if not issues:
        issues.append(("info", "No critical issues identified."))

    # — Single actionable recommendation —
    if not recs:
        recs.append(
            (
                "info",
                "Consider walk-forward validation for production readiness and robustness testing.",
            )
        )

    _render_issues(L, issues, recs)


# ---------------------------------------------------------------------------
# Chart helpers (formerly report/main.py)
# ---------------------------------------------------------------------------


def _plot_equity_curve(trades: list[dict], config: Config, out_dir: Path) -> None:
    """Render and save an equity curve image from trade history."""
    if not trades:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times, equity = _build_equity_series(trades, config.backtest.initial_capital)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(times, equity, linewidth=1)
    ax.set_title("Equity Curve")
    ax.set_ylabel("Equity (USD)")
    ax.set_xlabel("Date")
    fig.autofmt_xdate()
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / "equity_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart saved: equity_curve.png")


def _build_equity_series(
    trades: list[dict], initial_capital: float
) -> tuple[list, list]:
    """Build timestamp and cumulative equity series from trades.

    Note:
        The equity curve is trade-by-trade (closed-trade PnL), not
        mark-to-market. Intra-trade drawdowns are not visible.
    """
    times = [pd.to_datetime(trades[0]["entry_time"])]
    equity = [initial_capital]
    for t in trades:
        times.append(pd.to_datetime(t["exit_time"]))
        equity.append(equity[-1] + t["pnl"])
    return times, equity


def _plot_feature_importance(feature_importance: dict, out_dir: Path) -> None:
    """Render and save a top-20 feature-importance chart."""
    if not feature_importance:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = dict(
        sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20]
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(list(top.keys()), list(top.values()))
    ax.set_title("Feature Importance (Top 20)")
    ax.invert_yaxis()
    fig.savefig(out_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart saved: feature_importance.png")


def _load_feature_importance(config: Config, out_dir: Path) -> dict:
    """Load feature-importance JSON from session report outputs."""
    fi_path = (
        Path(config.paths.session_dir) / "reports" / "feature_importance.json"
        if config.paths.session_dir
        else out_dir.parent / "feature_importance.json"
    )
    if not fi_path.exists():
        return {}
    with open(fi_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Public entry point (formerly report/main.py → generate_report)
# ---------------------------------------------------------------------------


def generate_report(config: Config) -> None:
    """Generate thesis report with static charts and markdown.

    Args:
        config: Loaded application configuration.
    """
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

    _plot_equity_curve(trades, config, out_dir)
    feature_importance = _load_feature_importance(config, out_dir)
    _plot_feature_importance(feature_importance, out_dir)
    # Markdown Report
    pred_stats = _load_prediction_stats(Path(config.paths.predictions))
    md = _build_markdown(
        config,
        metrics,
        trades,
        feature_importance,
        pred_stats,
    )
    report_path = Path(config.paths.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(md)
    logger.info("Report saved: %s", report_path)
