"""Data quality evidence for the thesis report.

OHLCV consistency, missing-bar gaps, label distribution, outlier detection,
and combined report rendered as markdown.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl

from thesis.shared.constants import timeframe_to_ms
from thesis.shared.data_quality import (
    check_gap_report,
    check_outlier_returns,
    classify_calendar_gaps,
    validate_ohlcv,
)


def compute_ohlcv_consistency(df: pl.DataFrame) -> dict[str, Any]:
    """Check OHLCV relationships via validate_ohlcv."""
    result = validate_ohlcv(df)
    return {
        "total_rows": result["total_rows"],
        "ohlc_violations": result["ohlc_violations"],
        "price_negative_count": result["price_negative_count"],
        "is_consistent": result["is_valid"],
    }


def compute_missing_bar_stats(
    df: pl.DataFrame, expected_interval: str = "1h"
) -> dict[str, Any]:
    """Analyse gaps between consecutive bars."""
    total_bars = len(df)

    if "timestamp" not in df.columns or total_bars < 2:
        return {
            "total_bars": total_bars,
            "gaps_found": 0,
            "weekend_gaps": 0,
            "missing_ratio": 0.0,
        }

    timeframe_ms = timeframe_to_ms(expected_interval)
    result = check_gap_report(df, timeframe_ms)

    calendar = classify_calendar_gaps(df, timeframe_ms)
    ts = df["timestamp"].sort().to_list()
    weekend_heuristic = 0
    for prev_ts, curr_ts in zip(ts[:-1], ts[1:]):
        delta_ms = int((curr_ts - prev_ts).total_seconds() * 1000)
        if delta_ms <= timeframe_ms:
            continue
        if (
            prev_ts.weekday() >= 5
            or curr_ts.weekday() >= 5
            or curr_ts.weekday() < prev_ts.weekday()
        ):
            weekend_heuristic += 1

    gaps_found = result["gap_count"]
    weekend_gaps = max(calendar.calendar_gap_count, weekend_heuristic)
    missing_ratio = (
        (gaps_found - calendar.real_gap_count) / total_bars
        if total_bars > 0
        else 0.0
    )

    return {
        "total_bars": total_bars,
        "gaps_found": gaps_found,
        "weekend_gaps": weekend_gaps,
        "real_gaps": calendar.real_gap_count,
        "missing_ratio": round(max(missing_ratio, 0.0), 6),
        "calendar_warnings": calendar.warnings,
    }


def compute_label_distribution(
    labels: npt.NDArray, classes: list[int] | None = None
) -> dict[str, Any]:
    """Count and percentage of each label class, plus imbalance ratio."""
    if classes is None:
        classes = [-1, 0, 1]

    total = len(labels)
    counts: dict[int, int] = {}
    percentages: dict[int, float] = {}

    for c in classes:
        cnt = int((labels == c).sum())
        counts[c] = cnt
        percentages[c] = round(cnt / total * 100, 2) if total > 0 else 0.0

    non_zero = [counts[c] for c in classes if counts[c] > 0]
    imbalance_ratio = (
        round(max(non_zero) / min(non_zero), 2) if len(non_zero) >= 2 else 0.0
    )

    return {
        "total": total,
        "counts": counts,
        "percentages": percentages,
        "imbalance_ratio": imbalance_ratio,
    }


def compute_outlier_returns(
    df: pl.DataFrame, z_threshold: float = 5.0
) -> dict[str, Any]:
    """Flag returns that exceed z_threshold standard deviations."""
    result = check_outlier_returns(df, z_threshold)
    outlier_count = result["outlier_count"]

    close = df["close"].cast(pl.Float64).to_numpy() if "close" in df.columns else None
    log_returns = (
        np.diff(np.log(close)) if close is not None and len(close) >= 2 else None
    )

    n_returns = len(log_returns) if log_returns is not None else 0
    outlier_ratio = round(outlier_count / n_returns, 6) if n_returns > 0 else 0.0
    max_return = float(np.max(log_returns)) if n_returns > 0 else 0.0
    min_return = float(np.min(log_returns)) if n_returns > 0 else 0.0

    outlier_dates: list[str] = []
    if "timestamp" in df.columns:
        ts = df["timestamp"].to_list()
        for idx in result["outlier_indices"]:
            if idx < len(ts):
                outlier_dates.append(str(ts[idx]))

    return {
        "outlier_count": outlier_count,
        "outlier_ratio": outlier_ratio,
        "max_return": max_return,
        "min_return": min_return,
        "outlier_dates": outlier_dates,
    }


def render_data_quality_markdown(stats: dict[str, Any]) -> str:
    """Render all quality stats as a markdown section."""
    lines: list[str] = ["## Data Quality Report", ""]

    ohlcv = stats.get("ohlcv_consistency", {})
    lines.append("### OHLCV Consistency")
    lines.append("")
    lines.append(f"- Total rows: {ohlcv.get('total_rows', 'N/A')}")
    lines.append(f"- OHLC violations: {ohlcv.get('ohlc_violations', 'N/A')}")
    lines.append(f"- Negative prices: {ohlcv.get('price_negative_count', 'N/A')}")
    lines.append(f"- Consistent: {'Yes' if ohlcv.get('is_consistent') else 'No'}")
    lines.append("")

    mb = stats.get("missing_bars", {})
    lines.append("### Missing Bar Analysis")
    lines.append("")
    lines.append(f"- Total bars: {mb.get('total_bars', 'N/A')}")
    lines.append(f"- Gaps found: {mb.get('gaps_found', 'N/A')}")
    lines.append(f"- Weekend gaps: {mb.get('weekend_gaps', 'N/A')}")
    lines.append(f"- Missing ratio: {mb.get('missing_ratio', 0.0):.6f}")
    lines.append("")

    lbl = stats.get("label_distribution")
    if lbl:
        lines.append("### Label Distribution")
        lines.append("")
        lines.append(f"- Total samples: {lbl.get('total', 'N/A')}")
        for cls_val, name in {**{-1: "Short", 0: "Hold", 1: "Long"}}.items():
            cnt = lbl.get("counts", {}).get(cls_val, "N/A")
            pct = lbl.get("percentages", {}).get(cls_val, "N/A")
            lines.append(f"- {name} ({cls_val}): {cnt} ({pct}%)")
        lines.append(f"- Imbalance ratio: {lbl.get('imbalance_ratio', 'N/A')}")
        lines.append("")

    out = stats.get("outlier_returns", {})
    lines.append("### Outlier Returns")
    lines.append("")
    lines.append(f"- Outlier count: {out.get('outlier_count', 'N/A')}")
    lines.append(f"- Outlier ratio: {out.get('outlier_ratio', 0.0):.6f}")
    lines.append(f"- Max return: {out.get('max_return', 0.0):.8f}")
    lines.append(f"- Min return: {out.get('min_return', 0.0):.8f}")
    lines.append("")

    return "\n".join(lines)


def compute_data_quality_report(
    df: pl.DataFrame, labels: npt.NDArray | None = None
) -> dict[str, Any]:
    """Run all quality checks and return a comprehensive dict."""
    stats: dict[str, Any] = {
        "ohlcv_consistency": compute_ohlcv_consistency(df),
        "missing_bars": compute_missing_bar_stats(df),
        "outlier_returns": compute_outlier_returns(df),
    }

    if labels is not None:
        stats["label_distribution"] = compute_label_distribution(labels)

    stats["markdown"] = render_data_quality_markdown(stats)
    return stats
