"""Shared Markdown formatting helpers for stage 6 reporting.

This module centralizes tiny formatting helpers that were previously duplicated
across report-generation modules. Keep these functions stable: many report
sections rely on their exact output.
"""

from __future__ import annotations

import math

from thesis.shared.zones import get_metric_zone

# Zone emoji mapping
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


def _tbl_row(*cells: str) -> str:
    """Format cells as a markdown table row."""
    return "| " + " | ".join(cells) + " |"


def _fmt_pct(v: float) -> str:
    return f"{v:.1f}%"


def _fmt_f2(v: float) -> str:
    return f"{v:.2f}"


def _fmt_dollar(v: float) -> str:
    return f"${v:,.0f}"


__all__ = ["_ZONE_EMOJI", "_zone", "_tbl_row", "_fmt_pct", "_fmt_f2", "_fmt_dollar"]
