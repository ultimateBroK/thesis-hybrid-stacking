"""Shared Markdown formatting helpers for stage 6 reporting."""

from __future__ import annotations

import math

from thesis.shared.zones import get_metric_zone

_ZONE_EMOJI = {
    "excellent": "✅",
    "good": "🟢",
    "moderate": "🟡",
    "poor": "🟠",
    "dangerous": "🔴",
}


def _zone(key: str, value: float) -> str:
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


__all__ = ["_ZONE_EMOJI", "_zone", "_tbl_row", "_fmt_pct", "_fmt_f2", "_fmt_dollar"]
