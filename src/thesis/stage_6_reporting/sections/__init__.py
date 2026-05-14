"""Markdown section renderers for the thesis report."""

from __future__ import annotations

from thesis.stage_6_reporting.sections.assess import (
    MIN_TRADES_DEPLOYABLE,
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
from thesis.stage_6_reporting.sections.backtest import (
    compute_avg_win_loss_ratio,
    render_baseline_comparison_section,
    render_issues,
    render_metric_zones_section,
    render_ml_quality_paragraph,
    render_primary_issue,
    render_synthesized_verdict,
)
from thesis.stage_6_reporting.sections.data import (
    load_label_distribution,
    render_data_quality_section,
    render_label_design_section,
    render_validation_methodology_section,
)
from thesis.stage_6_reporting.sections.oof import render_oof_vs_oos_section

__all__ = [
    "MIN_TRADES_DEPLOYABLE",
    "PRIORITY_ICON",
    "PRIORITY_ORDER",
    "SEVERITY_ICON",
    "SEVERITY_ORDER",
    "assess_model_quality",
    "assess_trading_edge",
    "compute_avg_win_loss_ratio",
    "derive_recommendation",
    "get_zone_info",
    "identify_primary_issue",
    "load_label_distribution",
    "render_baseline_comparison_section",
    "render_data_quality_section",
    "render_issues",
    "render_label_design_section",
    "render_metric_zones_section",
    "render_ml_quality_paragraph",
    "render_oof_vs_oos_section",
    "render_primary_issue",
    "render_synthesized_verdict",
    "render_validation_methodology_section",
]
