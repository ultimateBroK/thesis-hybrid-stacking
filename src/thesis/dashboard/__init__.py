"""Dashboard package for thesis session visualization."""

from thesis.dashboard.backtest import render_backtest_section
from thesis.dashboard.cards import render_metric_card, render_zoned_metric
from thesis.dashboard.data import render_data_section
from thesis.dashboard.main import main
from thesis.dashboard.model import render_model_section
from thesis.dashboard.reports import render_reports_section
from thesis.dashboard.session import (
    find_sessions,
    load_config,
    parse_session_meta,
    session_selector_fragment,
)
from thesis.dashboard.shared import (
    date_only,
    render_chart,
    render_config_summary,
    render_trade_direction_summary,
    trim_generated_visual_sections,
)
from thesis.dashboard.training import render_training_section

__all__ = [
    "render_backtest_section",
    "render_data_section",
    "render_model_section",
    "render_reports_section",
    "render_training_section",
    "render_metric_card",
    "render_zoned_metric",
    "render_chart",
    "render_config_summary",
    "render_trade_direction_summary",
    "date_only",
    "trim_generated_visual_sections",
    "find_sessions",
    "load_config",
    "parse_session_meta",
    "session_selector_fragment",
    "main",
]
