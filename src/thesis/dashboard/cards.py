"""Metric card rendering for the Streamlit dashboard.

Provides HTML-based metric cards with zone-colored gradient backgrounds
and recommendation badges. Depends on zones.py for classification.
"""

import html

from thesis.dashboard.zones import _ZONE_COLORS, _get_metric_zone, _is_extreme_value


def _render_zoned_metric(
    col: object,
    label: str,
    value: float,
    metric_key: str,
    format_str: str = "{:.2f}",
    unit: str = "",
) -> None:
    """Render a metric card with color-coded zone indicator.

    Args:
        col: Streamlit column container to render into.
        label: Human-readable metric name.
        value: Numeric metric value.
        metric_key: Key for zone lookup (e.g., 'sharpe_ratio').
        format_str: Python format string for the value.
        unit: Suffix unit label (e.g., '%', '$').
    """
    is_extreme, _ = _is_extreme_value(metric_key, value)
    color, zone_label, recommendation = _get_metric_zone(metric_key, value)

    hex_color = _ZONE_COLORS.get(color, "#6b7280")
    display_suffix = " ⚠️" if is_extreme else ""
    safe_label = html.escape(label)
    safe_value = html.escape(format_str.format(value))
    safe_unit = html.escape(unit)
    safe_zone = html.escape(zone_label)
    safe_rec = html.escape(recommendation)

    col.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {hex_color}22 0%, {hex_color}11 100%);
            border-left: 3px solid {hex_color};
            border-radius: 8px;
            padding: 12px 14px;
            margin: 4px 0;
            min-height: 110px;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            box-sizing: border-box;
        ">
            <div>
                <div style="font-size: 0.7rem; color: inherit; opacity: 0.7; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">{safe_label}</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: inherit; line-height: 1.2;">
                    {safe_value}{safe_unit}{display_suffix}
                </div>
            </div>
            <div style="margin-top: 8px;">
                <span style="
                    background: {hex_color}33;
                    color: {hex_color};
                    padding: 2px 10px;
                    border-radius: 12px;
                    font-size: 0.65rem;
                    font-weight: 700;
                    text-transform: uppercase;
                    letter-spacing: 0.03em;
                ">{safe_zone}</span>
                <div style="font-size: 0.65rem; color: inherit; opacity: 0.6; margin-top: 4px; line-height: 1.3;">{safe_rec}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_metric_card(
    col: object,
    label: str,
    value: str,
    caption: str | None,
    color: str,
) -> None:
    """Render a styled metric card with gradient background and accent border.

    Args:
        col: Streamlit column container to render into.
        label: Human-readable metric name.
        value: Formatted value string to display.
        caption: Optional caption text below the value.
        color: CSS color string for the gradient accent.
    """
    safe_label = html.escape(label)
    safe_value = html.escape(value)
    caption_html = (
        f'<div style="font-size: 0.65rem; color: inherit; opacity: 0.6; margin-top: 4px; line-height: 1.3;">{html.escape(caption)}</div>'
        if caption
        else ""
    )
    col.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {color}22 0%, {color}11 100%);
            border-left: 3px solid {color};
            border-radius: 8px;
            padding: 12px 14px;
            margin: 4px 0;
            min-height: 90px;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            box-sizing: border-box;
        ">
            <div>
                <div style="font-size: 0.7rem; color: inherit; opacity: 0.7; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">{safe_label}</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: inherit; line-height: 1.2;">{safe_value}</div>
            </div>
            {caption_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
