"""Chart renderer + config/trade helpers used across sections."""

from __future__ import annotations

import re

import streamlit as st


def render_chart(chart: object, height: str = "500px") -> None:
    """Render pyecharts chart via streamlit-echarts."""
    try:
        from streamlit_echarts import st_pyecharts

        st_pyecharts(chart, height=height)
    except ImportError as e:
        st.warning(f"Chart render failed: {e}")


def date_only(value: str) -> str:
    """Return date part of ISO timestamp string."""
    return str(value).split()[0]


def trim_generated_visual_sections(content: str) -> str:
    """Strip 'Visual Evidence' section — duplicated by dashboard charts."""
    m = re.compile(r"^##\s+\d*\.?\s*Visual Evidence", re.MULTILINE).search(content)
    return content[: m.start()].rstrip() if m else content


def render_config_summary(config: object) -> None:
    """Sidebar: current experiment config."""
    data = getattr(config, "data", None)
    val = getattr(config, "validation", None)
    model = getattr(config, "model", None)
    labels = getattr(config, "labels", None)
    bt = getattr(config, "backtest", None)
    wf = getattr(config, "workflow", None)

    if data:
        # data_range is a top-level Config attribute, not nested under data
        dr = getattr(config, "data_range", None)
        start = date_only(dr.start) if dr else "?"
        end = date_only(dr.end) if dr else "?"
        st.markdown(f"**Data**: {data.symbol} {data.timeframe}  {start}→{end}")
    if val:
        st.markdown(
            f"**Walk-forward**: {val.method}, "
            f"train={val.train_window_bars:,}, "
            f"test={val.test_window_bars:,}, "
            f"purge={val.purge_bars} bars"
        )
    if model:
        st.markdown(
            f"**Model**: {model.architecture}, "
            f"objective={model.objective}, leaves={model.num_leaves}, "
            f"lr={model.learning_rate}"
        )
    if labels:
        st.markdown(
            f"**Labels**: horizon={labels.horizon_bars}, "
            f"TP={labels.atr_tp_multiplier}xATR, "
            f"SL={labels.atr_sl_multiplier}xATR"
        )
    if bt:
        st.markdown(
            f"**Backtest**: capital=${bt.initial_capital:,.0f}, "
            f"spread={bt.spread_ticks:g} ticks, "
            f"conf>={bt.confidence_threshold:.2f}"
        )
    if wf:
        flags = []
        for key in ("run_data", "run_dataset", "run_models", "run_reporting"):
            if getattr(wf, key, False):
                flags.append(key.replace("run_", ""))
        st.markdown(f"**Pipeline stages**: {', '.join(flags) if flags else 'none'}")
