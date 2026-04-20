"""Data Exploration section for the Streamlit dashboard.

Renders OHLCV candlestick charts, feature correlations, label distributions,
and per-feature distribution histograms.
"""

from datetime import timedelta

import numpy as np
import polars as pl
import streamlit as st
from pyecharts import options as opts
from pyecharts.charts import Bar

from thesis.charts import (
    COLORS,
    EXCLUDED_FEATURE_COLS,
    build_candlestick_chart,
    build_correlation_heatmap,
    build_label_distribution_chart,
)


def _render_chart(chart: object, height: str = "500px") -> None:
    """Render a pyecharts chart into the Streamlit app.

    Args:
        chart: PyECharts chart object to render.
        height: CSS height string for the chart container.
    """
    try:
        from streamlit_echarts import st_pyecharts

        st_pyecharts(chart, height=height)
    except Exception as e:
        st.warning(f"Chart render failed: {e}")


def _render_data_section(data: dict, config: object) -> None:
    """Render the Data Exploration section with charts and controls.

    Produces charts for OHLCV candlestick data, feature correlations,
    label distributions, and per-feature distribution histograms.

    Args:
        data: Session data with optional keys 'ohlcv', 'features', 'labels'.
        config: Session configuration object passed to chart builders.
    """
    st.markdown("> 🏠 Dashboard > **Data Exploration**")
    st.header("Data Exploration")

    ohlcv = data.get("ohlcv")
    if ohlcv is not None:
        st.caption(
            f"{len(ohlcv):,} bars | "
            f"{ohlcv['timestamp'].cast(pl.Utf8).min()} → {ohlcv['timestamp'].cast(pl.Utf8).max()}"
        )
    features = data.get("features")
    labels = data.get("labels")

    if ohlcv is not None and len(ohlcv) > 0:
        st.subheader("Candlestick Chart")

        ts_col = ohlcv["timestamp"]
        if ts_col.dtype == pl.Utf8:
            ts_parsed = ts_col.str.to_datetime()
        else:
            ts_parsed = ts_col.cast(pl.Datetime)

        min_dt = ts_parsed.min()
        max_dt = ts_parsed.max()
        total_bars = len(ohlcv)

        if min_dt is not None and max_dt is not None:
            min_date = min_dt.date()
            max_date = max_dt.date()
            default_end = max_date
            default_start = max(min_date, max_date - timedelta(days=180))

            col_range1, col_range2 = st.columns(2)
            with col_range1:
                start_date = st.date_input(
                    "From",
                    value=default_start,
                    min_value=min_date,
                    max_value=max_date,
                    key="_candle_start",
                )
            with col_range2:
                end_date = st.date_input(
                    "To",
                    value=default_end,
                    min_value=min_date,
                    max_value=max_date,
                    key="_candle_end",
                )

            start_str = str(start_date)
            end_str = str(end_date) + " 23:59:59"
            ohlcv_filtered = ohlcv.filter(
                (ts_parsed >= pl.lit(start_str).str.to_datetime())
                & (ts_parsed <= pl.lit(end_str).str.to_datetime())
            )
        else:
            ohlcv_filtered = ohlcv

        if len(ohlcv_filtered) > 0:
            chart, info = build_candlestick_chart(ohlcv_filtered, config)
            _render_chart(chart, height="700px")
            if info["total_bars"] < total_bars:
                st.caption(
                    f"Showing {info['displayed_bars']:,} of {total_bars:,} total bars "
                    f"({len(ohlcv_filtered):,} in selected range)"
                )
            elif info["downsampled"]:
                st.caption(
                    f"Showing {info['displayed_bars']:,} of {info['total_bars']:,} bars "
                    f"(downsampled). Use DataZoom to navigate."
                )
        else:
            st.info("No data in selected date range.")
    else:
        st.info("No OHLCV data available.")

    if features is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Feature Correlation")
            chart = build_correlation_heatmap(features)
            _render_chart(chart, height="600px")
        with col2:
            st.subheader("Label Distribution")
            if labels is not None and "label" in labels.columns:
                chart = build_label_distribution_chart(labels)
                _render_chart(chart, height="500px")
            else:
                st.info("No labels data.")

        st.subheader("Feature Distributions")
        feature_cols = [c for c in features.columns if c not in EXCLUDED_FEATURE_COLS]
        if feature_cols:
            tabs = st.tabs(feature_cols)
            for col, tab in zip(feature_cols, tabs):
                with tab:
                    vals = features[col].drop_nulls().to_numpy()
                    if len(vals) > 0:
                        counts, bin_edges = np.histogram(vals, bins=50)
                        bin_centers = [
                            (bin_edges[i] + bin_edges[i + 1]) / 2
                            for i in range(len(counts))
                        ]
                        x_labels = [f"{v:.2f}" for v in bin_centers]
                        bar = (
                            Bar(init_opts=opts.InitOpts(height="400px"))
                            .add_xaxis(x_labels)
                            .add_yaxis(
                                series_name=col,
                                y_axis=counts.tolist(),
                                label_opts=opts.LabelOpts(is_show=False),
                                itemstyle_opts=opts.ItemStyleOpts(
                                    color=COLORS["primary"]
                                ),
                            )
                            .set_global_opts(
                                title_opts=opts.TitleOpts(title=f"Distribution: {col}"),
                                xaxis_opts=opts.AxisOpts(name=col),
                                yaxis_opts=opts.AxisOpts(name="Count"),
                                tooltip_opts=opts.TooltipOpts(trigger="axis"),
                                datazoom_opts=[opts.DataZoomOpts(type_="inside")],
                            )
                        )
                        _render_chart(bar, height="400px")
                    else:
                        st.info(f"No data for {col}")
    else:
        st.info("No features data available.")
