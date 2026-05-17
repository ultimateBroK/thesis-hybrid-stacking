"""Data Exploration: candlestick, feature correlations, label distributions."""

from __future__ import annotations

from datetime import timedelta

import polars as pl
import streamlit as st

from thesis.charts import (
    EXCLUDED_FEATURE_COLS,
    build_candlestick_chart,
    build_correlation_heatmap,
    build_feature_distribution_chart,
    build_label_distribution_chart,
)
from thesis.dashboard.shared import render_chart


def render_data_section(data: dict, config: object, session_dir: str) -> None:
    """OHLCV candlestick + feature/label charts with date range filter."""
    st.markdown("> 🏠 Dashboard > **Data Exploration**")
    st.header("Data Exploration")

    ohlcv = data.get("ohlcv")
    features = data.get("features")
    labels = data.get("labels")

    # OHLCV summary
    if ohlcv is not None:
        st.caption(
            f"{len(ohlcv):,} bars | "
            f"{ohlcv['timestamp'].cast(pl.Utf8).min()} "
            f"→ {ohlcv['timestamp'].cast(pl.Utf8).max()}"
        )

    if ohlcv is not None and not ohlcv.is_empty():
        st.subheader("Candlestick Chart")

        ts_col = ohlcv["timestamp"]
        ts_parsed = (
            ts_col.str.to_datetime()
            if ts_col.dtype == pl.Utf8
            else ts_col.cast(pl.Datetime)
        )

        min_dt, max_dt = ts_parsed.min(), ts_parsed.max()

        # Date range filter (default: last 180 days)
        if min_dt is not None and max_dt is not None:
            min_date, max_date = min_dt.date(), max_dt.date()
            default_end = max_date
            default_start = max(min_date, max_date - timedelta(days=180))

            c1, c2 = st.columns(2)
            with c1:
                start_date = st.date_input(
                    "From",
                    value=default_start,
                    min_value=min_date,
                    max_value=max_date,
                    key="_candle_start",
                )
            with c2:
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

        if not ohlcv_filtered.is_empty():
            chart, info = build_candlestick_chart(ohlcv_filtered, None)
            render_chart(chart, height="700px")

            if info["downsampled"]:
                st.caption(
                    f"Showing {info['displayed_bars']:,} of {info['total_bars']:,} bars"
                    " (downsampled). DataZoom to navigate."
                )
        else:
            st.info("No data in selected date range.")
    else:
        st.info("No OHLCV data available.")

    # Feature + Label charts
    if features is not None:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Feature Correlation")
            render_chart(build_correlation_heatmap(features), height="600px")
        with c2:
            st.subheader("Label Distribution")
            if labels is not None and "label" in labels.columns:
                render_chart(build_label_distribution_chart(labels), height="500px")
            else:
                st.info("No labels data.")

        # Per-feature histogram
        st.subheader("Feature Distributions")
        feature_cols = [c for c in features.columns if c not in EXCLUDED_FEATURE_COLS]
        st.caption(f"{len(feature_cols)} model-facing columns.")
        if feature_cols:
            selected = st.selectbox(
                "Feature", feature_cols, key="_feature_distribution_select"
            )
            render_chart(
                build_feature_distribution_chart(features, selected), height="450px"
            )
    else:
        st.info("No features data available.")
