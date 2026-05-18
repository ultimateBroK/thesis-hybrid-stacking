"""Dataset section: data, labels, and feature overview."""

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
    st.markdown("> Home > **Dataset**")
    st.header("Dataset")

    ohlcv = data.get("ohlcv")
    features = data.get("features")
    labels = data.get("labels")

    label_count = len(labels) if labels is not None else 0
    feature_cols = (
        [c for c in features.columns if c not in EXCLUDED_FEATURE_COLS]
        if features is not None
        else []
    )
    if ohlcv is not None:
        cols = st.columns(4, gap="small")
        cols[0].metric("Total bars", f"{len(ohlcv):,}")
        start = ohlcv["timestamp"].cast(pl.Utf8).min()
        end = ohlcv["timestamp"].cast(pl.Utf8).max()
        cols[1].metric("Date range", f"{start} -> {end}")
        cols[2].metric("Features", f"{len(feature_cols):,}")
        cols[3].metric("Labels", f"{label_count:,}")

    if ohlcv is not None and not ohlcv.is_empty():
        st.subheader("Dataset Overview")
        st.caption(
            "Candlestick sample. Use date filter to inspect raw OHLCV cleanliness."
        )

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

    st.subheader("Label Construction")
    if labels is not None and "label" in labels.columns:
        render_chart(build_label_distribution_chart(labels), height="500px")
    else:
        st.info("No labels data.")

    st.subheader("Feature Engineering")
    if features is not None:
        render_chart(build_correlation_heatmap(features), height="600px")

        # Per-feature histogram
        st.subheader("Feature Distributions")
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
