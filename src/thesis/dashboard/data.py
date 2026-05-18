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


def filter_ohlcv_by_selected_date_range(ohlcv: pl.DataFrame) -> pl.DataFrame:
    """Streamlit date-input filter applied to OHLCV timestamp column."""
    ts_col = ohlcv["timestamp"]
    ts_parsed = (
        ts_col.str.to_datetime()
        if ts_col.dtype == pl.Utf8
        else ts_col.cast(pl.Datetime)
    )

    min_dt, max_dt = ts_parsed.min(), ts_parsed.max()
    if min_dt is None or max_dt is None:
        return ohlcv

    min_date, max_date = min_dt.date(), max_dt.date()
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
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            key="_candle_end",
        )

    start_str = str(start_date)
    end_str = str(end_date) + " 23:59:59"
    return ohlcv.filter(
        (ts_parsed >= pl.lit(start_str).str.to_datetime())
        & (ts_parsed <= pl.lit(end_str).str.to_datetime())
    )


def render_filtered_candlestick(
    ohlcv_filtered: pl.DataFrame,
    config: object,
) -> None:
    """Build and render candlestick chart for filtered OHLCV."""
    if ohlcv_filtered.is_empty():
        st.info("No data in selected date range.")
        return

    chart, info = build_candlestick_chart(ohlcv_filtered, config)
    render_chart(chart, height="700px")

    if info["downsampled"]:
        st.caption(
            f"Showing {info['displayed_bars']:,} of {info['total_bars']:,} bars"
            " (downsampled). DataZoom to navigate."
        )


def show_dataset_breadcrumb() -> None:
    """Breadcrumb + header for Dataset section."""
    st.markdown("> Home > **Dataset**")
    st.header("Dataset")


def render_dataset_summary(data: dict) -> None:
    """Top metrics: bars, date range, features, labels."""
    ohlcv = data.get("ohlcv")
    features = data.get("features")
    labels = data.get("labels")

    if ohlcv is None:
        return

    feature_cols = (
        [c for c in features.columns if c not in EXCLUDED_FEATURE_COLS]
        if features is not None
        else []
    )
    label_count = len(labels) if labels is not None else 0

    cols = st.columns(4, gap="small")
    cols[0].metric("Total bars", f"{len(ohlcv):,}")
    start = ohlcv["timestamp"].cast(pl.Utf8).min()
    end = ohlcv["timestamp"].cast(pl.Utf8).max()
    cols[1].metric("Date range", f"{start} -> {end}")
    cols[2].metric("Features", f"{len(feature_cols):,}")
    cols[3].metric("Labels", f"{label_count:,}")


def render_candlestick_panel(data: dict, config: object) -> None:
    """Date filter + candlestick chart."""
    ohlcv = data.get("ohlcv")
    if ohlcv is None or ohlcv.is_empty():
        st.info("No OHLCV data available.")
        return

    st.subheader("Dataset Overview")
    st.caption("Candlestick sample. Use date filter to inspect raw OHLCV cleanliness.")

    filtered = filter_ohlcv_by_selected_date_range(ohlcv)
    render_filtered_candlestick(filtered, None)


def render_label_panel(data: dict) -> None:
    """Label distribution chart."""
    st.subheader("Label Construction")
    labels = data.get("labels")
    if labels is not None and "label" in labels.columns:
        render_chart(build_label_distribution_chart(labels), height="500px")
    else:
        st.info("No labels data.")


def render_feature_panel(data: dict) -> None:
    """Correlation heatmap + per-feature histogram."""
    features = data.get("features")
    if features is None:
        st.info("No features data available.")
        return

    feature_cols = [c for c in features.columns if c not in EXCLUDED_FEATURE_COLS]

    st.subheader("Feature Engineering")
    render_chart(build_correlation_heatmap(features), height="600px")

    st.subheader("Feature Distributions")
    st.caption(f"{len(feature_cols)} model-facing columns.")
    if feature_cols:
        selected = st.selectbox(
            "Feature", feature_cols, key="_feature_distribution_select"
        )
        render_chart(
            build_feature_distribution_chart(features, selected), height="450px"
        )


def render_data_section(data: dict, config: object, session_dir: str) -> None:
    """OHLCV candlestick + feature/label charts with date range filter."""
    show_dataset_breadcrumb()
    render_dataset_summary(data)
    render_candlestick_panel(data, config)
    render_label_panel(data)
    render_feature_panel(data)
