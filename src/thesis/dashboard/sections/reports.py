"""Reports section for the Streamlit dashboard."""

import json
from pathlib import Path

import streamlit as st
from streamlit_echarts import st_pyecharts

from thesis.charts import build_shap_chart


def _render_chart(chart: object, height: str = "500px") -> None:
    try:
        st_pyecharts(chart, height=height)
    except Exception as e:
        st.warning(f"Chart render failed: {e}")


def _render_reports_section(session_dir: str) -> None:
    """Render the reports page with thesis markdown and artifact visuals.

    Args:
        session_dir: Session directory path containing generated report artifacts.
    """
    st.markdown("> 🏠 Dashboard > **Reports**")

    session_path = Path(session_dir)
    reports_dir = session_path / "reports"

    report_md_path = reports_dir / "thesis_report.md"
    if report_md_path.exists():
        content = report_md_path.read_text()
        section_10_marker = "## 10. Visual Evidence & Analytics"
        if section_10_marker in content:
            content = content.split(section_10_marker)[0]
        st.markdown(content, unsafe_allow_html=True)
    else:
        st.info("No thesis report available.")

    st.divider()

    equity_png = reports_dir / "equity_curve.png"
    if equity_png.exists():
        st.subheader("Equity Curve")
        st.image(str(equity_png), width="stretch")

    shap_json_path = reports_dir / "shap_values.json"
    if shap_json_path.exists():
        with open(shap_json_path) as f:
            shap_data = json.load(f)
        st.subheader("SHAP Feature Importance")
        chart = build_shap_chart(shap_data)
        _render_chart(chart, height="600px")
    else:
        shap_png = reports_dir / "shap_summary.png"
        if shap_png.exists():
            st.subheader("SHAP Feature Importance")
            st.image(str(shap_png), width="stretch")

    st.divider()

    bt_charts_dir = reports_dir / "charts" / "backtest"
    if bt_charts_dir.exists():
        st.subheader("Backtest Charts")
        cols = st.columns(2)
        chart_files = [
            ("equity_drawdown.png", "Equity & Drawdown"),
            ("monthly_returns.png", "Monthly Returns"),
            ("pnl_histogram.png", "PnL Distribution"),
            ("duration_vs_pnl.png", "Duration vs PnL"),
        ]
        for idx, (fname, title) in enumerate(chart_files):
            with cols[idx % 2]:
                fpath = bt_charts_dir / fname
                if fpath.exists():
                    st.markdown(f"**{title}**")
                    st.image(str(fpath), width="stretch")

    st.divider()

    model_charts_dir = reports_dir / "charts" / "model"
    if model_charts_dir.exists():
        st.subheader("Model Charts")
        cols = st.columns(2)
        chart_files = [
            ("confusion_matrix.png", "Confusion Matrix"),
            ("confidence_distribution.png", "Confidence Distribution"),
            ("feature_importance.png", "LightGBM Feature Importance"),
        ]
        for idx, (fname, title) in enumerate(chart_files):
            with cols[idx % 2]:
                fpath = model_charts_dir / fname
                if fpath.exists():
                    st.markdown(f"**{title}**")
                    st.image(str(fpath), width="stretch")

    st.divider()

    data_charts_dir = reports_dir / "charts" / "data"
    if data_charts_dir.exists():
        st.subheader("Data Charts")
        cols = st.columns(2)
        chart_files = [
            ("candlestick.png", "Candlestick Chart"),
            ("feature_correlation.png", "Feature Correlation"),
            ("label_distribution.png", "Label Distribution"),
            ("feature_distributions.png", "Feature Distributions"),
        ]
        for idx, (fname, title) in enumerate(chart_files):
            with cols[idx % 2]:
                fpath = data_charts_dir / fname
                if fpath.exists():
                    st.markdown(f"**{title}**")
                    st.image(str(fpath), width="stretch")

    bt_html = session_path / "backtest" / "backtest_chart.html"
    if bt_html.exists():
        st.divider()
        st.subheader("Interactive Backtest Chart")
        with open(bt_html) as f:
            html_content = f.read()
        st.iframe(html_content, height=1000)
