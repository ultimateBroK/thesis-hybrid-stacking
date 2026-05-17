"""Reports: thesis markdown, equity image, walk-forward history, feature importance."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from thesis.charts import build_feature_importance_chart
from thesis.dashboard.backtest import render_backtest_section
from thesis.dashboard.cards import render_metric_card
from thesis.dashboard.shared import render_chart, trim_generated_visual_sections


def render_reports_section(data: dict, config: object, session_dir: str) -> None:
    """Markdown report + equity image + walk-forward history + feature importance."""
    st.markdown("> 🏠 Dashboard > **Reports**")

    # Sub-tabs: Reports content + Application Demo
    report_tab, demo_tab = st.tabs(["📝 Report", "🎯 Application Demo"])

    with report_tab:
        _render_report_content(data, session_dir)

    with demo_tab:
        render_backtest_section(data, config, session_dir)


def _render_report_content(data: dict, session_dir: str) -> None:
    """Markdown report + equity image + walk-forward history + feature importance."""
    st.markdown("> 🏠 Dashboard > **Reports**")

    reports_dir = Path(session_dir) / "reports"

    # Thesis markdown report (strip duplicated visual sections)
    report_md = reports_dir / "thesis_report.md"
    if report_md.exists():
        st.markdown(trim_generated_visual_sections(report_md.read_text()))
    else:
        st.info("No thesis report available.")

    st.divider()

    # Equity curve image
    equity_png = reports_dir / "equity_curve.png"
    if equity_png.exists():
        st.subheader("Equity Curve")
        st.image(str(equity_png), width="stretch")

    st.divider()

    # Walk-forward summary
    wf_path = reports_dir / "walk_forward_history.json"
    if wf_path.exists():
        with open(wf_path) as f:
            wf = json.load(f)

        with st.container(border=True):
            st.subheader("Walk-Forward History")
            st.caption("Sliding-window validation summary")
            cols = st.columns(3, gap="small")
            render_metric_card(
                cols[0],
                "Windows",
                str(wf.get("num_windows", "?")),
                "Total walk-forward windows",
                "#3b82f6",
            )
            render_metric_card(
                cols[1],
                "OOF Predictions",
                f"{wf.get('total_oof_predictions', 0):,}",
                "Out-of-fold prediction count",
                "#10b981",
            )
            render_metric_card(
                cols[2],
                "Architecture",
                str(wf.get("architecture", "hybrid")),
                "Model architecture used",
                "#8b5cf6",
            )

            details = wf.get("window_details", [])
            if details:
                with st.expander("Window Details", expanded=False):
                    st.dataframe(
                        [
                            {
                                "Window": d["window"],
                                "Train Start": d["train_start_idx"],
                                "Train End": d["train_end_idx"],
                                "Test Start": d["test_start_idx"],
                                "Test End": d["test_end_idx"],
                            }
                            for d in details
                        ],
                        width="stretch",
                        hide_index=True,
                    )

    # Feature importance
    fi_path = reports_dir / "feature_importance.json"
    if fi_path.exists():
        with open(fi_path) as f:
            fi = json.load(f)
        if fi:
            st.divider()
            st.subheader("Feature Importance (Hybrid)")
            render_chart(build_feature_importance_chart(fi), height="600px")
