"""Training: LightGBM config summary + pipeline log viewer."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from thesis.dashboard.cards import render_metric_card


def render_training_section(data: dict, session_dir: str) -> None:
    """Training history metrics + full pipeline log."""
    st.markdown("> 🏠 Dashboard > **Training**")
    st.header("Training History")

    session_path = Path(session_dir)

    # LightGBM training summary
    history_path = session_path / "models" / "training_history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)

        lgbm = history.get("lgbm") or history.get("model") or {}
        if lgbm:
            with st.container(border=True):
                st.subheader("LightGBM Configuration")
                st.caption("Gradient boosting model training parameters and results")
                cols = st.columns(3, gap="small")
                render_metric_card(
                    cols[0],
                    "Best Iteration",
                    str(lgbm.get("best_iteration", "N/A")),
                    "Optimal boosting round",
                    "#22c55e",
                )
                render_metric_card(
                    cols[1],
                    "Features",
                    str(lgbm.get("n_features", "N/A")),
                    "Input feature count",
                    "#3b82f6",
                )
                render_metric_card(
                    cols[2],
                    "Classes",
                    str(lgbm.get("n_classes", "N/A")),
                    "Target labels",
                    "#8b5cf6",
                )
    else:
        st.info("No training history file found for this session.")

    st.divider()

    # Pipeline log viewer
    log_path = session_path / "logs" / "pipeline.log"
    if log_path.exists():
        st.subheader("Pipeline Log")
        with open(log_path) as f:
            lines = f.readlines()
        with st.expander("Recent Log (last 150 lines)", expanded=True):
            st.code("".join(lines[-150:]), language="log")
        with st.expander("Full Pipeline Log", expanded=False):
            st.code("".join(lines), language="log")
    else:
        st.info("No pipeline log found for this session.")
