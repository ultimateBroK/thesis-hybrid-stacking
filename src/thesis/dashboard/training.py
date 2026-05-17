"""Training: architecture summary, walk-forward per-window accuracy, pipeline log."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import streamlit as st

from thesis.dashboard.cards import render_metric_card


def render_training_section(data: dict, config: object, session_dir: str) -> None:
    """Training history metrics + per-window accuracy + full pipeline log."""
    st.markdown("> 🏠 Dashboard > **Training**")
    st.header("Training History")

    session_path = Path(session_dir)

    # Architecture summary from training_history.json
    history_path = session_path / "models" / "training_history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)

        arch = history.get("architecture", "unknown")
        stacking = history.get("stacking", {})
        lgbm = history.get("lgbm") or history.get("model") or {}

        with st.container(border=True):
            st.subheader(f"Architecture: {arch.title()}")
            st.caption("Model training parameters and walk-forward results")

            if stacking:
                cols = st.columns(4, gap="small")
                render_metric_card(
                    cols[0],
                    "Features",
                    str(stacking.get("n_features", "N/A")),
                    "Input feature count",
                    "#3b82f6",
                )
                render_metric_card(
                    cols[1],
                    "Base Models",
                    str(len(stacking.get("base_models", []))),
                    ", ".join(stacking.get("base_models", [])),
                    "#22c55e",
                )
                render_metric_card(
                    cols[2],
                    "Meta Model",
                    str(stacking.get("meta_model", "N/A")),
                    "Stacking meta learner",
                    "#8b5cf6",
                )
                render_metric_card(
                    cols[3],
                    "Internal Folds",
                    str(
                        stacking.get("validation_protocol", {}).get(
                            "internal_folds", "N/A"
                        )
                    ),
                    "OOF fold count",
                    "#f59e0b",
                )
            elif lgbm:
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

        # Per-window accuracy chart
        per_window = history.get("per_window_accuracies", {})
        if per_window:
            st.divider()
            st.subheader("Walk-Forward Per-Window Accuracy")
            st.caption("OOF accuracy per walk-forward window (lower = harder period)")

            wf_df = pl.DataFrame(
                {
                    "Window": [int(k) for k in per_window],
                    "Accuracy": [float(v) * 100 for v in per_window.values()],
                }
            ).sort("Window")

            mean_acc = wf_df["Accuracy"].mean()
            min_acc = wf_df["Accuracy"].min()
            max_acc = wf_df["Accuracy"].max()

            stat_cols = st.columns(3, gap="small")
            render_metric_card(
                stat_cols[0],
                "Mean Accuracy",
                f"{mean_acc:.1f}%",
                f"Across {len(per_window)} windows",
                "#3b82f6",
            )
            render_metric_card(
                stat_cols[1],
                "Best Window",
                f"{max_acc:.1f}%",
                f"Window {wf_df.filter(pl.col('Accuracy') == max_acc)['Window'][0]}",
                "#22c55e",
            )
            render_metric_card(
                stat_cols[2],
                "Worst Window",
                f"{min_acc:.1f}%",
                f"Window {wf_df.filter(pl.col('Accuracy') == min_acc)['Window'][0]}",
                "#ef4444",
            )

            st.bar_chart(wf_df, x="Window", y="Accuracy", height=350)

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
