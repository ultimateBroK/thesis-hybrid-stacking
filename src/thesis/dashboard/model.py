"""Model and evaluation sections for the ML result viewer."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from thesis.charts import (
    build_confidence_distribution_chart,
    build_confusion_matrix_chart,
    build_feature_importance_chart,
    build_model_comparison_chart,
)
from thesis.dashboard.cards import render_metric_card
from thesis.dashboard.shared import render_chart
from thesis.dashboard.training import render_training_section
from thesis.visualization.summaries import compute_prediction_summary


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def render_model_section(data: dict, config: object, session_dir: str) -> None:
    """Architecture summary and feature importance."""
    st.markdown("> Home > **Model**")
    st.header("Model")
    st.caption("Hybrid Stacking classifier: base learners feed a meta learner.")

    history_path = Path(session_dir) / "models" / "training_history.json"
    history = _load_json(history_path)
    stacking = history.get("stacking", {}) if history else {}

    with st.container(border=True):
        cols = st.columns(4, gap="small")
        render_metric_card(
            cols[0],
            "Architecture",
            str(history.get("architecture", "Hybrid")).title() if history else "Hybrid",
            "Stacking classifier",
            "#3b82f6",
        )
        render_metric_card(
            cols[1],
            "Base Models",
            str(len(stacking.get("base_models", [])) or 3),
            ", ".join(stacking.get("base_models", [])) or "LR, RF, LightGBM",
            "#22c55e",
        )
        render_metric_card(
            cols[2],
            "Meta Model",
            str(stacking.get("meta_model", "Logistic Regression")),
            "Combines base probabilities",
            "#8b5cf6",
        )
        render_metric_card(
            cols[3],
            "Validation",
            "Walk-forward",
            "Chronological split; no random shuffle",
            "#f59e0b",
        )

    st.subheader("Stacking Flow")
    st.markdown(
        "Feature matrix -> Logistic Regression / Random Forest / LightGBM -> "
        "probability features -> meta Logistic Regression -> "
        "final Short/Hold/Long label."
    )

    fi = data.get("feature_importance", {})
    if fi:
        st.subheader("Feature Importance")
        render_chart(build_feature_importance_chart(fi), height="600px")
    else:
        st.info("No feature importance data available.")


def render_evaluation_section(data: dict, config: object, session_dir: str) -> None:
    """ML evaluation metrics and charts."""
    st.markdown("> Home > **Evaluation**")
    st.header("Evaluation")

    summary = compute_prediction_summary(data)
    if not summary:
        st.info("No predictions data available.")
        return

    with st.container(border=True):
        st.subheader("Classification Summary")
        cols = st.columns(4, gap="small")
        render_metric_card(
            cols[0], "Accuracy", f"{summary['accuracy']:.1%}", None, "#3b82f6"
        )
        render_metric_card(
            cols[1], "Macro F1", f"{summary['macro_f1']:.3f}", None, "#8b5cf6"
        )
        render_metric_card(
            cols[2],
            "Directional Acc.",
            f"{summary['directional_accuracy']:.1%}",
            "Excludes Hold predictions",
            "#22c55e",
        )
        render_metric_card(
            cols[3],
            "Best Base Model",
            str(summary["best_base_model"]),
            f"{summary['total_predictions']:,} predictions",
            "#f59e0b",
        )

    rows = data.get("model_comparison", [])
    if rows:
        st.subheader("Model Comparison")
        render_chart(build_model_comparison_chart(rows), height="430px")

    st.subheader("Confusion Matrix")
    render_chart(
        build_confusion_matrix_chart(summary["y_true"], summary["y_pred"]),
        height="500px",
    )

    st.subheader("Per-Class Performance")
    cls_cols = st.columns(3)
    for idx, (name, m) in enumerate(summary["per_class"].items()):
        with cls_cols[idx]:
            st.markdown(f"**{name}**")
            st.caption(f"True: {m['true_count']:,} | Predicted: {m['pred_count']:,}")
            st.progress(float(m["recall"]), text=f"Recall: {m['recall']:.1%}")
            st.progress(float(m["precision"]), text=f"Precision: {m['precision']:.1%}")
            st.progress(float(m["f1"]), text=f"F1: {m['f1']:.2f}")

    with st.expander("Training details / logs", expanded=False):
        render_training_section(data, config, session_dir)

    with st.expander("Confidence distribution (secondary)", expanded=False):
        st.caption(
            "Hidden by default because probability calibration is not the thesis focus."
        )
        render_chart(
            build_confidence_distribution_chart(data["predictions"]), height="420px"
        )


__all__ = [
    "compute_prediction_summary",
    "render_evaluation_section",
    "render_model_section",
]
