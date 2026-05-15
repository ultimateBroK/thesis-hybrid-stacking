"""Model Performance: accuracy, confusion matrix, confidence, feature importance."""

from __future__ import annotations

import streamlit as st

from thesis.charts import (
    build_confidence_distribution_chart,
    build_confusion_matrix_chart,
    build_feature_importance_chart,
    build_prediction_distribution_chart,
)
from thesis.dashboard.cards import render_metric_card
from thesis.dashboard.shared import render_chart


def render_model_section(data: dict, session_dir: str) -> None:
    """Classification metrics, confusion matrix, confidence, feature importance."""
    st.markdown("> 🏠 Dashboard > **Model Performance**")
    st.header("Model Performance")

    preds = data.get("predictions")
    fi = data.get("feature_importance", {})

    if not preds or len(preds) == 0:
        st.info("No predictions data available.")
        return

    required = {"true_label", "pred_label"}
    if not required.issubset(set(preds.columns)):
        st.warning(f"Predictions missing columns: {required - set(preds.columns)}")
        return

    y_true = preds["true_label"].to_numpy()
    y_pred = preds["pred_label"].to_numpy()
    total = len(y_true)

    # Exact match accuracy
    exact_acc = float((y_true == y_pred).mean())

    # Directional accuracy: non-hold bars only
    non_hold_mask = (y_true != 0) & (y_pred != 0)
    dir_acc = (
        float((y_true[non_hold_mask] == y_pred[non_hold_mask]).mean())
        if non_hold_mask.sum() > 0
        else 0.0
    )
    dir_baseline = 0.5

    # Per-class: recall, precision, F1
    per_class: dict[str, dict[str, float | int]] = {}
    for cls, name in [(-1, "Short"), (0, "Hold"), (1, "Long")]:
        true_mask = y_true == cls
        pred_mask = y_pred == cls
        recall = (
            float((y_pred[true_mask] == cls).mean()) if true_mask.sum() > 0 else 0.0
        )
        precision = (
            float((y_true[pred_mask] == cls).mean()) if pred_mask.sum() > 0 else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        per_class[name] = {
            "true_count": int(true_mask.sum()),
            "pred_count": int(pred_mask.sum()),
            "recall": recall,
            "precision": precision,
            "f1": f1,
        }

    # Accuracy summary cards
    with st.container(border=True):
        st.subheader("Accuracy Metrics")
        st.caption("Model prediction accuracy against test set labels")
        acc_cols = st.columns(4, gap="small")
        render_metric_card(
            acc_cols[0],
            "Directional Accuracy",
            f"{dir_acc:.1%}",
            f"+{(dir_acc - dir_baseline) * 100:.1f}pp vs random",
            "#3b82f6",
        )
        render_metric_card(
            acc_cols[1],
            "Exact-Match Accuracy",
            f"{exact_acc:.1%}",
            None,
            "#8b5cf6",
        )
        render_metric_card(
            acc_cols[2],
            "Directional Baseline",
            f"{dir_baseline:.1%}",
            "Random guess baseline",
            "#6b7280",
        )
        render_metric_card(
            acc_cols[3],
            "Test Samples",
            f"{total:,}",
            None,
            "#10b981",
        )

    # Per-class recall/precision/F1
    st.subheader("Per-Class Performance")
    cls_cols = st.columns(3)
    for idx, (name, m) in enumerate(per_class.items()):
        with cls_cols[idx]:
            st.markdown(f"**{name}**")
            st.caption(f"True: {m['true_count']:,} | Predicted: {m['pred_count']:,}")
            st.progress(m["recall"], text=f"Recall: {m['recall']:.1%}")
            st.progress(m["precision"], text=f"Precision: {m['precision']:.1%}")
            st.progress(m["f1"], text=f"F1: {m['f1']:.2f}")

    st.divider()

    # Charts
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Confusion Matrix")
        render_chart(build_confusion_matrix_chart(y_true, y_pred), height="500px")
    with c2:
        st.subheader("Confidence Distribution")
        render_chart(build_confidence_distribution_chart(preds), height="500px")

    st.subheader("Prediction Distribution")
    render_chart(build_prediction_distribution_chart(y_true, y_pred), height="400px")

    # Feature importance
    if fi:
        st.subheader("Feature Importance (Hybrid)")
        render_chart(build_feature_importance_chart(fi), height="600px")
    else:
        st.info("No feature importance data available.")
