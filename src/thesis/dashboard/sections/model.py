"""Model Performance section for the Streamlit dashboard."""

from pathlib import Path

import streamlit as st
from pyecharts import options as opts
from pyecharts.charts import Bar
from streamlit_echarts import st_pyecharts

from thesis.charts import (
    COLORS,
    build_confidence_distribution_chart,
    build_confusion_matrix_chart,
    build_feature_importance_chart,
    build_shap_chart,
)
from thesis.dashboard.cards import _render_metric_card


def _render_chart(chart: object, height: str = "500px") -> None:
    try:
        st_pyecharts(chart, height=height)
    except Exception as e:
        st.warning(f"Chart render failed: {e}")


def _render_model_section(data: dict, session_dir: str = "") -> None:
    """Render model performance metrics and model-analysis charts.

    Args:
        data: Session data mapping containing predictions, feature importance,
            and optional SHAP payload.
        session_dir: Session directory used to locate fallback SHAP PNG.
    """
    st.markdown("> 🏠 Dashboard > **Model Performance**")
    st.header("Model Performance")

    preds = data.get("predictions")
    fi = data.get("feature_importance", {})

    if preds is not None and len(preds) > 0:
        required_cols = {"true_label", "pred_label"}
        if not required_cols.issubset(set(preds.columns)):
            st.warning(
                f"Predictions missing columns: {required_cols - set(preds.columns)}"
            )
            return

        y_true = preds["true_label"].to_numpy()
        y_pred = preds["pred_label"].to_numpy()
        total = len(y_true)

        exact_acc = float((y_true == y_pred).mean())

        non_hold_mask = (y_true != 0) & (y_pred != 0)
        if non_hold_mask.sum() > 0:
            dir_correct = y_true[non_hold_mask] == y_pred[non_hold_mask]
            dir_acc = float(dir_correct.mean())
            dir_baseline = 0.5
        else:
            dir_acc = 0.0
            dir_baseline = 0.5

        per_class = {}
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

        with st.container(border=True):
            st.subheader("Accuracy Metrics")
            st.caption("Model prediction accuracy against test set labels")

            acc_cols = st.columns(4, gap="small")
            _render_metric_card(
                acc_cols[0],
                "Directional Accuracy",
                f"{dir_acc:.1%}",
                f"+{(dir_acc - dir_baseline) * 100:.1f}pp vs random",
                "#3b82f6",
            )
            _render_metric_card(
                acc_cols[1],
                "Exact-Match Accuracy",
                f"{exact_acc:.1%}",
                None,
                "#8b5cf6",
            )
            _render_metric_card(
                acc_cols[2],
                "Directional Baseline",
                f"{dir_baseline:.1%}",
                "Random guess baseline",
                "#6b7280",
            )
            _render_metric_card(
                acc_cols[3],
                "Test Samples",
                f"{total:,}",
                None,
                "#10b981",
            )

        st.subheader("Per-Class Performance")
        cls_col1, cls_col2, cls_col3 = st.columns(3)
        for idx, (name, metrics) in enumerate(per_class.items()):
            col = [cls_col1, cls_col2, cls_col3][idx]
            with col:
                st.markdown(f"**{name}**")
                st.caption(
                    f"True: {metrics['true_count']:,} | Predicted: {metrics['pred_count']:,}"
                )
                st.progress(metrics["recall"], text=f"Recall: {metrics['recall']:.1%}")
                st.progress(
                    metrics["precision"], text=f"Precision: {metrics['precision']:.1%}"
                )
                st.progress(metrics["f1"], text=f"F1: {metrics['f1']:.2f}")

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Confusion Matrix")
            chart = build_confusion_matrix_chart(y_true, y_pred)
            _render_chart(chart, height="500px")
        with col2:
            st.subheader("Confidence Distribution")
            chart = build_confidence_distribution_chart(preds)
            _render_chart(chart, height="500px")

        st.subheader("Prediction Distribution")
        pred_counts = {
            "Short": int((y_pred == -1).sum()),
            "Hold": int((y_pred == 0).sum()),
            "Long": int((y_pred == 1).sum()),
        }
        true_counts = {
            "Short": int((y_true == -1).sum()),
            "Hold": int((y_true == 0).sum()),
            "Long": int((y_true == 1).sum()),
        }
        labels = list(true_counts.keys())
        actual_vals = [true_counts[k] for k in labels]
        predicted_vals = [pred_counts[k] for k in labels]
        dist_chart = (
            Bar(init_opts=opts.InitOpts(height="400px"))
            .add_xaxis(labels)
            .add_yaxis(
                series_name="Actual",
                y_axis=actual_vals,
                itemstyle_opts=opts.ItemStyleOpts(color=COLORS["primary"]),
                label_opts=opts.LabelOpts(is_show=True, position="top"),
            )
            .add_yaxis(
                series_name="Predicted",
                y_axis=predicted_vals,
                itemstyle_opts=opts.ItemStyleOpts(color=COLORS["secondary"]),
                label_opts=opts.LabelOpts(is_show=True, position="top"),
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title="Actual vs Predicted Label Distribution"
                ),
                xaxis_opts=opts.AxisOpts(name="Label"),
                yaxis_opts=opts.AxisOpts(name="Count"),
                tooltip_opts=opts.TooltipOpts(trigger="axis"),
                legend_opts=opts.LegendOpts(),
            )
        )
        _render_chart(dist_chart, height="400px")
    else:
        st.info("No predictions data available.")

    if fi:
        st.subheader("LightGBM Feature Importance")
        chart = build_feature_importance_chart(fi)
        _render_chart(chart, height="600px")
    else:
        st.info("No feature importance data available.")

    shap_data = data.get("shap_values")
    if shap_data:
        st.subheader("SHAP Summary")
        chart = build_shap_chart(shap_data)
        _render_chart(chart, height="600px")
    elif session_dir:
        shap_png = Path(session_dir) / "reports" / "shap_summary.png"
        if shap_png.exists():
            st.subheader("SHAP Summary")
            st.image(str(shap_png), width="stretch")
