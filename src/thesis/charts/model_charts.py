"""Model performance interactive ECharts: confusion matrix, confidence, feature importance."""

import logging

import numpy as np
import polars as pl
from pyecharts import options as opts
from pyecharts.charts import Bar, HeatMap

from .data import COLORS

logger = logging.getLogger("thesis.charts.model_charts")


def build_confusion_matrix_chart(
    true: np.ndarray,
    pred: np.ndarray,
) -> HeatMap:
    """Build a normalized confusion-matrix heatmap for 3-class labels.

    Args:
        true: Ground-truth labels encoded as `-1`, `0`, or `1`.
        pred: Predicted labels encoded as `-1`, `0`, or `1`.

    Returns:
        A pyecharts `HeatMap` showing row-normalized confusion values.
    """
    labels_order = [-1, 0, 1]
    display_labels = ["Short (-1)", "Hold (0)", "Long (1)"]
    n = len(labels_order)

    # Build raw confusion matrix
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(true, pred, strict=True):
        if t in labels_order and p in labels_order:
            ti = labels_order.index(int(t))
            pi = labels_order.index(int(p))
            cm[ti, pi] += 1

    # Normalize by row
    cm_norm = cm.astype(float)
    for i in range(n):
        row_sum = cm[i].sum()
        if row_sum > 0:
            cm_norm[i] = cm[i] / row_sum

    data = []
    for i in range(n):
        for j in range(n):
            data.append([j, i, round(float(cm_norm[i, j]), 3)])

    chart = (
        HeatMap(init_opts=opts.InitOpts(height="500px"))
        .add_xaxis(display_labels)
        .add_yaxis(
            series_name="Confusion",
            yaxis_data=display_labels,
            value=data,
            label_opts=opts.LabelOpts(is_show=True),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Normalized Confusion Matrix (Test Set)"),
            visualmap_opts=opts.VisualMapOpts(
                min_=0,
                max_=1,
                is_calculable=True,
                orient="vertical",
                pos_right="0%",
                pos_top="center",
                range_color=["#FFFFFF", "#93C5FD", "#2563EB"],
            ),
            tooltip_opts=opts.TooltipOpts(
                trigger="item",
            ),
        )
    )
    return chart


def build_confidence_distribution_chart(preds_df: pl.DataFrame) -> Bar:
    """Build grouped confidence-distribution bars for long/short predictions.

    Args:
        preds_df: Prediction dataframe containing predicted labels and class
            probabilities.

    Returns:
        A pyecharts `Bar` chart with normalized long/short confidence
        distributions, or an empty chart when required columns are missing.
    """
    if "pred_label" not in preds_df.columns:
        return Bar()
    y_pred = preds_df["pred_label"].to_numpy()

    if "pred_proba_class_1" not in preds_df.columns:
        return Bar()

    if "pred_proba_class_minus1" not in preds_df.columns:
        return Bar()

    long_conf = preds_df["pred_proba_class_1"].to_numpy()
    short_conf = preds_df["pred_proba_class_minus1"].to_numpy()

    long_vals = long_conf[y_pred == 1]
    short_vals = short_conf[y_pred == -1]

    # Histogram bins - use 20 bins for cleaner visualization
    bins = np.linspace(0, 1, 21)
    long_counts, _ = np.histogram(long_vals, bins=bins)
    short_counts, _ = np.histogram(short_vals, bins=bins)
    bin_labels = [f"{bins[i]:.2f}" for i in range(len(bins) - 1)]

    # Normalize to relative frequency (percentage) for comparison
    long_total = long_counts.sum()
    short_total = short_counts.sum()
    long_pct = (long_counts / long_total * 100) if long_total > 0 else long_counts
    short_pct = (short_counts / short_total * 100) if short_total > 0 else short_counts

    chart = (
        Bar(init_opts=opts.InitOpts(height="500px"))
        .add_xaxis(bin_labels)
        .add_yaxis(
            series_name="Long",
            y_axis=[round(v, 2) for v in long_pct.tolist()],
            itemstyle_opts=opts.ItemStyleOpts(color=COLORS["long"]),
            label_opts=opts.LabelOpts(is_show=False),
        )
        .add_yaxis(
            series_name="Short",
            y_axis=[round(v, 2) for v in short_pct.tolist()],
            itemstyle_opts=opts.ItemStyleOpts(color=COLORS["short"]),
            label_opts=opts.LabelOpts(is_show=False),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Prediction Confidence Distribution"),
            xaxis_opts=opts.AxisOpts(
                name="Confidence", axislabel_opts=opts.LabelOpts(rotate=30)
            ),
            yaxis_opts=opts.AxisOpts(name="Relative Frequency (%)"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            legend_opts=opts.LegendOpts(),
            datazoom_opts=[
                opts.DataZoomOpts(
                    is_show=False,
                    type_="slider",
                    range_start=0,
                    range_end=100,
                ),
                opts.DataZoomOpts(type_="inside", range_start=0, range_end=100),
            ],
        )
    )
    return chart


def build_feature_importance_chart(
    fi: dict[str, float],
    top_n: int = 20,
) -> Bar:
    """Build a horizontal top-N feature-importance chart.

    Args:
        fi: Mapping from feature name to importance score.
        top_n: Number of top-ranked features to display.

    Returns:
        A pyecharts `Bar` chart with stacked GRU/static contributions.
    """
    items = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:top_n]
    items = items[::-1]
    names = [n for n, _ in items]

    # Split into two series: static features and GRU features
    static_values = [v if not n.startswith("gru_") else 0 for n, v in items]
    gru_values = [v if n.startswith("gru_") else 0 for n, v in items]

    chart = (
        Bar(init_opts=opts.InitOpts(height="600px"))
        .add_xaxis(names)
        .add_yaxis(
            series_name="Static Features",
            y_axis=static_values,
            stack="importance",
            label_opts=opts.LabelOpts(is_show=False),
            itemstyle_opts=opts.ItemStyleOpts(color=COLORS["primary"]),
        )
        .add_yaxis(
            series_name="GRU Features",
            y_axis=gru_values,
            stack="importance",
            label_opts=opts.LabelOpts(is_show=False),
            itemstyle_opts=opts.ItemStyleOpts(color=COLORS["secondary"]),
        )
        .reversal_axis()
        .set_global_opts(
            title_opts=opts.TitleOpts(title=f"Feature Importance (Top {top_n})"),
            xaxis_opts=opts.AxisOpts(name="Importance"),
            yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(font_size=9)),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            legend_opts=opts.LegendOpts(),
        )
    )
    return chart


def build_shap_chart(shap_data: dict, top_n: int = 20) -> Bar:
    """Build a stacked horizontal SHAP-importance chart by class.

    Args:
        shap_data: SHAP payload containing feature names, class names, and
            mean absolute SHAP arrays.
        top_n: Number of top features to display.

    Returns:
        A pyecharts `Bar` chart showing class-wise mean |SHAP| values, or an
        empty chart when SHAP input is incomplete.
    """
    features = shap_data.get("features", [])
    class_names = shap_data.get("class_names", ["Short", "Hold", "Long"])
    mean_abs_shap = shap_data.get("mean_abs_shap", [])

    if not features or not mean_abs_shap:
        return Bar()

    # Validate that SHAP array lengths match feature count
    for cls_idx, cls_vals in enumerate(mean_abs_shap):
        if len(cls_vals) != len(features):
            logger.warning(
                "SHAP class %d has %d values but %d features — skipping chart",
                cls_idx,
                len(cls_vals),
                len(features),
            )
            return Bar()

    # Compute total importance per feature for sorting
    totals = [
        sum(cls_vals[i] for cls_vals in mean_abs_shap) for i in range(len(features))
    ]
    sorted_indices = sorted(
        range(len(features)), key=lambda i: totals[i], reverse=True
    )[:top_n]
    sorted_indices = sorted_indices[
        ::-1
    ]  # Reverse for horizontal bar (bottom = highest)

    sorted_features = [features[i] for i in sorted_indices]
    class_colors = [COLORS["short"], COLORS["flat"], COLORS["long"]]

    chart = Bar(init_opts=opts.InitOpts(height="600px"))
    chart.add_xaxis(sorted_features)

    for cls_idx, cls_name in enumerate(class_names):
        if cls_idx < len(mean_abs_shap):
            cls_vals = mean_abs_shap[cls_idx]
            y_data = [round(cls_vals[i], 4) for i in sorted_indices]
        else:
            y_data = [0] * len(sorted_features)
        chart.add_yaxis(
            series_name=cls_name,
            y_axis=y_data,
            stack="shap",
            label_opts=opts.LabelOpts(is_show=False),
            itemstyle_opts=opts.ItemStyleOpts(
                color=class_colors[cls_idx]
                if cls_idx < len(class_colors)
                else COLORS["primary"]
            ),
        )

    chart.reversal_axis().set_global_opts(
        title_opts=opts.TitleOpts(title="SHAP Feature Importance by Class"),
        xaxis_opts=opts.AxisOpts(name="Mean |SHAP Value|"),
        yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(font_size=9)),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        legend_opts=opts.LegendOpts(),
    )
    return chart
