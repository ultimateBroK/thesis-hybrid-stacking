"""Model performance interactive ECharts: confusion matrix, confidence, feature importance."""

import numpy as np
import polars as pl
from pyecharts import options as opts
from pyecharts.charts import Bar, HeatMap

from .data import COLORS


def build_confusion_matrix_chart(
    true: np.ndarray,
    pred: np.ndarray,
) -> HeatMap:
    """
    Builds a normalized 3×3 confusion matrix heatmap for labels Short (-1), Hold (0), and Long (1).

    The matrix is row-normalized so each true-label row sums to 1; cell values are rounded to three decimals and shown on the heatmap.

    Parameters:
        true: Array of true labels encoded as -1, 0, or 1; must align in length with `pred`.
        pred: Array of predicted labels encoded as -1, 0, or 1; must align in length with `true`.

    Returns:
        A pyecharts HeatMap configured to display the normalized confusion matrix with labeled cells, axis display labels, item tooltips, and a blue visual color scale from 0 to 1.
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
            label_opts=opts.LabelOpts(is_show=True, formatter="{c}"),
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
            tooltip_opts=opts.TooltipOpts(trigger="item"),
        )
    )
    return chart


def build_confidence_distribution_chart(preds_df: pl.DataFrame) -> Bar:
    """
    Create a bar chart showing the distribution of prediction confidence for Long and Short predictions.

    Only rows where `pred_label` equals the corresponding class are used: `pred_proba_class_1` for Long (`pred_label == 1`) and `pred_proba_class_minus1` for Short (`pred_label == -1`). Confidence values are binned into 50 equal-width intervals between 0 and 1. If the `pred_proba_class_1` column is absent, an empty `Bar` is returned.

    Parameters:
        preds_df (pl.DataFrame): Predictions DataFrame. Must contain `pred_label` and `pred_proba_class_1`; `pred_proba_class_minus1` is also required for Short counts.

    Returns:
        Bar: A configured pyecharts `Bar` chart with two series ("Long confidence" and "Short confidence") showing counts per confidence bin, or an empty `Bar` if `pred_proba_class_1` is missing.
    """
    y_pred = preds_df["pred_label"].to_numpy()

    if "pred_proba_class_1" not in preds_df.columns:
        return Bar()

    long_conf = preds_df["pred_proba_class_1"].to_numpy()
    short_conf = preds_df["pred_proba_class_minus1"].to_numpy()

    long_vals = long_conf[y_pred == 1]
    short_vals = short_conf[y_pred == -1]

    # Histogram bins
    bins = np.linspace(0, 1, 51)
    long_counts, _ = np.histogram(long_vals, bins=bins)
    short_counts, _ = np.histogram(short_vals, bins=bins)
    bin_labels = [f"{bins[i]:.2f}" for i in range(len(bins) - 1)]

    chart = (
        Bar(init_opts=opts.InitOpts(height="500px"))
        .add_xaxis(bin_labels)
        .add_yaxis(
            series_name="Long confidence",
            y_axis=long_counts.tolist(),
            itemstyle_opts=opts.ItemStyleOpts(color=COLORS["long"]),
            label_opts=opts.LabelOpts(is_show=False),
        )
        .add_yaxis(
            series_name="Short confidence",
            y_axis=short_counts.tolist(),
            itemstyle_opts=opts.ItemStyleOpts(color=COLORS["short"]),
            label_opts=opts.LabelOpts(is_show=False),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Prediction Confidence Distribution"),
            xaxis_opts=opts.AxisOpts(name="Confidence"),
            yaxis_opts=opts.AxisOpts(name="Count"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            legend_opts=opts.LegendOpts(),
        )
    )
    return chart


def build_feature_importance_chart(
    fi: dict[str, float],
    top_n: int = 20,
) -> Bar:
    """
    Build a horizontal bar chart showing the top feature importances.

    Splits features into "Static Features" and "GRU Features" based on names that start with "gru_" and displays their contributions as two stacked series for the top `top_n` features.

    Args:
        fi (dict[str, float]): Mapping from feature name to importance score.
        top_n (int): Number of top features to display.

    Returns:
        Bar: A pyecharts Bar chart with a reversed (horizontal) axis showing the top features.
    """
    items = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:top_n]
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
