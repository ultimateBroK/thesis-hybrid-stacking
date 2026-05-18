"""Model performance charts."""

from __future__ import annotations

import numpy as np
import polars as pl
from pyecharts import options as opts
from pyecharts.charts import Bar, HeatMap

from thesis.charts.shared import COLORS


def build_confusion_matrix_chart(
    true: np.ndarray,
    pred: np.ndarray,
) -> HeatMap:
    """Normalized confusion matrix heatmap for 3-class labels."""
    labels_order = [-1, 0, 1]
    display_labels = ["Short (-1)", "Hold (0)", "Long (1)"]
    n = len(labels_order)

    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(true, pred, strict=True):
        if t in labels_order and p in labels_order:
            ti = labels_order.index(int(t))
            pi = labels_order.index(int(p))
            cm[ti, pi] += 1

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
            tooltip_opts=opts.TooltipOpts(trigger="item"),
        )
    )
    return chart


def build_confidence_distribution_chart(preds_df: pl.DataFrame) -> Bar:
    """Long/short confidence distribution bars."""
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

    bins = np.linspace(0, 1, 21)
    long_counts, _ = np.histogram(long_vals, bins=bins)
    short_counts, _ = np.histogram(short_vals, bins=bins)
    bin_labels = [f"{bins[i]:.2f}" for i in range(len(bins) - 1)]

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
    """Horizontal top-N feature importance chart."""
    items = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:top_n]
    items = items[::-1]
    names = [n for n, _ in items]

    static_values = [v if not n.startswith("model_") else 0 for n, v in items]
    model_values = [v if n.startswith("model_") else 0 for n, v in items]

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
            series_name="Model Features",
            y_axis=model_values,
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


def build_model_comparison_chart(rows: list[dict]) -> Bar:
    """Accuracy/Macro-F1 comparison across baseline and model variants."""
    usable = [
        r
        for r in rows
        if r.get("accuracy") is not None or r.get("macro_f1") is not None
    ]
    if not usable:
        return Bar()

    names = [str(r.get("model", "Model")) for r in usable]
    accuracy = [
        round(float(r["accuracy"]) * 100, 2) if r.get("accuracy") is not None else None
        for r in usable
    ]
    macro_f1 = [
        round(float(r["macro_f1"]) * 100, 2) if r.get("macro_f1") is not None else None
        for r in usable
    ]

    return (
        Bar(init_opts=opts.InitOpts(height="420px"))
        .add_xaxis(names)
        .add_yaxis(
            series_name="Accuracy (%)",
            y_axis=accuracy,
            itemstyle_opts=opts.ItemStyleOpts(color=COLORS["primary"]),
            label_opts=opts.LabelOpts(is_show=True, position="top"),
        )
        .add_yaxis(
            series_name="Macro F1 (%)",
            y_axis=macro_f1,
            itemstyle_opts=opts.ItemStyleOpts(color=COLORS["secondary"]),
            label_opts=opts.LabelOpts(is_show=True, position="top"),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Model Comparison"),
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=20)),
            yaxis_opts=opts.AxisOpts(name="Score (%)", max_=100),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            legend_opts=opts.LegendOpts(),
        )
    )


__all__ = [
    "build_confusion_matrix_chart",
    "build_confidence_distribution_chart",
    "build_feature_importance_chart",
    "build_model_comparison_chart",
]
