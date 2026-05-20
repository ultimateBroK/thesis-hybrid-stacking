"""Report generation — simplified Vietnamese-first format.

Produces: thesis_report.md, model_evaluation.md.
"""

from __future__ import annotations

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from polars.exceptions import ColumnNotFoundError, ComputeError

from thesis.models import baselines as baselines_mod
from thesis.reporting.figures import export_all_figures
from thesis.reporting.metrics import compute_all_classification_metrics
from thesis.shared.config import Config
from thesis.shared.constants import H1_BARS_PER_YEAR
from thesis.shared.utils import console

logger = logging.getLogger("thesis.report")

BARS_PER_YEAR = H1_BARS_PER_YEAR
MODEL_NAME_MAP = {
    "logreg": "Logistic Regression",
    "rf": "Random Forest",
    "lgbm": "LightGBM",
    "hybrid_stacking": "Hybrid Stacking",
}


def model_label(config: Config) -> str:
    """Return display label for configured model architecture."""
    return "Hybrid Stacking"


def _fmt_pct(v: float) -> str:
    return f"{v:.2f}%"


def _fmt_f2(v: float) -> str:
    return f"{v:.2f}"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def load_prediction_stats(preds_path: Path, num_classes: int = 2) -> dict | None:
    """Load prediction CSV and compute per-class classification statistics.

    Supports binary (Short/Long) and ternary (Short/Hold/Long) label modes.
    """
    if not preds_path.exists():
        return None
    try:
        df = pl.read_csv(preds_path)
        true = df["true_label"].to_numpy()
        pred = df["pred_label"].to_numpy()

        classes = [-1, 0, 1] if num_classes == 3 else [-1, 1]
        class_names = {-1: "Short", 0: "Hold", 1: "Long"}

        proba_cols = [
            f"pred_proba_class_{suffix}"
            for cls in classes
            for suffix in (["minus1"] if cls == -1 else ["0"] if cls == 0 else ["1"])
        ]
        proba = (
            df.select(proba_cols).to_numpy()
            if all(c in df.columns for c in proba_cols)
            else None
        )

        raw = compute_all_classification_metrics(
            true,
            pred,
            y_proba=proba,
            classes=classes,
            class_names=class_names,
        )
        per_class_metrics = raw["precision_recall_f1_per_class"]
        per_class = {
            class_names[c]: {
                "true_count": int((true == c).sum()),
                "pred_count": int((pred == c).sum()),
                "precision": float(per_class_metrics[class_names[c]]["precision"]),
                "recall": float(per_class_metrics[class_names[c]]["recall"]),
                "f1": float(per_class_metrics[class_names[c]]["f1"]),
            }
            for c in classes
        }
        result: dict[str, Any] = {
            "total": int(raw["total"]),
            "accuracy": float(raw["accuracy"]),
            "balanced_accuracy": float(raw["balanced_accuracy"]),
            "directional_accuracy": float(raw["directional_accuracy"]),
            "majority_baseline": float(raw["majority_baseline_accuracy"]),
            "macro_f1": float(raw["macro_f1"]),
            "weighted_f1": float(raw["weighted_f1"]),
            "per_class": per_class,
            "confusion_matrix": raw["confusion_matrix"],
            "direction_confusion_matrix": raw["direction_confusion_matrix"],
            "num_classes": num_classes,
        }

        if proba is not None:
            result["threshold_sweep"] = _threshold_sweep(true, pred, proba, classes)

        return result
    except (ComputeError, ColumnNotFoundError, OSError):
        logger.warning("Failed to load prediction stats: %s", preds_path, exc_info=True)
        return None


def _threshold_sweep(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    classes: list[int],
    thresholds: list[float] | None = None,
) -> list[dict[str, float | int]]:
    """Compute accuracy and trade count at each confidence threshold.

    Filters predictions where max probability exceeds threshold.
    """
    if thresholds is None:
        thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]
    max_proba = y_proba.max(axis=1)
    total = len(y_true)
    rows: list[dict[str, float | int]] = []
    for t in thresholds:
        mask = max_proba >= t
        count = int(mask.sum())
        if count == 0:
            rows.append({"threshold": t, "accuracy": 0.0, "count": 0, "pct": 0.0})
            continue
        acc = float((y_true[mask] == y_pred[mask]).mean())
        rows.append(
            {
                "threshold": t,
                "accuracy": acc,
                "count": count,
                "pct": count / total * 100,
            }
        )
    return rows


def _add_current_session_row(
    pred_stats: dict | None, config: Config
) -> list[dict[str, Any]]:
    if not pred_stats:
        return []
    return [
        {
            "model": model_label(config),
            "accuracy": pred_stats.get("accuracy"),
            "macro_f1": pred_stats.get("macro_f1"),
            "directional_accuracy": pred_stats.get("directional_accuracy"),
            "source": "current_session",
        }
    ]


def _add_baseline_rows(config: Config, existing: set[str]) -> list[dict[str, Any]]:
    preds_path = Path(config.paths.predictions)
    if not preds_path.exists():
        return []
    try:
        df = pl.read_csv(preds_path)
        y_true = df["true_label"].to_numpy()
        close_path = Path(config.paths.ohlcv)
        y_returns = np.zeros(len(y_true), dtype=np.float64)
        if close_path.exists():
            ohlcv = pl.read_parquet(close_path, columns=["close"])
            close = ohlcv["close"].to_numpy()
            if len(close) > 1:
                bar_returns = np.diff(close) / close[:-1]
                n = min(len(y_true), len(bar_returns))
                y_returns = bar_returns[-n:]
                y_true = y_true[-n:]
        baselines = baselines_mod.run_all(
            y_true, y_returns, seed=config.workflow.random_seed
        )
        rows: list[dict[str, Any]] = []
        for baseline_key, label in (("majority_class", "Majority Baseline"),):
            if baseline_key not in baselines:
                continue
            if label.lower() in existing:
                continue
            m = baselines[baseline_key]
            rows.append(
                {
                    "model": label,
                    "accuracy": m.get("accuracy"),
                    "macro_f1": m.get("macro_f1"),
                    "directional_accuracy": m.get("directional_accuracy"),
                    "source": "derived_baseline",
                }
            )
            existing.add(label.lower())
        return rows
    except (ColumnNotFoundError, ValueError):
        logger.warning("Failed to build baseline rows", exc_info=True)
        return []


def _add_json_comparison_rows(
    session_dir: str | None, existing: set[str]
) -> list[dict[str, Any]]:
    if not session_dir:
        return []
    comparison_json = Path(session_dir) / "reports" / "model_comparison.json"
    if not comparison_json.exists():
        return []
    try:
        model_comparison = json.loads(comparison_json.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    rows: list[dict[str, Any]] = []
    for key, metrics in model_comparison.items():
        model_name = MODEL_NAME_MAP.get(key, str(key).replace("_", " ").title())
        if model_name.lower() in existing:
            continue
        rows.append(
            {
                "model": model_name,
                "accuracy": metrics.get("accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "directional_accuracy": metrics.get("directional_accuracy"),
                "source": "walk_forward_model_comparison",
            }
        )
        existing.add(model_name.lower())
    return rows


def _fill_missing_models(existing: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_name in (
        "Logistic Regression",
        "Random Forest",
        "LightGBM",
        "Hybrid Stacking",
    ):
        if model_name.lower() in existing:
            continue
        rows.append(
            {
                "model": model_name,
                "accuracy": None,
                "macro_f1": None,
                "directional_accuracy": None,
                "source": "pending_experiment",
            }
        )
    return rows


def build_model_comparison_rows(
    config: Config, pred_stats: dict | None
) -> list[dict[str, Any]]:
    """Build compact model comparison rows for report tables."""
    rows = _add_current_session_row(pred_stats, config)
    existing = {str(r["model"]).lower() for r in rows}
    rows.extend(_add_baseline_rows(config, existing))
    rows.extend(_add_json_comparison_rows(config.paths.session_dir, existing))
    rows.extend(_fill_missing_models(existing))
    return rows


def _build_thesis_report(
    config: Config,
    metrics: dict,
    pred_stats: dict | None,
    model_comparison_rows: list[dict[str, Any]],
) -> str:
    L: list[str] = []

    L.append("# Báo cáo thí nghiệm — Hybrid Stacking dự báo tín hiệu XAU/USD")
    L.append("")
    L.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    L.append("")

    L.append("## 🎯 1. Mục tiêu")
    L.append("")
    num_label_classes = config.labels.num_classes
    label_mode_desc = (
        "3 lớp: **Short / Hold / Long**"
        if num_label_classes == 3
        else "2 lớp: **Short / Long**"
    )
    L.append("Phân loại tín hiệu XAU/USD khung H1 thành " + label_mode_desc + ".")
    L.append("Trọng tâm: đánh giá mô hình ML, không phải hệ thống giao dịch tự động.")
    L.append("")

    L.append("## 🧱 2. Pipeline tổng quan")
    L.append("")
    L.append("```")
    L.append("Dữ liệu XAU/USD H1")
    L.append("→ Feature Engineering")
    L.append("→ Triple-barrier Labeling")
    L.append("→ Walk-forward Validation")
    L.append("→ Hybrid Stacking")
    L.append("→ Classification Metrics")
    L.append("```")
    L.append("")

    L.append("## 📦 3. Dữ liệu")
    L.append("")
    dq_path = Path(config.paths.data_quality_json)
    if dq_path.exists():
        try:
            with open(dq_path) as f:
                dq = json.load(f)
            L.append(
                _md_table(
                    ["Mục", "Giá trị"],
                    [
                        ["Số lượng bars", f"{dq.get('total_bars', 0):,}"],
                        ["Thời gian bắt đầu", str(dq.get("start_date", "N/A"))[:10]],
                        ["Thời gian kết thúc", str(dq.get("end_date", "N/A"))[:10]],
                        ["Real gaps", f"{dq.get('real_gaps', 0):,}"],
                    ],
                )
            )
        except (OSError, json.JSONDecodeError):
            L.append("Dữ liệu chất lượng: không đọc được.")
    else:
        L.append("Data quality JSON not found.")
    L.append("")

    L.append("## 🏷️ 4. Thiết kế nhãn")
    L.append("")
    L.append("Triple-barrier labeling:")
    L.append(
        _md_table(
            ["Tham số", "Giá trị"],
            [
                ["TP", f"{config.labels.atr_tp_multiplier} × ATR"],
                ["SL", f"{config.labels.atr_sl_multiplier} × ATR"],
                ["Horizon", f"{config.labels.horizon_bars} bars"],
                ["Số lớp", str(config.labels.num_classes)],
            ],
        )
    )
    L.append("")
    labels_path = Path(config.paths.labels)
    if labels_path.exists():
        try:
            df = pl.read_parquet(labels_path, columns=["label"])
            total = len(df)
            L.append("**Phân phối nhãn:**")
            L.append("")
            label_items = [(-1, "Short"), (1, "Long")]
            if config.labels.num_classes == 3:
                label_items = [(-1, "Short"), (0, "Hold"), (1, "Long")]
            L.append(_md_table(["Class", "Tỷ lệ"], []))
            for label_val, name in label_items:
                count = int((df["label"] == label_val).sum())
                pct = count / total * 100 if total > 0 else 0.0
                L.append(f"| {name} | {pct:.1f}% |")
            L.append(f"| **Total** | **{total:,}** |")
        except (ComputeError, OSError):
            logger.warning("Failed to load labels", exc_info=True)
    L.append("")

    L.append("## 🤖 5. Mô hình")
    L.append("")
    L.append("Hybrid Stacking:")
    L.append(
        _md_table(
            ["Tầng", "Mô hình"],
            [
                ["Base models", "Logistic Regression, Random Forest, LightGBM"],
                ["Meta model", "Logistic Regression"],
            ],
        )
    )
    L.append("")
    L.append("```")
    L.append("Base models dự đoán xác suất")
    L.append("→ Meta model học cách kết hợp")
    predict_desc = (
        "→ Dự đoán Short / Hold / Long"
        if config.labels.num_classes == 3
        else "→ Dự đoán Short / Long"
    )
    L.append(predict_desc)
    L.append("```")
    L.append("")

    L.append("## 🧪 6. Đánh giá")
    L.append("")
    L.append("Walk-forward validation (sliding window):")
    L.append(
        _md_table(
            ["Tham số", "Giá trị"],
            [
                ["Train window", f"{config.validation.train_window_bars:,} bars"],
                ["Test window", f"{config.validation.test_window_bars:,} bars"],
                ["Purge gap", f"{config.validation.purge_bars} bars"],
                ["Embargo gap", f"{config.validation.embargo_bars} bars"],
            ],
        )
    )
    L.append("")

    L.append("## 📊 7. Kết quả chính")
    L.append("")
    if pred_stats:
        acc = pred_stats["accuracy"]
        maj_bl = pred_stats["majority_baseline"]
        macro_f1 = pred_stats["macro_f1"]
        dir_acc = pred_stats["directional_accuracy"]
        L.append(
            _md_table(
                ["Metric", "Giá trị", "Đánh giá"],
                [
                    [
                        "Accuracy",
                        _fmt_pct(acc * 100),
                        "⚠️ Thấp" if acc < maj_bl else "🟡",
                    ],
                    [
                        "Macro F1",
                        f"{macro_f1:.4f}",
                        "⚠️ Yếu" if macro_f1 < 0.35 else "🟡",
                    ],
                    [
                        "Directional Accuracy",
                        _fmt_pct(dir_acc * 100),
                        "⚠️ Gần random"
                        if dir_acc < 0.55
                        else ("🟢 Có edge" if dir_acc > 0.60 else "🟡"),
                    ],
                    ["Majority Baseline", _fmt_pct(maj_bl * 100), "Tham chiếu"],
                ],
            )
        )
        L.append("")
        if acc < maj_bl:
            L.append(
                f"> **Kết luận:** {model_label(config)} chưa vượt Majority Baseline."
            )
        else:
            L.append(f"> **Kết luận:** {model_label(config)} vượt Majority Baseline.")
    else:
        L.append("Không có kết quả dự đoán.")
    L.append("")

    L.append("## 🔍 8. Phân tích lỗi")
    L.append("")
    if pred_stats:
        per_class = pred_stats.get("per_class", {})
        rows: list[list[str]] = []
        if per_class:
            if acc < maj_bl:
                rows.append(
                    [
                        "Stacking chưa vượt baseline",
                        "Cần cải thiện base model diversity",
                    ]
                )

            da = pred_stats.get("directional_accuracy", 0)
            if da < 0.55:
                rows.append(["Directional Accuracy gần random", f"DA = {da:.4f}"])
            elif da < 0.60:
                rows.append(["Directional Accuracy trung bình", f"DA = {da:.4f}"])
            else:
                rows.append(["Directional Accuracy có edge", f"DA = {da:.4f}"])

        if rows:
            L.append(_md_table(["Vấn đề", "Giải thích"], rows))
        else:
            L.append("Không có vấn đề đáng kể.")
    else:
        L.append("Không có dữ liệu phân tích.")
    L.append("")

    L.append("## 💼 9. Backtest demo")
    L.append("")
    L.append("*Backtest chỉ là minh họa ứng dụng, không phải bằng chứng chính.*")
    L.append("")
    if metrics:
        L.append(
            _md_table(
                ["Metric", "Giá trị"],
                [
                    ["Return", _fmt_pct(metrics.get("return_pct", 0))],
                    ["Max Drawdown", _fmt_pct(abs(metrics.get("max_drawdown_pct", 0)))],
                    ["Trades", str(metrics.get("num_trades", 0))],
                    ["Profit Factor", _fmt_f2(metrics.get("profit_factor", 0))],
                ],
            )
        )
    else:
        L.append("Không có kết quả backtest.")
    L.append("")

    L.append("## ✅ 10. Kết luận")
    L.append("")
    L.append("Dự án đã xây dựng pipeline ML hoàn chỉnh:")
    L.append("- Causal feature engineering")
    L.append("- Triple-barrier labeling")
    L.append("- Walk-forward validation")
    L.append("- Baseline comparison")
    L.append(f"- {model_label(config)} model")
    L.append("")

    best_base_name = None
    best_base_acc_val = 0.0
    stacking_acc_val = 0.0
    for row in model_comparison_rows:
        if row.get("accuracy") is None:
            continue
        m = row.get("model", "")
        a = float(row["accuracy"])
        if m == model_label(config):
            stacking_acc_val = a
        elif m not in ("Majority Baseline",) and a > best_base_acc_val:
            best_base_name = m
            best_base_acc_val = a

    if best_base_name and stacking_acc_val > 0:
        if stacking_acc_val >= best_base_acc_val:
            L.append(
                f"{model_label(config)} ({_fmt_pct(stacking_acc_val * 100)}) "
                f"vượt base tốt nhất {best_base_name} "
                f"({_fmt_pct(best_base_acc_val * 100)})."
            )
        else:
            L.append(
                f"Tuy nhiên, kết quả hiện tại cho thấy {model_label(config)} "
                f"({_fmt_pct(stacking_acc_val * 100)}) chưa vượt base tốt nhất "
                f"{best_base_name} ({_fmt_pct(best_base_acc_val * 100)})."
            )
            L.append(
                "Hướng cải thiện: tăng đa dạng base models, điều chỉnh meta learner."
            )
    else:
        L.append("Chưa có đủ dữ liệu so sánh base models.")

    L.append("")

    return "\n".join(L)


def _build_model_evaluation(
    config: Config,
    pred_stats: dict | None,
    model_comparison_rows: list[dict[str, Any]],
) -> str:
    L: list[str] = []
    L.append("# 📊 Model Evaluation — Hybrid Stacking")
    L.append("")
    L.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    L.append("")

    L.append("## 1. Thông tin thí nghiệm")
    L.append("")
    L.append(
        _md_table(
            ["Mục", "Giá trị"],
            [
                ["Dataset", config.data.symbol],
                ["Timeframe", config.data.timeframe],
                ["Model chính", "Hybrid Stacking"],
                ["Validation", "Walk-forward"],
                ["Seed", config.workflow.random_seed],
            ],
        )
    )
    L.append("")

    L.append("## 2. Kết quả chính")
    L.append("")
    if pred_stats:
        acc = pred_stats["accuracy"]
        maj_bl = pred_stats["majority_baseline"]
        macro_f1 = pred_stats["macro_f1"]
        dir_acc = pred_stats["directional_accuracy"]
        bal_acc = pred_stats["balanced_accuracy"]
        L.append(
            _md_table(
                ["Metric", "Giá trị", "Nhận xét"],
                [
                    [
                        "Accuracy",
                        _fmt_pct(acc * 100),
                        "⚠️ Thấp hơn Majority" if acc < maj_bl else "🟡",
                    ],
                    [
                        "Macro F1",
                        f"{macro_f1:.4f}",
                        "⚠️ Yếu" if macro_f1 < 0.35 else "🟡",
                    ],
                    [
                        "Directional Accuracy",
                        _fmt_pct(dir_acc * 100),
                        "⚠️ Gần random"
                        if dir_acc < 0.55
                        else ("🟢 Có edge" if dir_acc > 0.60 else "🟡"),
                    ],
                    [
                        "Balanced Accuracy",
                        _fmt_pct(bal_acc * 100),
                        "⚠️ Thấp" if bal_acc < 0.40 else "🟡",
                    ],
                    ["Total Predictions", f"{pred_stats.get('total', 0):,}", ""],
                ],
            )
        )
    else:
        L.append("Không có kết quả.")
    L.append("")

    L.append("## 3. Kết quả theo class")
    L.append("")
    if pred_stats:
        per_class = pred_stats.get("per_class", {})
        class_names_list = (
            ["Short", "Hold", "Long"]
            if pred_stats.get("num_classes", 2) == 3
            else ["Short", "Long"]
        )
        f1_scores = {
            name: per_class.get(name, {}).get("f1", 0) for name in class_names_list
        }
        nonzero_f1 = {k: v for k, v in f1_scores.items() if v > 0}
        weakest = min(nonzero_f1, key=nonzero_f1.get) if nonzero_f1 else None
        strongest = max(nonzero_f1, key=nonzero_f1.get) if nonzero_f1 else None
        class_rows: list[list[str]] = []
        for class_name in class_names_list:
            pc = per_class.get(class_name, {})
            f1 = f1_scores.get(class_name, 0)
            if class_name == weakest:
                remark = "🔴 Yếu nhất"
            elif class_name == strongest:
                remark = "🟢 Tốt nhất"
            else:
                remark = "🟡"
            class_rows.append(
                [
                    class_name,
                    f"{pc.get('precision', 0):.4f}",
                    f"{pc.get('recall', 0):.4f}",
                    f"{f1:.4f}",
                    remark,
                ]
            )
        L.append(
            _md_table(
                ["Class", "Precision", "Recall", "F1", "Nhận xét"],
                class_rows,
            )
        )
    L.append("")

    L.append("## 4. Confidence Threshold Sweep")
    L.append("")
    sweep = pred_stats.get("threshold_sweep") if pred_stats else None
    if sweep:
        L.append(
            _md_table(
                ["Threshold", "Accuracy", "Predictions", "% Total"],
                [
                    [
                        f"{r['threshold']:.2f}",
                        _fmt_pct(r["accuracy"] * 100),
                        str(r["count"]),
                        f"{r['pct']:.1f}%",
                    ]
                    for r in sweep
                ],
            )
        )
        L.append("")
        best = max(sweep, key=lambda r: r["accuracy"] if r["count"] > 0 else 0)
        L.append(
            f"> Best threshold = {best['threshold']:.2f} "
            f"(accuracy={_fmt_pct(best['accuracy'] * 100)}, "
            f"n={best['count']:,})"
        )
    else:
        L.append("Threshold sweep requires probability predictions.")
    L.append("")

    L.append("## 5. So sánh mô hình")
    L.append("")
    best_base_name_eval = None
    best_base_acc_eval = 0.0
    for r in model_comparison_rows:
        if r.get("accuracy") is None:
            continue
        if r.get("source") not in (
            "derived_baseline",
            "current_session",
            "walk_forward_model_comparison",
        ):
            continue
        m = r.get("model", "")
        if (
            m not in ("Majority Baseline", model_label(config))
            and float(r["accuracy"]) > best_base_acc_eval
        ):
            best_base_name_eval = m
            best_base_acc_eval = float(r["accuracy"])

    stacking_acc_eval: float | None = None
    for r in model_comparison_rows:
        if r.get("model") == model_label(config) and r.get("accuracy"):
            stacking_acc_eval = float(r["accuracy"])
            break

    L.append(_md_table(["Model", "Accuracy", "Macro F1", "Ghi chú"], []))
    for row in model_comparison_rows:
        if row.get("source") not in (
            "derived_baseline",
            "current_session",
            "walk_forward_model_comparison",
        ):
            continue
        model = row.get("model", "")
        acc = _fmt_pct(float(row["accuracy"]) * 100) if row.get("accuracy") else "N/A"
        f1 = f"{float(row['macro_f1']):.4f}" if row.get("macro_f1") else "N/A"
        if row.get("source") == "current_session":
            if stacking_acc_eval and best_base_acc_eval:
                note = (
                    "✅ Vượt base tốt nhất"
                    if stacking_acc_eval >= best_base_acc_eval
                    else "⚠️ Chưa vượt base tốt nhất"
                )
            else:
                note = "Model chính"
        elif model == "Majority Baseline":
            note = "Baseline đơn giản"
        elif model == best_base_name_eval:
            note = "✅ Base tốt nhất"
        else:
            note = "Base model"
        L.append(f"| {model} | {acc} | {f1} | {note} |")
    L.append("")

    L.append("## 6. Nhận xét ngắn")
    L.append("")
    if pred_stats:
        per_class = pred_stats.get("per_class", {})
        if stacking_acc_eval is not None and best_base_name_eval:
            if stacking_acc_eval >= best_base_acc_eval:
                L.append(
                    f"- ✅ {model_label(config)} vượt base tốt nhất "
                    f"({best_base_name_eval})."
                )
            else:
                L.append(f"- ⚠️ {model_label(config)} chưa vượt {best_base_name_eval}.")
        else:
            L.append(f"- {model_label(config)}: chưa đủ dữ liệu so sánh.")

        da = pred_stats.get("directional_accuracy", 0)
        if da < 0.55:
            L.append("- ⚠️ Directional Accuracy gần random, chưa có edge rõ.")
        elif da > 0.60:
            L.append("- ✅ Directional Accuracy cho thấy có edge định hướng.")
        else:
            L.append("- 🟡 Directional Accuracy ở mức trung bình.")

        L.append(
            "- ✅ Pipeline ML hợp lệ: feature causal, triple-barrier label, "
            "walk-forward validation."
        )
    else:
        L.append("- Không có dữ liệu dự đoán.")
    L.append("")

    L.append("## 7. Hướng thử tiếp")
    L.append("")
    L.append("1. Tối ưu ngưỡng confidence theo threshold sweep table ở section 4.")
    L.append("2. Giữ LightGBM làm baseline chính.")
    L.append(
        "3. So sánh xác suất Short/Long với kết quả backtest sau chi phí giao dịch."
    )
    L.append("")

    return "\n".join(L)


def _load_backtest_metrics(config: Config) -> dict:
    bt_path = Path(config.paths.backtest_results)
    if not bt_path.exists():
        return {}
    with console.status("[cyan]Loading backtest results[/]"):
        bt = json.loads(bt_path.read_text())
    return bt.get("metrics", {})


def _load_report_inputs(config: Config) -> tuple[dict | None, list[dict[str, Any]]]:
    with console.status("[cyan]Building reports[/]"):
        pred_stats = load_prediction_stats(
            Path(config.paths.predictions), num_classes=config.labels.num_classes
        )
        model_comparison_rows = build_model_comparison_rows(config, pred_stats)
    return pred_stats, model_comparison_rows


def _write_report_artifacts(
    config: Config,
    out_dir: Path,
    metrics: dict,
    pred_stats: dict | None,
    model_comparison_rows: list[dict[str, Any]],
) -> None:
    thesis_md = _build_thesis_report(config, metrics, pred_stats, model_comparison_rows)
    model_eval_md = _build_model_evaluation(config, pred_stats, model_comparison_rows)

    report_path = Path(config.paths.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(thesis_md, encoding="utf-8")
    logger.info("Thesis report saved: %s", report_path)

    model_eval_path = out_dir / "model_evaluation.md"
    model_eval_path.write_text(model_eval_md, encoding="utf-8")
    logger.info("Model evaluation saved: %s", model_eval_path)

    metrics_json_path = out_dir / "metrics.json"
    metrics_json_path.write_text(
        json.dumps(pred_stats or {}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Metrics saved: %s", metrics_json_path)

    model_cmp_csv = write_model_comparison_csv(out_dir, model_comparison_rows)
    logger.info("Model comparison saved: %s", model_cmp_csv)


def write_model_comparison_csv(out_dir: Path, rows: list[dict[str, Any]]) -> Path:
    """Write model comparison rows to CSV."""
    import pandas as pd

    csv_path = out_dir / "model_comparison.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def _setup_matplotlib() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
        try:
            plt.style.use(style)
            break
        except Exception:
            continue
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        }
    )


def _prepare_report_dir(config: Config) -> Path:
    out_dir = (
        Path(config.paths.session_dir) / "reports"
        if config.paths.session_dir
        else Path("results")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _maybe_export_figures(config: Config, pred_stats: dict | None) -> None:
    if not config.report_figures.enabled:
        return
    with console.status("[cyan]Exporting report figures[/]"):
        cm_dict = pred_stats.get("confusion_matrix") if pred_stats else None
        export_all_figures(
            session_dir=Path(config.paths.session_dir),
            config=config,
            artifacts={"confusion_matrix": cm_dict} if cm_dict else {},
            dpi=config.report_figures.dpi,
            top_n_features=config.report_figures.top_n_features,
        )


def generate_report(config: Config) -> None:
    """Generate thesis report and model evaluation."""
    _setup_matplotlib()
    out_dir = _prepare_report_dir(config)
    metrics = _load_backtest_metrics(config)
    pred_stats, model_comparison_rows = _load_report_inputs(config)
    _write_report_artifacts(config, out_dir, metrics, pred_stats, model_comparison_rows)
    _maybe_export_figures(config, pred_stats)
