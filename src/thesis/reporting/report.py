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
from thesis.reporting.metrics import compute_all_classification_metrics
from thesis.reporting.plots import (
    load_feature_importance,
    plot_confusion_matrix,
    plot_equity_curve,
    plot_feature_importance,
)
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
    architecture = config.model.architecture
    if architecture in ("static", "lgbm"):
        return "LightGBM"
    if architecture == "stacking":
        return "Hybrid Stacking"
    return f"{architecture.title()} Model"


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


def load_prediction_stats(preds_path: Path) -> dict | None:
    """Load final prediction CSV and compute classification statistics."""
    if not preds_path.exists():
        return None
    try:
        df = pl.read_csv(preds_path)
        true = df["true_label"].to_numpy()
        pred = df["pred_label"].to_numpy()
        proba_cols = [
            "pred_proba_class_minus1",
            "pred_proba_class_0",
            "pred_proba_class_1",
        ]
        proba = (
            df.select(proba_cols).to_numpy()
            if all(c in df.columns for c in proba_cols)
            else None
        )
        raw = compute_all_classification_metrics(true, pred, y_proba=proba)
        per_class_metrics = raw["precision_recall_f1_per_class"]
        class_map = {-1: "Short", 0: "Hold", 1: "Long"}
        per_class = {
            class_map[c]: {
                "true_count": int((true == c).sum()),
                "pred_count": int((pred == c).sum()),
                "precision": float(per_class_metrics[class_map[c]]["precision"]),
                "recall": float(per_class_metrics[class_map[c]]["recall"]),
                "f1": float(per_class_metrics[class_map[c]]["f1"]),
            }
            for c in (-1, 0, 1)
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
        }
        return result
    except (ComputeError, ColumnNotFoundError, OSError):
        logger.warning("Failed to load prediction stats: %s", preds_path, exc_info=True)
        return None


def build_model_comparison_rows(
    config: Config, pred_stats: dict | None
) -> list[dict[str, Any]]:
    """Build compact model comparison rows for report tables."""
    rows: list[dict[str, Any]] = []
    if pred_stats:
        rows.append(
            {
                "model": model_label(config),
                "accuracy": pred_stats.get("accuracy"),
                "macro_f1": pred_stats.get("macro_f1"),
                "directional_accuracy": pred_stats.get("directional_accuracy"),
                "source": "current_session",
            }
        )
    preds_path = Path(config.paths.predictions)
    if preds_path.exists():
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
            for baseline_key, label in (("majority_class", "Majority Baseline"),):
                if baseline_key not in baselines:
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
        except (ColumnNotFoundError, ValueError):
            logger.warning("Failed to build baseline rows", exc_info=True)
    session_dir = config.paths.session_dir
    existing = {str(r["model"]).lower() for r in rows}
    comparison_json = (
        Path(session_dir) / "reports" / "model_comparison.json" if session_dir else None
    )
    if comparison_json and comparison_json.exists():
        try:
            model_comparison = json.loads(comparison_json.read_text())
        except (OSError, json.JSONDecodeError):
            model_comparison = {}
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


def _build_thesis_report(
    config: Config,
    metrics: dict,
    pred_stats: dict | None,
    model_comparison_rows: list[dict[str, Any]],
) -> str:
    L: list[str] = []

    # Header
    L.append("# Báo cáo thí nghiệm — Hybrid Stacking dự báo tín hiệu XAU/USD")
    L.append("")
    L.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    L.append("")

    # 1. Mục tiêu
    L.append("## 🎯 1. Mục tiêu")
    L.append("")
    L.append(
        "Phân loại tín hiệu XAU/USD khung H1 thành 3 lớp: **Short / Hold / Long**."
    )
    L.append("Trọng tâm: đánh giá mô hình ML, không phải hệ thống giao dịch tự động.")
    L.append("")

    # 2. Pipeline
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

    # 3. Dữ liệu
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

    # 4. Thiết kế nhãn
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
            L.append(_md_table(["Class", "Tỷ lệ"], []))
            for label_val, name in [(-1, "Short"), (0, "Hold"), (1, "Long")]:
                count = int((df["label"] == label_val).sum())
                pct = count / total * 100 if total > 0 else 0.0
                L.append(f"| {name} | {pct:.1f}% |")
            L.append(f"| **Total** | **{total:,}** |")
        except (ComputeError, OSError):
            logger.warning("Failed to load labels", exc_info=True)
    L.append("")

    # 5. Mô hình
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
    L.append("→ Dự đoán Short / Hold / Long")
    L.append("```")
    L.append("")

    # 6. Validation
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

    # 7. Kết quả chính
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
                    ["Directional Accuracy", _fmt_pct(dir_acc * 100), "⚠️ Gần random"],
                    ["Majority Baseline", _fmt_pct(maj_bl * 100), "Tham chiếu"],
                ],
            )
        )
        L.append("")
        if acc < maj_bl:
            L.append("> **Kết luận:** Hybrid Stacking chưa vượt Majority Baseline.")
        else:
            L.append("> **Kết luận:** Hybrid Stacking vượt Majority Baseline.")
    else:
        L.append("Không có kết quả dự đoán.")
    L.append("")

    # 8. Phân tích lỗi
    L.append("## 🔍 8. Phân tích lỗi")
    L.append("")
    if pred_stats:
        per_class = pred_stats.get("per_class", {})
        L.append(
            _md_table(
                ["Vấn đề", "Giải thích"],
                [
                    [
                        "Hold F1 thấp",
                        f"F1 = {per_class.get('Hold', {}).get('f1', 0):.4f}",
                    ],
                    ["Stacking chưa hiệu quả", "Base models có thể dự đoán giống nhau"],
                    ["Directional Accuracy yếu", "Tín hiệu hướng giá chưa đủ rõ"],
                ],
            )
        )
    else:
        L.append("Không có dữ liệu phân tích.")
    L.append("")

    # 9. Backtest demo
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

    # 10. Kết luận
    L.append("## ✅ 10. Kết luận")
    L.append("")
    L.append("Dự án đã xây dựng pipeline ML hoàn chỉnh:")
    L.append("- Causal feature engineering")
    L.append("- Triple-barrier labeling")
    L.append("- Walk-forward validation")
    L.append("- Baseline comparison")
    L.append("- Hybrid Stacking model")
    L.append("")
    L.append("Tuy nhiên, kết quả hiện tại cho thấy Hybrid Stacking chưa vượt")
    L.append("LightGBM đơn lẻ.")
    L.append("Hướng cải thiện: giảm nhiễu class Hold, thử bài toán 2 class,")
    L.append("tăng đa dạng base models.")
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

    # 1. Thông tin thí nghiệm
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

    # 2. Kết quả chính
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
                    ["Directional Accuracy", _fmt_pct(dir_acc * 100), "⚠️ Gần random"],
                    ["Balanced Accuracy", _fmt_pct(bal_acc * 100), "⚠️ Chưa tốt"],
                    ["Total Predictions", f"{pred_stats.get('total', 0):,}", ""],
                ],
            )
        )
    else:
        L.append("Không có kết quả.")
    L.append("")

    # 3. Kết quả theo class
    L.append("## 3. Kết quả theo class")
    L.append("")
    if pred_stats:
        per_class = pred_stats.get("per_class", {})
        L.append(
            _md_table(
                ["Class", "Precision", "Recall", "F1", "Nhận xét"],
                [
                    [
                        "Short",
                        f"{per_class.get('Short', {}).get('precision', 0):.4f}",
                        f"{per_class.get('Short', {}).get('recall', 0):.4f}",
                        f"{per_class.get('Short', {}).get('f1', 0):.4f}",
                        "Trung bình",
                    ],
                    [
                        "Hold",
                        f"{per_class.get('Hold', {}).get('precision', 0):.4f}",
                        f"{per_class.get('Hold', {}).get('recall', 0):.4f}",
                        f"{per_class.get('Hold', {}).get('f1', 0):.4f}",
                        "🔴 Yếu nhất",
                    ],
                    [
                        "Long",
                        f"{per_class.get('Long', {}).get('precision', 0):.4f}",
                        f"{per_class.get('Long', {}).get('recall', 0):.4f}",
                        f"{per_class.get('Long', {}).get('f1', 0):.4f}",
                        "Tốt nhất",
                    ],
                ],
            )
        )
    L.append("")

    # 4. So sánh mô hình
    L.append("## 4. So sánh mô hình")
    L.append("")
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
            note = "⚠️ Chưa vượt LightGBM"
        elif model == "Majority Baseline":
            note = "Baseline đơn giản"
        elif model == "LightGBM":
            note = "✅ Base tốt nhất"
        else:
            note = "Base model"
        L.append(f"| {model} | {acc} | {f1} | {note} |")
    L.append("")

    # 5. Nhận xét ngắn
    L.append("## 5. Nhận xét ngắn")
    L.append("")
    if pred_stats:
        per_class = pred_stats.get("per_class", {})
        hold_f1 = per_class.get("Hold", {}).get("f1", 0)
        L.append("- ⚠️ Hybrid Stacking chưa vượt LightGBM.")
        L.append(
            "- 🔴 Class Hold khó học nhất."
            if hold_f1 < 0.20
            else "- 🟡 Class Hold yếu."
        )
        L.append("- ⚠️ Directional Accuracy gần random, chưa có edge rõ.")
        L.append(
            "- ✅ Pipeline ML hợp lệ: feature causal, triple-barrier label, "
            "walk-forward validation."
        )
    else:
        L.append("- Không có dữ liệu dự đoán.")
    L.append("")

    # 6. Hướng thử tiếp
    L.append("## 6. Hướng thử tiếp")
    L.append("")
    L.append("1. Thử bài toán 2 class: Short / Long.")
    L.append("2. Giữ LightGBM làm baseline chính.")
    L.append("3. Cải thiện label Hold hoặc bỏ Hold khỏi thí nghiệm phụ.")
    L.append("")

    return "\n".join(L)


def write_model_comparison_csv(out_dir: Path, rows: list[dict[str, Any]]) -> Path:
    """Write model comparison rows to CSV and return output path."""
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


def generate_report(config: Config) -> None:
    """Generate thesis report and model evaluation in new Vietnamese-first format."""
    _setup_matplotlib()
    out_dir = (
        Path(config.paths.session_dir) / "reports"
        if config.paths.session_dir
        else Path("results")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(config.paths.report)
    bt_path = Path(config.paths.backtest_results)

    metrics: dict = {}
    trades: list[dict] = []
    if bt_path.exists():
        with console.status("[cyan]Loading backtest results[/]"):
            with open(bt_path) as f:
                bt = json.load(f)
            metrics = bt.get("metrics", {})
            trades = bt.get("trades", [])

    with console.status("[cyan]Rendering report charts[/]"):
        fi_path = (
            Path(config.paths.session_dir) / "reports" / "feature_importance.json"
            if config.paths.session_dir
            else out_dir.parent / "feature_importance.json"
        )
        feature_importance = load_feature_importance(fi_path)
        plot_equity_curve(
            trades, config.backtest.initial_capital, out_dir / "equity_curve.png"
        )
        plot_feature_importance(feature_importance, out_dir / "feature_importance.png")

    with console.status("[cyan]Building reports[/]"):
        pred_stats = load_prediction_stats(Path(config.paths.predictions))
        model_comparison_rows = build_model_comparison_rows(config, pred_stats)

        if pred_stats and pred_stats.get("confusion_matrix"):
            preds_path = Path(config.paths.predictions)
            if preds_path.exists():
                try:
                    df = pl.read_csv(preds_path)
                    if "true_label" in df.columns and "pred_label" in df.columns:
                        plot_confusion_matrix(
                            df["true_label"].to_numpy(),
                            df["pred_label"].to_numpy(),
                            labels=["Short", "Hold", "Long"],
                            output_path=out_dir / "confusion_matrix.png",
                        )
                except (ComputeError, ColumnNotFoundError):
                    logger.warning("Failed to render confusion matrix", exc_info=True)

    thesis_md = _build_thesis_report(config, metrics, pred_stats, model_comparison_rows)
    model_eval_md = _build_model_evaluation(config, pred_stats, model_comparison_rows)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(thesis_md)
    logger.info("Thesis report saved: %s", report_path)

    model_eval_path = out_dir / "model_evaluation.md"
    with model_eval_path.open("w", encoding="utf-8") as f:
        f.write(model_eval_md)
    logger.info("Model evaluation saved: %s", model_eval_path)

    metrics_json_path = out_dir / "metrics.json"
    with metrics_json_path.open("w", encoding="utf-8") as f:
        json.dump(pred_stats or {}, f, indent=2, ensure_ascii=False)
    logger.info("Metrics saved: %s", metrics_json_path)

    model_cmp_csv = write_model_comparison_csv(out_dir, model_comparison_rows)
    logger.info("Model comparison saved: %s", model_cmp_csv)
