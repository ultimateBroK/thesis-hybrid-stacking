"""Tests for compact report rendering helpers."""

from __future__ import annotations

import pytest

from thesis.reporting.report import (
    _build_model_evaluation,
    _build_thesis_report,
    _md_table,
    build_model_comparison_rows,
    model_label,
)
from thesis.shared.config import Config


@pytest.mark.unit
def test_model_label_stacking() -> None:
    """Stacking architecture renders as Hybrid Stacking."""
    config = Config()
    config.model.architecture = "stacking"

    assert model_label(config) == "Hybrid Stacking"


@pytest.mark.unit
def test_model_label_lgbm() -> None:
    """LightGBM architecture renders as LightGBM."""
    config = Config()
    config.model.architecture = "lgbm"

    assert model_label(config) == "LightGBM"


@pytest.mark.unit
def test_md_table_renders_header_and_rows() -> None:
    """Markdown table helper renders header and body rows."""
    table = _md_table(["Metric", "Value"], [["Accuracy", "36.43%"]])

    assert "| Metric | Value |" in table
    assert "| Accuracy | 36.43% |" in table


@pytest.mark.unit
def test_model_comparison_includes_main_five_models() -> None:
    """Comparison excludes noisy baselines and keeps core models."""
    config = Config()
    config.model.architecture = "stacking"
    pred_stats = {
        "accuracy": 0.3643,
        "macro_f1": 0.3357,
        "directional_accuracy": 0.5026,
    }

    rows = build_model_comparison_rows(config, pred_stats)
    names = [row["model"] for row in rows]

    assert "Hybrid Stacking" in names
    assert "Logistic Regression" in names
    assert "Random Forest" in names
    assert "LightGBM" in names
    assert "Naive Direction" not in names
    assert "Random Baseline" not in names


@pytest.mark.unit
def test_model_evaluation_uses_compact_vietnamese_format() -> None:
    """Model evaluation follows compact report recommendation."""
    config = Config()
    pred_stats = {
        "total": 23752,
        "accuracy": 0.3643,
        "majority_baseline": 0.4901,
        "macro_f1": 0.3357,
        "directional_accuracy": 0.5026,
        "balanced_accuracy": 0.4017,
        "per_class": {
            "Short": {"precision": 0.4388, "recall": 0.3263, "f1": 0.3743},
            "Hold": {"precision": 0.1268, "recall": 0.5037, "f1": 0.2026},
            "Long": {"precision": 0.5042, "recall": 0.3750, "f1": 0.4301},
        },
    }
    rows = [
        {
            "model": "Hybrid Stacking",
            "accuracy": 0.3643,
            "macro_f1": 0.3357,
            "source": "current_session",
        }
    ]

    report = _build_model_evaluation(config, pred_stats, rows)

    assert "# 📊 Model Evaluation — Hybrid Stacking" in report
    assert "## 4. So sánh mô hình" in report
    assert "Naive Direction" not in report
    assert "Random Baseline" not in report
    assert "High-confidence" not in report


@pytest.mark.unit
def test_thesis_report_excludes_calibration_main_section() -> None:
    """Thesis report excludes calibration/generalization sections."""
    config = Config()
    metrics = {
        "return_pct": 11.6,
        "max_drawdown_pct": -4.0,
        "num_trades": 275,
        "profit_factor": 1.28,
    }
    pred_stats = {
        "accuracy": 0.3643,
        "majority_baseline": 0.4901,
        "macro_f1": 0.3357,
        "directional_accuracy": 0.5026,
        "per_class": {"Hold": {"f1": 0.2026}},
    }

    report = _build_thesis_report(config, metrics, pred_stats, [])

    assert "# Báo cáo thí nghiệm" in report
    assert "## 🎯 1. Mục tiêu" in report
    assert "## 💼 9. Backtest demo" in report
    assert "Calibration" not in report
    assert "Generalization" not in report
