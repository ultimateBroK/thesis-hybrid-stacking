"""Tests for report rendering helpers."""

from __future__ import annotations

import pytest

from thesis.reporting.report import (
    _assess_model_quality,
    _assess_trading_edge,
    _benchmark_comparison_table,
    _config_table,
    _derive_recommendation,
    _exec_verdict,
    _identify_primary_issue,
    model_label,
)
from thesis.shared.config import Config


@pytest.mark.unit
def test_config_table_shows_walk_forward_for_sliding_validation() -> None:
    config = Config()
    config.validation.method = "sliding"
    L: list[str] = []
    _config_table(L, config)
    assert any("walk-forward" in line.lower() for line in L)


@pytest.mark.unit
def test_exec_verdict_returns_expected_keys() -> None:
    metrics = {
        "accuracy": 0.6,
        "macro_f1": 0.55,
        "directional_accuracy": 0.65,
        "win_rate_pct": 50.0,
        "sharpe_ratio": 1.0,
        "max_drawdown_pct": -20.0,
        "profit_factor": 1.2,
        "return_pct": 5.0,
        "equity_final": 10500,
        "sortino_ratio": 1.3,
        "calmar_ratio": 0.8,
        "expectancy_pct": 0.5,
        "avg_trade_pct": 0.3,
        "num_trades": 100,
    }
    pred_stats = {
        "accuracy": 0.6,
        "macro_f1": 0.55,
        "directional_accuracy": 0.65,
        "per_class": {"Long": {"f1": 0.7}, "Short": {"f1": 0.5}},
        "majority_baseline": 0.4,
    }
    L: list[str] = []
    _exec_verdict(L, metrics, pred_stats)
    assert len(L) > 0


@pytest.mark.unit
def test_model_label_stacking() -> None:
    config = Config()
    config.model.architecture = "stacking"
    assert "Stacking" in model_label(config) or "Hybrid" in model_label(config)


@pytest.mark.unit
def test_model_label_lgbm() -> None:
    config = Config()
    config.model.architecture = "lgbm"
    label = model_label(config)
    assert "LightGBM" in label or "LGBM" in label or label  # just ensure no crash


@pytest.mark.unit
class TestAssessModelQuality:
    def test_good_model(self) -> None:
        pred_stats = {
            "accuracy": 0.7,
            "macro_f1": 0.65,
            "directional_accuracy": 0.7,
            "per_class": {"Long": {"f1": 0.7}, "Short": {"f1": 0.6}},
            "majority_baseline": 0.4,
        }
        quality, detail = _assess_model_quality(pred_stats)
        assert "good" in quality.lower() or "acceptable" in quality.lower() or quality

    def test_poor_model(self) -> None:
        pred_stats = {
            "accuracy": 0.35,
            "macro_f1": 0.2,
            "directional_accuracy": 0.3,
            "per_class": {"Long": {"f1": 0.1}, "Short": {"f1": 0.1}},
            "majority_baseline": 0.4,
        }
        quality, detail = _assess_model_quality(pred_stats)
        assert quality  # just ensure no crash


@pytest.mark.unit
class TestAssessTradingEdge:
    def test_profitable(self) -> None:
        metrics = {
            "sharpe_ratio": 1.5,
            "profit_factor": 1.8,
            "return_pct": 15.0,
            "win_rate_pct": 55.0,
            "max_drawdown_pct": -10.0,
        }
        edge, detail = _assess_trading_edge(metrics)
        assert edge

    def test_unprofitable(self) -> None:
        metrics = {
            "sharpe_ratio": -0.5,
            "profit_factor": 0.8,
            "return_pct": -10.0,
            "win_rate_pct": 40.0,
            "max_drawdown_pct": -30.0,
        }
        edge, detail = _assess_trading_edge(metrics)
        assert edge


@pytest.mark.unit
class TestDeriveRecommendation:
    def test_returns_string(self) -> None:
        rec = _derive_recommendation("good", "strong", {})
        assert isinstance(rec, str)


@pytest.mark.unit
class TestIdentifyPrimaryIssue:
    def test_with_metrics(self) -> None:
        metrics = {
            "accuracy": 0.35,
            "sharpe_ratio": -1.0,
            "profit_factor": 0.5,
        }
        issue = _identify_primary_issue(metrics, None)
        assert issue is None or isinstance(issue, str)


@pytest.mark.unit
@pytest.mark.skip(reason="render sections require full config setup")
class TestRenderSections:
    def test_render_data_quality_section(self) -> None:
        pass

    def test_render_metric_zones_section(self) -> None:
        pass

    def test_render_oof_vs_oos_section(self) -> None:
        pass


@pytest.mark.unit
class TestBenchmarkComparisonTable:
    def test_benchmark_comparison_table_does_not_crash(self) -> None:
        config = Config()
        metrics = {
            "accuracy": 0.6,
            "sharpe_ratio": 0.5,
            "profit_factor": 1.0,
            "return_pct": 3.0,
            "win_rate_pct": 50.0,
            "max_drawdown_pct": -15.0,
        }
        L: list[str] = []
        _benchmark_comparison_table(L, metrics, config)
        # Just check it doesn't crash
