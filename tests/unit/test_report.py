"""Tests for report rendering helpers."""

from __future__ import annotations


import pytest

from thesis.config import Config
from thesis.report import (
    _assess_model_quality,
    _assess_trading_edge,
    _benchmark_comparison_table,
    _config_table,
    _derive_recommendation,
    _exec_verdict,
    _identify_primary_issue,
    _model_label,
)


@pytest.mark.unit
def test_config_table_shows_walk_forward_for_sliding_validation() -> None:
    """Sliding validation should not render stale static split ranges."""
    cfg = Config()
    cfg.validation.method = "sliding"
    lines: list[str] = []

    _config_table(lines, cfg)
    rendered = "\n".join(lines)

    assert "bar-based walk-forward" in rendered
    assert "train/test/step bars" in rendered
    assert cfg.splitting.train_start not in rendered
    assert cfg.splitting.test_end not in rendered


@pytest.mark.unit
def test_config_table_shows_static_ranges_for_static_validation() -> None:
    """Static validation should keep explicit train/val/test ranges."""
    cfg = Config()
    cfg.validation.method = "static"
    lines: list[str] = []

    _config_table(lines, cfg)
    rendered = "\n".join(lines)

    assert cfg.splitting.train_start in rendered
    assert cfg.splitting.val_start in rendered
    assert cfg.splitting.test_start in rendered
    assert "bar-based walk-forward" not in rendered


@pytest.mark.unit
def test_model_label_matches_architecture() -> None:
    """Report title should reflect the configured architecture."""
    cfg = Config()

    cfg.model.architecture = "static"
    assert _model_label(cfg) == "Static LightGBM"

    cfg.model.architecture = "hybrid"
    assert _model_label(cfg) == "Hybrid GRU + LightGBM"


@pytest.mark.unit
def test_benchmark_table_discloses_not_cost_equivalent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Benchmark section should state benchmark cost assumptions explicitly."""
    cfg = Config()
    metrics = {"return_pct": 1.0, "sharpe_ratio": 0.5, "max_drawdown_pct": -1.0}

    def fake_benchmarks(_test_path, _metrics, _config):
        return [
            {
                "strategy": "Hybrid GRU+LGBM",
                "return_pct": 1.0,
                "sharpe": 0.5,
                "max_dd_pct": 1.0,
                "win_rate_pct": 50.0,
                "num_trades": 1,
            }
        ]

    monkeypatch.setattr("thesis.report.compute_benchmark_comparison", fake_benchmarks)
    lines: list[str] = []

    _benchmark_comparison_table(lines, metrics, cfg)
    rendered = "\n".join(lines)

    assert "not trading-cost-equivalent" in rendered
    assert "Hybrid GRU + LightGBM model" in rendered


# ---------------------------------------------------------------------------
# Executive verdict tests (zone-based recommendation)
# ---------------------------------------------------------------------------


class TestAssessModelQuality:
    """Unit tests for _assess_model_quality."""

    def test_poor_acc_below_baseline(self) -> None:
        ps = {
            "accuracy": 0.50,
            "majority_baseline": 0.55,
            "directional_accuracy": 0.48,
            "per_class": {
                "Short": {"f1": 0.3},
                "Hold": {"f1": 0.3},
                "Long": {"f1": 0.3},
            },
        }
        quality, reason = _assess_model_quality(ps)
        assert quality == "POOR"
        assert "below" in reason.lower()

    def test_good_above_baseline_with_edge(self) -> None:
        ps = {
            "accuracy": 0.62,
            "majority_baseline": 0.50,
            "directional_accuracy": 0.58,
            "per_class": {
                "Short": {"f1": 0.5},
                "Hold": {"f1": 0.4},
                "Long": {"f1": 0.5},
            },
        }
        quality, reason = _assess_model_quality(ps)
        assert quality == "GOOD"
        assert "directional edge" in reason.lower()

    def test_fair_marginal_edge(self) -> None:
        ps = {
            "accuracy": 0.52,
            "majority_baseline": 0.50,
            "directional_accuracy": 0.51,
            "per_class": {
                "Short": {"f1": 0.3},
                "Hold": {"f1": 0.3},
                "Long": {"f1": 0.3},
            },
        }
        quality, reason = _assess_model_quality(ps)
        assert quality == "FAIR"


class TestAssessTradingEdge:
    """Unit tests for _assess_trading_edge."""

    def test_negative_profit_factor_below_one(self) -> None:
        edge, reason = _assess_trading_edge(
            {"profit_factor": 0.8, "sharpe_ratio": 0.5, "return_pct": 10}
        )
        assert edge == "NEGATIVE"

    def test_negative_sharpe(self) -> None:
        edge, reason = _assess_trading_edge(
            {"profit_factor": 1.5, "sharpe_ratio": -0.2, "return_pct": 5}
        )
        assert edge == "NEGATIVE"

    def test_marginal_low_sharpe(self) -> None:
        edge, reason = _assess_trading_edge(
            {"profit_factor": 1.3, "sharpe_ratio": 0.7, "return_pct": 5}
        )
        assert edge == "MARGINAL"

    def test_positive(self) -> None:
        edge, reason = _assess_trading_edge(
            {"profit_factor": 2.0, "sharpe_ratio": 2.0, "return_pct": 15}
        )
        assert edge == "POSITIVE"


class TestDeriveRecommendation:
    """Unit tests for _derive_recommendation."""

    def test_not_deployable_poor_model(self) -> None:
        rec = _derive_recommendation(
            "POOR", "POSITIVE", {"num_trades": 100, "return_pct": 10}
        )
        assert "NOT DEPLOYABLE" in rec

    def test_not_deployable_negative_edge(self) -> None:
        rec = _derive_recommendation(
            "GOOD", "NEGATIVE", {"num_trades": 100, "return_pct": 10}
        )
        assert "NOT DEPLOYABLE" in rec

    def test_dep_insufficient_trades(self) -> None:
        rec = _derive_recommendation(
            "GOOD", "POSITIVE", {"num_trades": 10, "return_pct": 10}
        )
        assert "NOT DEPLOYABLE" in rec and "insufficient" in rec.lower()

    def test_deployable_with_caution(self) -> None:
        rec = _derive_recommendation(
            "FAIR", "MARGINAL", {"num_trades": 100, "return_pct": 10}
        )
        assert "caution" in rec.lower()

    def test_deployable(self) -> None:
        rec = _derive_recommendation(
            "GOOD", "POSITIVE", {"num_trades": 150, "return_pct": 10}
        )
        assert rec == "DEPLOYABLE"


class TestIdentifyPrimaryIssue:
    """Unit tests for _identify_primary_issue."""

    def test_zero_trades(self) -> None:
        result = _identify_primary_issue(
            {
                "num_trades": 0,
                "sharpe_ratio": 0,
                "profit_factor": 0,
                "max_drawdown_pct": 0,
                "return_pct": 0,
                "win_rate_pct": 0,
            },
            None,
        )
        assert result is not None
        assert "Zero trades" in result

    def test_negative_sharpe(self) -> None:
        result = _identify_primary_issue(
            {
                "num_trades": 50,
                "sharpe_ratio": -0.3,
                "profit_factor": 0.9,
                "max_drawdown_pct": 10,
                "return_pct": -5,
                "win_rate_pct": 30,
            },
            None,
        )
        assert result is not None
        assert "negative" in result.lower()

    def test_drawdown_catastrophic(self) -> None:
        result = _identify_primary_issue(
            {
                "num_trades": 80,
                "sharpe_ratio": 0.6,
                "profit_factor": 1.1,
                "max_drawdown_pct": 55,
                "return_pct": 5,
                "win_rate_pct": 45,
            },
            None,
        )
        assert result is not None
        assert "catastrophic" in result.lower()

    def test_none_when_all_ok(self) -> None:
        result = _identify_primary_issue(
            {
                "num_trades": 200,
                "sharpe_ratio": 2.0,
                "profit_factor": 2.5,
                "max_drawdown_pct": 10,
                "return_pct": 30,
                "win_rate_pct": 55,
            },
            {"directional_accuracy": 0.60},
        )
        assert result is None


class TestExecVerdict:
    """Integration tests for the extended _exec_verdict function."""

    def _make_pred_stats(self, **overrides) -> dict:
        base = {
            "accuracy": 0.52,
            "majority_baseline": 0.50,
            "directional_accuracy": 0.51,
            "per_class": {
                "Short": {"f1": 0.3},
                "Hold": {"f1": 0.3},
                "Long": {"f1": 0.3},
            },
        }
        base.update(overrides)
        return base

    def test_verdict_line_present_with_metrics(self) -> None:
        """Verdict line must appear when both pred_stats and metrics exist."""
        L: list[str] = []
        metrics = {
            "profit_factor": 1.4,
            "sharpe_ratio": 0.8,
            "return_pct": -2,
            "num_trades": 80,
            "max_drawdown_pct": 15,
            "win_rate_pct": 40,
        }
        _exec_verdict(L, metrics, self._make_pred_stats())
        rendered = "\n".join(L)
        assert "**Verdict:**" in rendered
        assert "Primary issue" in rendered

    def test_no_metrics_still_shows_model_quality(self) -> None:
        """Verdict line shows model quality even without backtest metrics."""
        L: list[str] = []
        _exec_verdict(L, {}, self._make_pred_stats())
        rendered = "\n".join(L)
        assert "**Verdict:**" in rendered
        assert "no backtest metrics available" in rendered.lower()
        assert "Primary issue" in rendered

    def test_no_pred_stats_no_metrics(self) -> None:
        """Returns early with no output lines."""
        L: list[str] = []
        _exec_verdict(L, {}, None)
        # Should have no lines appended
        assert len(L) == 0

    def test_no_pred_stats_with_metrics(self) -> None:
        """Fallback message when only the demo ran."""
        L: list[str] = []
        _exec_verdict(L, {"return_pct": 5, "num_trades": 10}, None)
        rendered = "\n".join(L)
        assert "unavailable" in rendered.lower()
