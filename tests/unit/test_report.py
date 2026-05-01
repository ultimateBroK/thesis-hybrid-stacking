"""Tests for report rendering helpers."""

from __future__ import annotations

import pytest

from thesis.config import Config
from thesis.constants import H1_BARS_PER_YEAR
from thesis.model import _H1_BARS_PER_YEAR
from thesis.report import _benchmark_comparison_table, _config_table


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


@pytest.mark.unit
def test_h1_bars_per_year_is_shared_constant() -> None:
    """Report/model annualization should use one shared H1 constant."""
    assert H1_BARS_PER_YEAR == 24 * 5 * 52
    assert _H1_BARS_PER_YEAR == H1_BARS_PER_YEAR
