"""Tests for thesis report generation."""

import tempfile
from pathlib import Path
from types import SimpleNamespace

from thesis.reporting.thesis_report import (
    _create_markdown_report,
    _generate_interactive_equity_curve,
    _generate_interactive_trades,
)


def _build_plotly_config():
    return SimpleNamespace(
        enabled=True,
        equity_curve_path="interactive_equity.html",
        confidence_path="interactive_confidence.html",
        trades_path="interactive_trades.html",
        include_annotations=True,
    )


def _build_config():
    return SimpleNamespace(
        splitting=SimpleNamespace(
            train_start="2018-01-01",
            train_end="2022-12-31",
            val_start="2023-01-01",
            test_start="2024-01-01",
            test_end="2026-03-30",
            purge_bars=15,
            embargo_bars=10,
        ),
        models={
            "tree": SimpleNamespace(optuna_trials=100, use_class_weights=True),
            "lstm": SimpleNamespace(
                sequence_length=120,
                hidden_size=128,
                num_layers=2,
                dropout=0.3,
                model_type="gru",
            ),
            "stacking": SimpleNamespace(
                meta_learner="logistic_regression",
                calibrate_probabilities=True,
            ),
        },
        backtest=SimpleNamespace(
            initial_capital=100000.0,
            leverage=100,
            spread_pips=2.0,
            risk_per_trade=0.01,
            slippage_pips=1.0,
            backtest_results_path="results/backtest_results.json",
        ),
        reporting=SimpleNamespace(
            shap_summary_path="results/shap_summary.png",
            report_path="results/thesis_report.md",
            plotly=_build_plotly_config(),
        ),
        paths=SimpleNamespace(
            final_predictions="data/predictions/final_predictions.parquet",
            session_path="",
        ),
    )


def test_report_conclusion_uses_metrics():
    """Report conclusions should render metrics instead of placeholders."""
    backtest_data = {
        "metrics": {
            "total_trades": 1122,
            "winning_trades": 740,
            "losing_trades": 382,
            "win_rate": 0.6595,
            "profit_factor": 1.7224,
            "total_return_pct": 1067.4083,
            "avg_trade_dollar": 951.3443,
            "avg_win_dollar": 3439.1208,
            "avg_loss_dollar": -3867.9085,
            "sharpe_ratio": 2.9913,
            "max_drawdown_pct": 22.1321,
            "calmar_ratio": 48.2290,
            "final_capital": 1167408.3320,
            "total_pnl_pips": 582408.15,
            "avg_pips_per_trade": 519.0803,
        }
    }

    report = _create_markdown_report(backtest_data, _build_config())

    assert "[model effectiveness on test set]" not in report
    assert "1122 trades" in report
    assert "66.0% win rate" in report
    assert "1067.41% total return" in report


def test_report_shows_gru_model_type():
    """Report should show GRU when model_type is gru."""
    backtest_data = {
        "metrics": {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "total_return_pct": 0,
            "sharpe_ratio": 0,
            "max_drawdown_pct": 0,
            "calmar_ratio": 0,
            "avg_trade_dollar": 0,
            "avg_win_dollar": 0,
            "avg_loss_dollar": 0,
            "total_pnl_pips": 0,
            "avg_pips_per_trade": 0,
            "final_capital": 100000.0,
        }
    }
    report = _create_markdown_report(backtest_data, _build_config())

    assert "GRU" in report
    assert "Hybrid Stacking (GRU + LightGBM)" in report


def test_interactive_equity_curve_with_empty_data():
    """Interactive equity curve should return None with no data."""
    config = _build_config()
    result = _generate_interactive_equity_curve({"equity_curve": []}, config)
    assert result is None


def test_interactive_equity_curve_with_data():
    """Interactive equity curve should generate HTML file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _build_config()
        config.reporting.report_path = str(Path(tmpdir) / "thesis_report.md")
        backtest_data = {"equity_curve": [100000, 101000, 102000, 101500, 103000]}

        result = _generate_interactive_equity_curve(backtest_data, config)

        assert result is not None
        assert result.suffix == ".html"
        assert result.exists()


def test_interactive_trades_with_empty_data():
    """Interactive trades chart should return None with no data."""
    config = _build_config()
    result = _generate_interactive_trades({"trades": []}, config)
    assert result is None


def test_interactive_trades_with_data():
    """Interactive trades chart should generate HTML file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _build_config()
        config.reporting.report_path = str(Path(tmpdir) / "thesis_report.md")
        backtest_data = {
            "trades": [
                {"pnl": 500.0},
                {"pnl": -200.0},
                {"pnl": 300.0},
            ]
        }

        result = _generate_interactive_trades(backtest_data, config)

        assert result is not None
        assert result.suffix == ".html"
        assert result.exists()
