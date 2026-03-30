"""Tests for thesis report generation."""

from types import SimpleNamespace

from thesis.reporting.thesis_report import _create_markdown_report


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
        reporting=SimpleNamespace(shap_summary_path="results/shap_summary.png"),
        paths=SimpleNamespace(final_predictions="data/predictions/final_predictions.parquet"),
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
