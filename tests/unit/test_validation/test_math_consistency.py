"""Unit tests for math consistency validation module.

Tests for trading math validation, Kelly criterion calculation,
Calmar ratio validation, and statistical bounds checking.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


class TestTradingMathValidator:
    """Tests for the TradingMathValidator dataclass."""

    def test_expected_return_calculation(self):
        """Expected return should calculate correctly."""
        from thesis.validation.math_consistency import TradingMathValidator

        validator = TradingMathValidator(
            accuracy=0.55,
            win_rate=0.55,
            avg_win=0.02,  # 2% avg win
            avg_loss=0.01,  # 1% avg loss
            num_trades=100,
            risk_per_trade=0.01,
        )

        expected = validator.expected_return()
        # EV per trade = 0.55 * 0.02 - 0.45 * 0.01 = 0.011 - 0.0045 = 0.0065
        # Total = 0.0065 * 100 = 0.65 (65%)
        assert expected == pytest.approx(0.65, abs=0.001)

    def test_expected_return_zero_trades(self):
        """Expected return with zero trades should be zero."""
        from thesis.validation.math_consistency import TradingMathValidator

        validator = TradingMathValidator(
            accuracy=0.55,
            win_rate=0.55,
            avg_win=0.02,
            avg_loss=0.01,
            num_trades=0,
            risk_per_trade=0.01,
        )

        expected = validator.expected_return()
        assert expected == 0.0

    def test_kelly_fraction_positive_edge(self):
        """Kelly fraction should calculate for positive edge."""
        from thesis.validation.math_consistency import TradingMathValidator

        validator = TradingMathValidator(
            accuracy=0.60,
            win_rate=0.60,
            avg_win=0.03,
            avg_loss=0.01,
            num_trades=100,
            risk_per_trade=0.01,
        )

        kelly = validator.kelly_fraction()
        # Edge = 0.6 * 0.03 - 0.4 * 0.01 = 0.018 - 0.004 = 0.014
        # Kelly = 0.014 / 0.03 = 0.466...
        assert kelly > 0.0
        assert kelly < 1.0

    def test_kelly_fraction_negative_edge(self):
        """Kelly fraction should be zero for negative edge."""
        from thesis.validation.math_consistency import TradingMathValidator

        validator = TradingMathValidator(
            accuracy=0.40,
            win_rate=0.40,
            avg_win=0.01,
            avg_loss=0.03,
            num_trades=100,
            risk_per_trade=0.01,
        )

        kelly = validator.kelly_fraction()
        assert kelly == 0.0

    def test_kelly_fraction_zero_avg_win(self):
        """Kelly fraction should handle zero avg_win gracefully."""
        from thesis.validation.math_consistency import TradingMathValidator

        validator = TradingMathValidator(
            accuracy=0.50,
            win_rate=0.50,
            avg_win=0.0,
            avg_loss=0.01,
            num_trades=100,
            risk_per_trade=0.01,
        )

        kelly = validator.kelly_fraction()
        assert kelly == 0.0


class TestValidateConsistency:
    """Tests for the validate_consistency method."""

    def test_consistency_within_bounds(self):
        """Actual return within bounds should pass."""
        from thesis.validation.math_consistency import TradingMathValidator

        validator = TradingMathValidator(
            accuracy=0.50,
            win_rate=0.50,
            avg_win=0.02,
            avg_loss=0.02,
            num_trades=100,
            risk_per_trade=0.01,
        )

        # Expected is 0, actual close to 0
        results = validator.validate_consistency(0.01)

        assert results["within_1_sigma"] == True
        assert results["red_flag"] == False

    def test_consistency_red_flag_outside_6_sigma(self):
        """Returns beyond 6 sigma should trigger red flag."""
        from thesis.validation.math_consistency import TradingMathValidator

        validator = TradingMathValidator(
            accuracy=0.55,
            win_rate=0.55,
            avg_win=0.02,
            avg_loss=0.02,
            num_trades=100,
            risk_per_trade=0.01,
        )

        # Impossibly high return
        results = validator.validate_consistency(50.0)

        assert results["red_flag"] == True
        assert results["sigma_deviation"] > 6.0

    def test_consistency_warning_outside_3_sigma(self):
        """Returns outside 3-sigma but not 6-sigma should warn."""
        from thesis.validation.math_consistency import TradingMathValidator

        validator = TradingMathValidator(
            accuracy=0.50,
            win_rate=0.50,
            avg_win=0.02,
            avg_loss=0.02,
            num_trades=100,
            risk_per_trade=0.01,
        )

        # Expected return is ~0, std_dev is non-zero. Return of 1.0 should be outside 3-sigma but not trigger red flag
        results = validator.validate_consistency(1.0)

        assert results["warning"] == True
        assert results["within_3_sigma"] == False

    def test_consistency_calculates_sigma_deviation(self):
        """Sigma deviation should be calculated correctly."""
        from thesis.validation.math_consistency import TradingMathValidator

        validator = TradingMathValidator(
            accuracy=0.50,
            win_rate=0.50,
            avg_win=0.02,
            avg_loss=0.02,
            num_trades=100,
            risk_per_trade=0.01,
        )

        results = validator.validate_consistency(0.1)

        assert "sigma_deviation" in results
        assert isinstance(results["sigma_deviation"], float)


class TestValidateCalmar:
    """Tests for the validate_calmar method."""

    def test_calmar_acceptable(self):
        """Calmar below retail benchmark is acceptable."""
        from thesis.validation.math_consistency import TradingMathValidator

        validator = TradingMathValidator(
            accuracy=0.50,
            win_rate=0.50,
            avg_win=0.01,
            avg_loss=0.01,
            num_trades=100,
            risk_per_trade=0.01,
        )

        results = validator.validate_calmar(1.0, 0.10)

        assert results["passed"] is True
        assert results["status"] == "ACCEPTABLE"

    def test_calmar_good(self):
        """Calmar above retail benchmark is good."""
        from thesis.validation.math_consistency import TradingMathValidator

        validator = TradingMathValidator(
            accuracy=0.50,
            win_rate=0.50,
            avg_win=0.01,
            avg_loss=0.01,
            num_trades=100,
            risk_per_trade=0.01,
        )

        results = validator.validate_calmar(2.0, 0.05)

        assert results["passed"] is True
        assert results["status"] == "GOOD"

    def test_calmar_exceptional(self):
        """Calmar above professional benchmark is exceptional with warning."""
        from thesis.validation.math_consistency import TradingMathValidator

        validator = TradingMathValidator(
            accuracy=0.50,
            win_rate=0.50,
            avg_win=0.01,
            avg_loss=0.01,
            num_trades=100,
            risk_per_trade=0.01,
        )

        results = validator.validate_calmar(4.0, 0.02)

        assert results["passed"] is True
        assert results["status"] == "EXCEPTIONAL"
        assert results["warning"] is True

    def test_calmar_impossible(self):
        """Calmar above impossible threshold should fail."""
        from thesis.validation.math_consistency import TradingMathValidator

        validator = TradingMathValidator(
            accuracy=0.50,
            win_rate=0.50,
            avg_win=0.01,
            avg_loss=0.01,
            num_trades=100,
            risk_per_trade=0.01,
        )

        results = validator.validate_calmar(15.0, 0.05)

        assert results["passed"] is False
        assert results["status"] == "IMPOSSIBLE"


class TestValidateBacktestMath:
    """Tests for the validate_backtest_math function."""

    def test_valid_backtest_passes(self):
        """Valid backtest results should pass."""
        from thesis.validation.math_consistency import validate_backtest_math

        backtest_results = {
            "metrics": {
                "total_return_pct": 50.0,
                "win_rate": 0.55,
                "num_trades": 100,
                "calmar_ratio": 2.0,
                "max_drawdown_pct": 25.0,
                "avg_winner": 100,
                "avg_loser": 80,
            }
        }

        results = validate_backtest_math(backtest_results)

        assert results["passed"] is True
        assert "return_validation" in results
        assert "calmar_validation" in results

    def test_impossible_return_fails(self):
        """Impossible return should fail validation."""
        from thesis.validation.math_consistency import validate_backtest_math

        # 2000% return with 50% win rate is impossible
        backtest_results = {
            "metrics": {
                "total_return_pct": 2000.0,
                "win_rate": 0.50,
                "num_trades": 100,
                "calmar_ratio": 1.0,
                "max_drawdown_pct": 20.0,
            }
        }

        results = validate_backtest_math(backtest_results)

        assert results["passed"] is False

    def test_with_trades_list(self):
        """Validation should work with per-trade data."""
        from thesis.validation.math_consistency import validate_backtest_math

        backtest_results = {
            "trades": [
                {"pnl": 100, "entry_price": 1500},
                {"pnl": -50, "entry_price": 1500},
                {"pnl": 120, "entry_price": 1510},
                {"pnl": -40, "entry_price": 1510},
            ],
            "metrics": {
                "total_return_pct": 10.0,
                "win_rate": 0.50,
                "total_trades": 4,
                "calmar_ratio": 1.0,
                "max_drawdown_pct": 10.0,
            }
        }

        results = validate_backtest_math(backtest_results)

        assert "return_validation" in results
        assert "calmar_validation" in results

    def test_backtest_with_warnings_logged(self):
        """Warnings should be collected in results."""
        from thesis.validation.math_consistency import validate_backtest_math

        # High return that triggers warning
        backtest_results = {
            "metrics": {
                "total_return_pct": 500.0,
                "win_rate": 0.52,
                "num_trades": 50,
                "calmar_ratio": 8.0,
                "max_drawdown_pct": 5.0,
            }
        }

        results = validate_backtest_math(backtest_results)

        assert "warnings" in results
        assert len(results["warnings"]) > 0


class TestRunMathValidationFromFiles:
    """Tests for the run_math_validation_from_files function."""

    @patch("pathlib.Path.exists", return_value=True)
    @patch("builtins.open")
    @patch("json.load")
    def test_validation_from_file(self, mock_json_load, mock_open, mock_exists):
        """Should validate from JSON file."""
        from thesis.validation.math_consistency import run_math_validation_from_files

        mock_json_load.return_value = {
            "metrics": {
                "total_return_pct": 20.0,
                "win_rate": 0.55,
                "num_trades": 100,
                "calmar_ratio": 1.5,
                "max_drawdown_pct": 15.0,
            }
        }

        with patch("thesis.validation.math_consistency.validate_backtest_math") as mock_validate:
            mock_validate.return_value = {
                "return_validation": {},
                "calmar_validation": {"passed": True},
                "kelly_fraction": 0.1,
                "expected_return": 0.2,
                "passed": True,
                "warnings": [],
            }
            results = run_math_validation_from_files("results/backtest.json")

        assert results["passed"] is True

    @patch("pathlib.Path.exists", return_value=False)
    def test_file_not_found_returns_error(self, mock_exists):
        """Should return error if file not found."""
        from thesis.validation.math_consistency import run_math_validation_from_files

        results = run_math_validation_from_files("results/nonexistent.json")

        assert "error" in results
        assert results["passed"] is False

    @patch("pathlib.Path.exists", return_value=True)
    @patch("builtins.open")
    @patch("json.load")
    def test_validation_failure_raises(self, mock_json_load, mock_open, mock_exists):
        """Should raise ValueError on validation failure."""
        from thesis.validation.math_consistency import run_math_validation_from_files

        mock_json_load.return_value = {
            "metrics": {
                "total_return_pct": 2000.0,  # Impossible
                "win_rate": 0.50,
                "num_trades": 100,
                "calmar_ratio": 20.0,  # Impossible
                "max_drawdown_pct": 5.0,
            }
        }

        with pytest.raises(ValueError, match="Math consistency check failed"):
            run_math_validation_from_files("results/backtest.json")


class TestEdgeCases:
    """Edge case tests for math consistency."""

    def test_zero_avg_loss(self):
        """Zero avg_loss should not cause division by zero in Kelly."""
        from thesis.validation.math_consistency import TradingMathValidator

        validator = TradingMathValidator(
            accuracy=0.50,
            win_rate=0.50,
            avg_win=0.02,
            avg_loss=0.0,
            num_trades=100,
            risk_per_trade=0.01,
        )

        kelly = validator.kelly_fraction()
        assert kelly == 0.0  # Should handle gracefully

    def test_very_high_num_trades(self):
        """High number of trades should not cause overflow."""
        from thesis.validation.math_consistency import TradingMathValidator

        validator = TradingMathValidator(
            accuracy=0.51,
            win_rate=0.51,
            avg_win=0.001,
            avg_loss=0.001,
            num_trades=1000000,
            risk_per_trade=0.01,
        )

        expected = validator.expected_return()
        assert expected < 1000000  # Should be reasonable
        assert expected > 0

    def test_extreme_win_rate(self):
        """Near-perfect win rate should be handled."""
        from thesis.validation.math_consistency import TradingMathValidator

        validator = TradingMathValidator(
            accuracy=0.99,
            win_rate=0.99,
            avg_win=0.01,
            avg_loss=0.02,
            num_trades=100,
            risk_per_trade=0.01,
        )

        expected = validator.expected_return()
        assert expected > 0  # Should be positive with 99% win rate

    def test_zero_trades_variance_handling(self):
        """Zero trades should handle variance calculation gracefully."""
        from thesis.validation.math_consistency import TradingMathValidator

        validator = TradingMathValidator(
            accuracy=0.50,
            win_rate=0.50,
            avg_win=0.01,
            avg_loss=0.01,
            num_trades=0,
            risk_per_trade=0.01,
        )

        # Should not raise error
        results = validator.validate_consistency(0.0)
        assert results["std_dev"] == 1e-10  # Default small value
