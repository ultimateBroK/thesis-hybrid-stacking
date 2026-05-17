"""Tests for backtest module — backtesting.py integration.

NOTE: run_backtest_from_data / run_backtest_manual removed in refactor.
All tests skipped pending API rewrite.
"""

import pytest

pytestmark = pytest.mark.skip(
    reason="run_backtest_from_data / run_backtest_manual removed in refactor"
)
