"""Tests for pipeline module — OOF guards and validation checks."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from thesis.config import Config


# ---------------------------------------------------------------------------
# Purge guard — pipeline raises ValueError when gap < sequence_length
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_purge_guard_raises_on_insufficient_gap() -> None:
    """Pipeline must reject purge+embargo < sequence_length."""
    config = Config()
    # Set gap smaller than sequence_length to trigger the guard
    config.validation.purge_bars = 5
    config.validation.embargo_bars = 10  # gap = 15
    config.gru.sequence_length = 48  # gap < seq_len → should raise

    # The guard lives inside _run_walk_forward_hybrid which reads parquet.
    # We mock the file I/O and generate_windows to reach the guard.
    mock_df = MagicMock()
    mock_df.__len__ = MagicMock(return_value=100_000)
    mock_windows = [MagicMock()]  # non-empty to pass the first check

    with (
        patch("thesis.pipeline.pl.read_parquet", return_value=mock_df),
        patch("thesis.pipeline.generate_windows", return_value=mock_windows),
        patch("thesis.pipeline.log_windows"),
        pytest.raises(ValueError, match="Leakage risk"),
    ):
        from thesis.pipeline import _run_walk_forward_hybrid

        _run_walk_forward_hybrid(config)


@pytest.mark.unit
def test_purge_guard_passes_with_sufficient_gap() -> None:
    """Pipeline should NOT raise when gap >= sequence_length."""
    config = Config()
    config.validation.purge_bars = 25
    config.validation.embargo_bars = 50  # gap = 75
    config.gru.sequence_length = 48  # gap >= seq_len → OK

    mock_df = MagicMock()
    mock_df.__len__ = MagicMock(return_value=100_000)
    # Empty columns list for feature discovery
    mock_df.columns = []
    mock_windows = [MagicMock()]

    with (
        patch("thesis.pipeline.pl.read_parquet", return_value=mock_df),
        patch("thesis.pipeline.generate_windows", return_value=mock_windows),
        patch("thesis.pipeline.log_windows"),
    ):
        from thesis.pipeline import _run_walk_forward_hybrid

        # Should NOT raise the purge guard ValueError.
        # It will fail later (no real data), but the guard is what we test.
        try:
            _run_walk_forward_hybrid(config)
        except ValueError as e:
            # Must NOT be the leakage guard error
            assert "Leakage risk" not in str(e)
        except (RuntimeError, FileNotFoundError, AttributeError):
            # Expected — no real data/model artifacts
            pass
