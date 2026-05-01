"""Tests for labels module.

Tests triple-barrier labeling logic directly.
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from thesis.labels import _compute_labels, _merge_label_columns


@pytest.mark.unit
@pytest.mark.data
def test_labels_in_valid_set() -> None:
    """Test that labels are in {-1, 0, 1}."""
    n = 50
    close = np.linspace(1800, 1900, n)
    high = close + 5
    low = close - 5
    atr = np.ones(n) * 10

    labels, _, _, _, _ = _compute_labels(
        close=close,
        high=high,
        low=low,
        atr=atr,
        tp_mult=1.5,
        sl_mult=1.5,
        horizon=10,
        min_atr=0.0001,
    )

    unique_labels = np.unique(labels)

    # All labels should be in {-1, 0, 1}; -2 may appear for right-censored rows
    # (last `horizon` bars with insufficient forward data to evaluate)
    valid = np.all(np.isin(unique_labels, [-2, -1, 0, 1]))
    assert valid, (
        f"Unexpected labels {unique_labels}: expected subset of {{-2, -1, 0, 1}}"
    )


@pytest.mark.unit
@pytest.mark.data
def test_upper_lower_barrier_relationship() -> None:
    """Test that upper_barrier > close and lower_barrier < close for each bar (the bug fix!)."""
    n = 50
    close = np.linspace(1800, 1900, n)
    high = close + 5
    low = close - 5
    atr = np.ones(n) * 10

    _, upper_barriers, lower_barriers, _, _ = _compute_labels(
        close=close,
        high=high,
        low=low,
        atr=atr,
        tp_mult=1.5,
        sl_mult=1.5,
        horizon=10,
        min_atr=0.0001,
    )

    # For every bar, upper barrier should be above close and lower barrier below close
    for i in range(n):
        assert upper_barriers[i] > close[i], (
            f"upper barrier {upper_barriers[i]} not > close {close[i]} at index {i}"
        )
        assert lower_barriers[i] < close[i], (
            f"lower barrier {lower_barriers[i]} not < close {close[i]} at index {i}"
        )


@pytest.mark.unit
@pytest.mark.data
def test_touched_bars_for_hold() -> None:
    """Test that touched_bars is -1 for Hold labels."""
    n = 50
    close = np.linspace(1800, 1900, n)
    # Make high/low very close to close so barriers are never hit
    high = close + 0.1
    low = close - 0.1
    atr = np.ones(n) * 10  # Barriers will be at +/- 15

    labels, _, _, touched_bars, _ = _compute_labels(
        close=close,
        high=high,
        low=low,
        atr=atr,
        tp_mult=1.5,
        sl_mult=1.5,
        horizon=10,
        min_atr=0.0001,
    )

    # For Hold labels (0), touched_bar should be -1
    for i in range(n):
        if labels[i] == 0:
            assert touched_bars[i] == -1, (
                f"Hold label at {i} should have touched_bar=-1"
            )


@pytest.mark.unit
@pytest.mark.data
def test_same_bar_both_hit_counted_as_ambiguous_hold() -> None:
    """Same-bar upper/lower hit is neutral and counted for diagnostics."""
    close = np.array([100.0, 100.0, 100.0, 100.0])
    high = np.array([100.0, 103.0, 100.0, 100.0])
    low = np.array([100.0, 97.0, 100.0, 100.0])
    atr = np.ones(len(close))

    labels, _, _, touched_bars, ambiguous_count = _compute_labels(
        close=close,
        high=high,
        low=low,
        atr=atr,
        tp_mult=2.0,
        sl_mult=2.0,
        horizon=2,
        min_atr=0.0001,
    )

    assert labels[0] == 0
    assert touched_bars[0] == -1
    assert ambiguous_count == 1


@pytest.mark.unit
@pytest.mark.data
def test_label_columns_do_not_emit_legacy_tp_sl_aliases() -> None:
    """New label output uses upper/lower barrier names only."""
    df = pl.DataFrame({"timestamp": [1, 2, 3]})
    result = _merge_label_columns(
        df,
        labels_arr=np.array([1, 0, -1], dtype=np.int32),
        upper_arr=np.array([102.0, 103.0, 104.0]),
        lower_arr=np.array([98.0, 97.0, 96.0]),
        touched_bars_arr=np.array([1, -1, 2], dtype=np.int32),
    )

    assert "upper_barrier" in result.columns
    assert "lower_barrier" in result.columns
    assert "tp_price" not in result.columns
    assert "sl_price" not in result.columns


@pytest.mark.unit
@pytest.mark.data
def test_touched_bars_for_non_hold() -> None:
    """Test that touched_bars >= 0 for non-Hold labels."""
    n = 50
    close = np.linspace(1800, 1900, n)
    # Make high hit upper barrier quickly
    high = close + 20  # Will hit upper barrier
    low = close - 1
    atr = np.ones(n) * 10

    labels, _, _, touched_bars, _ = _compute_labels(
        close=close,
        high=high,
        low=low,
        atr=atr,
        tp_mult=1.5,
        sl_mult=1.5,
        horizon=10,
        min_atr=0.0001,
    )

    # For non-Hold labels, touched_bar should be >= 0
    for i in range(n - 10):  # Last 'horizon' bars may not have enough future
        if labels[i] != 0:
            assert touched_bars[i] >= 0, (
                f"Non-Hold label at {i} should have touched_bar >= 0"
            )
            assert touched_bars[i] < 10, f"touched_bar at {i} should be < horizon"


@pytest.mark.unit
@pytest.mark.data
def test_zero_atr_handled() -> None:
    """Test with zero ATR (min_atr kicks in)."""
    n = 50
    close = np.linspace(1800, 1900, n)
    high = close + 5
    low = close - 5
    atr = np.zeros(n)  # Zero ATR

    labels, upper_barriers, lower_barriers, _, _ = _compute_labels(
        close=close,
        high=high,
        low=low,
        atr=atr,
        tp_mult=1.5,
        sl_mult=1.5,
        horizon=10,
        min_atr=0.1,  # min_atr should kick in
    )

    # Should still produce valid labels; -2 may appear for right-censored rows
    assert len(labels) == n
    assert np.all(np.isin(labels, [-2, -1, 0, 1]))

    # upper barrier and lower barrier should still be valid
    for i in range(n):
        assert upper_barriers[i] > close[i]
        assert lower_barriers[i] < close[i]


@pytest.mark.unit
@pytest.mark.data
def test_extreme_volatility_all_long() -> None:
    """Test with extreme volatility (all Long)."""
    n = 50
    close = np.linspace(1800, 1900, n)
    # High always hits upper barrier
    high = close + 100
    low = close - 1
    atr = np.ones(n) * 10

    labels, _, _, _, _ = _compute_labels(
        close=close,
        high=high,
        low=low,
        atr=atr,
        tp_mult=1.5,
        sl_mult=1.5,
        horizon=10,
        min_atr=0.0001,
    )

    # Most labels should be Long (1)
    long_count = np.sum(labels == 1)
    assert long_count > 0, "Should have some Long labels"


@pytest.mark.unit
@pytest.mark.data
def test_extreme_volatility_all_short() -> None:
    """Test with extreme volatility (all Short)."""
    n = 50
    close = np.linspace(1800, 1900, n)
    high = close + 1
    # Low always hits lower barrier
    low = close - 100
    atr = np.ones(n) * 10

    labels, _, _, _, _ = _compute_labels(
        close=close,
        high=high,
        low=low,
        atr=atr,
        tp_mult=1.5,
        sl_mult=1.5,
        horizon=10,
        min_atr=0.0001,
    )

    # Most labels should be Short (-1)
    short_count = np.sum(labels == -1)
    assert short_count > 0, "Should have some Short labels"


@pytest.mark.unit
@pytest.mark.data
def test_horizon_boundary() -> None:
    """Test that horizon is respected."""
    n = 50
    close = np.linspace(1800, 1900, n)
    high = close + 5
    low = close - 5
    atr = np.ones(n) * 10
    horizon = 5

    labels, _, _, touched_bars, _ = _compute_labels(
        close=close,
        high=high,
        low=low,
        atr=atr,
        tp_mult=1.5,
        sl_mult=1.5,
        horizon=horizon,
        min_atr=0.0001,
    )

    # For non-Hold labels, touched_bar should be <= horizon
    # (range is i+1 to i+1+horizon, so max touched_bar = horizon)
    for i in range(n):
        if labels[i] != 0 and touched_bars[i] >= 0:
            assert touched_bars[i] <= horizon, (
                f"touched_bar {touched_bars[i]} exceeds horizon {horizon}"
            )


@pytest.mark.unit
@pytest.mark.data
def test_atr_multiplier_effect_asymmetric() -> None:
    """Test that asymmetric TP/SL multipliers create correct barrier widths."""
    n = 50
    close = np.linspace(1800, 1900, n)
    high = close + 5
    low = close - 5
    atr = np.ones(n) * 10

    labels_small, _, _, _, _ = _compute_labels(
        close=close,
        high=high,
        low=low,
        atr=atr,
        tp_mult=1.0,
        sl_mult=1.0,
        horizon=10,
        min_atr=0.0001,
    )

    labels_large, _, _, _, _ = _compute_labels(
        close=close,
        high=high,
        low=low,
        atr=atr,
        tp_mult=3.0,
        sl_mult=3.0,
        horizon=10,
        min_atr=0.0001,
    )

    # Larger multiplier should result in more Hold labels (wider barriers)
    holds_small = np.sum(labels_small == 0)
    holds_large = np.sum(labels_large == 0)

    assert holds_large >= holds_small, (
        "Larger multiplier should produce at least as many Hold labels"
    )


@pytest.mark.unit
@pytest.mark.data
def test_no_lookahead_bias() -> None:
    """Test that labels don't use future information (no lookahead bias)."""
    n = 100
    close = np.linspace(1800, 1900, n)
    # Create a pattern where future is predictable
    high = close + 5
    low = close - 5
    atr = np.ones(n) * 10

    labels, _, _, touched_bars, _ = _compute_labels(
        close=close,
        high=high,
        low=low,
        atr=atr,
        tp_mult=1.5,
        sl_mult=1.5,
        horizon=10,
        min_atr=0.0001,
    )

    # Check that touched_bar is always in the future relative to current index
    for i in range(n):
        if labels[i] != 0 and touched_bars[i] >= 0:
            # touched_bar is relative to current position
            absolute_touch = i + touched_bars[i]
            assert absolute_touch > i, "Touch must be in the future"
            assert absolute_touch < n, "Touch must be within bounds"


@pytest.mark.unit
@pytest.mark.data
def test_asymmetric_barriers_tp_sl_ratio() -> None:
    """Test that asymmetric TP/SL multipliers create correct barrier distances."""
    n = 50
    close = np.linspace(1800, 1900, n)
    high = close + 5
    low = close - 5
    atr = np.ones(n) * 10

    _, upper_barriers, lower_barriers, _, _ = _compute_labels(
        close=close,
        high=high,
        low=low,
        atr=atr,
        tp_mult=2.0,
        sl_mult=1.0,
        horizon=10,
        min_atr=0.0001,
    )

    for i in range(n):
        upper_dist = upper_barriers[i] - close[i]
        lower_dist = close[i] - lower_barriers[i]
        assert abs(upper_dist - 20.0) < 1e-10, (
            f"Upper barrier distance should be 20.0 (2.0 * 10.0 ATR), got {upper_dist}"
        )
        assert abs(lower_dist - 10.0) < 1e-10, (
            f"Lower barrier distance should be 10.0 (1.0 * 10.0 ATR), got {lower_dist}"
        )
