"""Numba-compiled triple-barrier labeling kernels."""

from __future__ import annotations

from numba import njit
import numpy as np

from thesis.shared.constants import CENSORED_LABEL


@njit(cache=True)
def compute_labels(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    tp_mult: float,
    sl_mult: float,
    horizon: int,
    min_atr: float,
    num_classes: int = 2,
) -> tuple:
    """Triple-barrier scan with directional vertical-barrier exit.

    Per bar i: upper = close[i] + tp_mult * max(atr[i], min_atr),
    lower = close[i] - sl_mult * max(atr[i], min_atr).
    Scan i+1..i+horizon for first horizontal-barrier touch.

    num_classes=2 (binary): vertical-barrier hit → label by sign of return
        at close[i+horizon] vs close[i].  Labels in {-2, -1, 1}.
    num_classes=3 (ternary): vertical-barrier hit → label=0 (HOLD).
        Labels in {-2, -1, 0, 1}.  AFML standard.

    -2 = censored (simultaneous hit, zero return, or insufficient horizon).

    Returns (labels, upper_barriers, lower_barriers, touched_bars, ambiguous_count).
    """
    n = len(close)
    labels = np.zeros(n, dtype=np.int32)
    upper_barriers = np.zeros(n, dtype=np.float64)
    lower_barriers = np.zeros(n, dtype=np.float64)
    touched_bars = np.full(n, CENSORED_LABEL, dtype=np.int32)
    ambiguous_count = 0

    for i in range(n):
        effective_atr = max(atr[i], min_atr)
        upper = close[i] + tp_mult * effective_atr
        lower = close[i] - sl_mult * effective_atr
        upper_barriers[i] = upper
        lower_barriers[i] = lower

        if i + horizon >= n:
            labels[i] = CENSORED_LABEL
            continue

        label = CENSORED_LABEL
        hit_offset = -1

        for j in range(i + 1, i + horizon + 1):
            if j >= n:
                break
            upper_hit = high[j] >= upper
            lower_hit = low[j] <= lower

            if upper_hit and lower_hit:
                ambiguous_count += 1
                label = CENSORED_LABEL
                hit_offset = j - i
                break
            if upper_hit:
                label = 1
                hit_offset = j - i
                break
            if lower_hit:
                label = -1
                hit_offset = j - i
                break

        if hit_offset < 0:
            if num_classes == 3:
                label = 0  # HOLD — no horizontal barrier triggered
            else:
                horizon_close = close[i + horizon]
                if horizon_close > close[i]:
                    label = 1
                elif horizon_close < close[i]:
                    label = -1

        labels[i] = label
        touched_bars[i] = hit_offset if hit_offset >= 0 else CENSORED_LABEL

    return labels, upper_barriers, lower_barriers, touched_bars, ambiguous_count


@njit(cache=True)
def compute_event_end(touched_bars: np.ndarray, horizon: int) -> np.ndarray:
    """Offset array → absolute end indices. -1/-2 → i+horizon, k≥0 → i+k."""
    n = len(touched_bars)
    event_end = np.empty(n, dtype=np.int32)

    for i in range(n):
        offset = touched_bars[i]
        if offset < 0:
            offset = horizon
        event_end[i] = i + offset

    return event_end


@njit(cache=True)
def compute_average_uniqueness(event_end: np.ndarray) -> np.ndarray:
    """Lopez de Prado average-uniqueness weights from concurrency.

    diff array → prefix sum → per-bar weight = mean(1/concurrency) over span.
    Normalized by mean.
    """
    n = len(event_end)
    diff = np.zeros(n + 1, dtype=np.float64)

    for i in range(n):
        end = event_end[i]
        if end < i:
            end = i
        if end >= n:
            end = n - 1
        diff[i] += 1.0
        diff[end + 1] -= 1.0

    concurrency = np.empty(n, dtype=np.float64)
    running = 0.0
    for i in range(n):
        running += diff[i]
        concurrency[i] = max(running, 1.0)

    inv_prefix = np.zeros(n + 1, dtype=np.float64)
    for i in range(n):
        inv_prefix[i + 1] = inv_prefix[i] + 1.0 / concurrency[i]

    weights = np.empty(n, dtype=np.float64)
    total = 0.0

    for i in range(n):
        end = event_end[i]
        if end < i:
            end = i
        if end >= n:
            end = n - 1
        span = end - i + 1
        weight = (inv_prefix[end + 1] - inv_prefix[i]) / span
        weight = max(weight, 1e-6)
        weights[i] = weight
        total += weight

    mean = total / n if n > 0 else 1.0
    if mean <= 0.0:
        mean = 1.0
    for i in range(n):
        weights[i] /= mean

    return weights
