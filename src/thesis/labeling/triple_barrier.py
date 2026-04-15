"""Triple-barrier labeling — simplified, no session-aware ATR.

Uses a single ``atr_multiplier`` for all hours. No DST detection,
no session definitions, no dead-hour filtering.

Classes:
    +1  Long  (take-profit barrier hit first)
     0  Hold  (neither barrier hit within horizon)
    -1  Short (stop-loss barrier hit first)
"""

import logging
from pathlib import Path

import numpy as np
import polars as pl

from thesis.config import Config

logger = logging.getLogger("thesis.labels")


def generate_labels(config: Config) -> None:
    """Generate triple-barrier labels and persist to parquet.

    Reads the features parquet (for ATR) and OHLCV parquet (for prices),
    computes labels, and writes the merged result to the labels path.

    Args:
        config: Loaded application configuration.
    """
    features_path = Path(config.paths.features)
    ohlcv_path = Path(config.paths.ohlcv)

    if not features_path.exists():
        raise FileNotFoundError(f"Features not found: {features_path}")
    if not ohlcv_path.exists():
        raise FileNotFoundError(f"OHLCV not found: {ohlcv_path}")

    logger.info("Loading features: %s", features_path)
    df_feat = pl.read_parquet(features_path)

    logger.info("Loading OHLCV: %s", ohlcv_path)
    df_ohlcv = pl.read_parquet(ohlcv_path).select(
        ["timestamp", "open", "high", "low", "close"]
    )

    # Join on timestamp
    df = df_feat.join(df_ohlcv, on="timestamp", how="inner")
    logger.info("Joined rows: %d", len(df))

    atr_col = f"atr_{config.features.atr_period}"
    if atr_col not in df.columns:
        raise ValueError(f"{atr_col} not in features. Run feature engineering first.")

    mult = config.labels.atr_multiplier
    horizon = config.labels.horizon_bars
    min_atr = config.labels.min_atr

    logger.info(
        "Triple-barrier params: mult=%.2f, horizon=%d, min_atr=%.6f",
        mult,
        horizon,
        min_atr,
    )

    # Compute labels
    labels_data = _compute_labels(
        close=df["close"].to_numpy(),
        high=df["high"].to_numpy(),
        low=df["low"].to_numpy(),
        atr=df[atr_col].to_numpy(),
        mult=mult,
        horizon=horizon,
        min_atr=min_atr,
    )

    ts_dtype = df["timestamp"].dtype
    labels_df = pl.DataFrame(
        {
            "timestamp": pl.Series(df["timestamp"].to_list(), dtype=ts_dtype),
            "label": labels_data["labels"],
            "tp_price": labels_data["tp_prices"],
            "sl_price": labels_data["sl_prices"],
            "touched_bar": labels_data["touched_bars"],
        }
    )

    df = df.join(labels_df, on="timestamp", how="left")

    # Log distribution
    _log_distribution(df)

    out_path = Path(config.paths.labels)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    logger.info("Labels saved: %s (%d rows)", out_path, len(df))


# ---------------------------------------------------------------------------
# Core labeling logic
# ---------------------------------------------------------------------------


def _compute_labels(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    mult: float,
    horizon: int,
    min_atr: float,
) -> dict:
    """Compute triple-barrier labels without lookahead bias.

    At each bar *i*, set barriers and scan bars *i+1 … i+horizon*
    sequentially. The first barrier touched determines the label.

    Returns:
        Dict with keys: labels, tp_prices, sl_prices, touched_bars.
    """
    n = len(close)
    labels = np.zeros(n, dtype=np.int32)
    tp_prices = np.zeros(n, dtype=np.float64)
    sl_prices = np.zeros(n, dtype=np.float64)
    touched_bars = np.full(n, -1, dtype=np.int32)

    for i in range(n):
        a = max(atr[i], min_atr)
        tp = close[i] + mult * a
        sl = close[i] - mult * a
        tp_prices[i] = tp
        sl_prices[i] = sl

        label = 0  # Hold by default
        for j in range(i + 1, min(i + 1 + horizon, n)):
            if high[j] >= tp:
                label = 1  # Long
                touched_bars[i] = j - i
                break
            if low[j] <= sl:
                label = -1  # Short
                touched_bars[i] = j - i
                break
        labels[i] = label

    return {
        "labels": labels,
        "tp_prices": tp_prices,
        "sl_prices": sl_prices,
        "touched_bars": touched_bars,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log_distribution(df: pl.DataFrame) -> None:
    """Log class distribution."""
    if "label" not in df.columns:
        return
    counts = df["label"].value_counts().sort("label")
    total = len(df)
    for row in counts.iter_rows():
        label, count = row
        logger.info("  Class %s: %d (%.1f%%)", label, count, count / total * 100)
