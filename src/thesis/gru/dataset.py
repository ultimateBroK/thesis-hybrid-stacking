"""GRU dataset and sequence preparation utilities."""

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """Sliding-window dataset for GRU input sequences.

    Each sample is a window of (sequence_length, input_size) values
    from the GRU input columns, plus the corresponding label.
    """

    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> None:
        self.sequences = torch.from_numpy(sequences.copy()).float()
        self.labels = (
            torch.from_numpy(labels.copy()).long() if labels is not None else None
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.labels is not None:
            return self.sequences[idx], self.labels[idx]
        return self.sequences[idx], None


def _sliding_windows(data: np.ndarray, window: int) -> np.ndarray:
    """Create sliding windows using stride tricks.

    Args:
        data: 2D array of shape (n_rows, n_features).
        window: Window size.

    Returns:
        3D array of shape (n_samples, window, n_features).
    """
    n_rows, n_features = data.shape
    n_samples = n_rows - window + 1

    strides = (data.strides[0], data.strides[0], data.strides[1])
    return np.lib.stride_tricks.as_strided(
        data,
        shape=(n_samples, window, n_features),
        strides=strides,
    )


def prepare_sequences(
    df: pl.DataFrame,
    gru_cols: list[str],
    sequence_length: int,
    label_col: str = "label",
    exclude_cols: frozenset[str] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    """Build sliding-window sequences for GRU training.

    Args:
        df: Feature-enriched DataFrame with GRU input columns and labels.
        gru_cols: Column names for GRU input (e.g. ['log_returns', 'rsi_14']).
        sequence_length: Window size for each sequence.
        label_col: Name of the label column.
        exclude_cols: Columns to exclude from static features.

    Returns:
        Tuple of (sequences, labels, static_feature_cols):
        - sequences: np.ndarray of shape (n_samples, seq_len, input_size)
        - labels: np.ndarray of shape (n_samples,) or None if label missing
        - static_feature_cols: list of column names for static features
    """
    if exclude_cols is None:
        exclude_cols = frozenset()

    # Compute log returns if not present
    if "log_returns" not in df.columns:
        df = df.with_columns(pl.col("close").log().diff().alias("log_returns"))
        # Fill first null
        df = df.fill_null(strategy="forward").fill_null(0.0)

    # Ensure gru_cols exist
    for col in gru_cols:
        if col not in df.columns:
            raise ValueError(f"GRU input column '{col}' not found in DataFrame")

    # Extract GRU input values
    gru_data = df.select(gru_cols).to_numpy()

    # Build sequences using sliding window
    n_rows = len(df)
    n_samples = n_rows - sequence_length + 1

    if n_samples <= 0:
        raise ValueError(
            f"DataFrame has {n_rows} rows, need at least {sequence_length} "
            f"for sequence_length={sequence_length}"
        )

    # Efficient sliding window using stride tricks
    sequences = _sliding_windows(gru_data, sequence_length)

    # Extract labels (aligned to end of each window)
    has_labels = label_col in df.columns
    labels = None
    if has_labels:
        label_values = df[label_col].to_numpy()
        labels = label_values[sequence_length - 1 :]

    # Identify static feature columns (everything except excluded + GRU inputs)
    gru_col_set = set(gru_cols)
    static_cols = [
        c
        for c in df.columns
        if c not in exclude_cols and c not in gru_col_set and c != label_col
    ]

    return sequences, labels, static_cols
