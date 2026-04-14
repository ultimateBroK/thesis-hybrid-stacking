"""GRU feature extractor for hybrid LightGBM pipeline.

Architecture:
    GRU Branch: log_returns + rsi_14 (24-bar window) → 64-dim hidden state
    The hidden state is concatenated with static features and fed to LightGBM.

The GRU is NOT an independent predictor — it's a sequence encoder that
captures temporal patterns in returns and momentum. LightGBM makes the
final Buy/Sell/Hold decision using both GRU hidden state and static features.

Key design choices:
    - GRU over LSTM: 25-30% fewer parameters, less overfitting on 33K samples
    - No Bidirectional: would introduce look-ahead bias in live trading
    - No Attention: not justified for 24-bar sequences
    - Single-direction, last hidden state only
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from thesis.config import Config

logger = logging.getLogger("thesis.gru_model")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class GRUExtractor(nn.Module):
    """GRU-based feature extractor.

    Encodes a (batch, seq_len, input_size) sequence into a single
    (batch, hidden_size) hidden state vector.

    Args:
        input_size: Number of features per timestep (2: log_returns + rsi_14).
        hidden_size: GRU hidden dimension.
        num_layers: Number of stacked GRU layers.
        dropout: Dropout between GRU layers (applied only if num_layers > 1).
    """

    def __init__(
        self,
        input_size: int = 2,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — return last hidden state.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size).

        Returns:
            Hidden state of shape (batch, hidden_size).
        """
        _, hidden = self.gru(x)
        # hidden shape: (num_layers, batch, hidden_size)
        # Take last layer's hidden state
        return hidden[-1]


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_gru(
    config: Config,
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
) -> tuple[GRUExtractor, nn.Linear, np.ndarray, np.ndarray]:
    """Train GRU feature extractor and produce hidden states for LightGBM.

    The GRU is trained as a classifier on the labels. After training,
    we extract the hidden state for each sample — these become features
    for LightGBM.

    Args:
        config: Application configuration with GRU parameters.
        train_df: Training DataFrame.
        val_df: Validation DataFrame.

    Returns:
        Tuple of (trained_model, classifier_head, train_hidden, val_hidden):
        - trained_model: GRUExtractor for hidden state extraction
        - classifier_head: nn.Linear for direct GRU predictions (ablation)
        - train_hidden_states: np.ndarray of shape (n_train, hidden_size)
        - val_hidden_states: np.ndarray of shape (n_val, hidden_size)
    """
    gru_cfg = config.gru
    gru_cols = ["log_returns", "rsi_14", "atr_14", "macd_hist"]
    seed = config.workflow.random_seed

    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Prepare sequences
    train_seq, train_labels, _ = prepare_sequences(
        train_df, gru_cols, gru_cfg.sequence_length
    )
    val_seq, val_labels, _ = prepare_sequences(
        val_df, gru_cols, gru_cfg.sequence_length
    )

    logger.info(
        "GRU sequences — train: %d, val: %d (seq_len=%d)",
        len(train_seq),
        len(val_seq),
        gru_cfg.sequence_length,
    )

    # Remap labels from {-1, 0, 1} to {0, 1, 2} for PyTorch CrossEntropyLoss
    if train_labels is not None:
        train_labels = (train_labels + 1).astype(np.int32)
    if val_labels is not None:
        val_labels = (val_labels + 1).astype(np.int32)

    # Create datasets & loaders
    train_dataset = SequenceDataset(train_seq, train_labels)
    val_dataset = SequenceDataset(val_seq, val_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=gru_cfg.batch_size,
        shuffle=False,  # Never shuffle time series!
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=gru_cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUExtractor(
        input_size=gru_cfg.input_size,
        hidden_size=gru_cfg.hidden_size,
        num_layers=gru_cfg.num_layers,
        dropout=gru_cfg.dropout,
    ).to(device)

    # Classification head for training only
    num_classes = config.labels.num_classes
    classifier = nn.Linear(gru_cfg.hidden_size, num_classes).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(classifier.parameters()),
        lr=gru_cfg.learning_rate,
    )
    criterion = nn.CrossEntropyLoss()

    # Training loop with early stopping
    best_val_loss = float("inf")
    best_state: dict[str, Any] | None = None
    patience_counter = 0

    for epoch in range(gru_cfg.epochs):
        # Train
        model.train()
        classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            hidden = model(batch_x)
            logits = classifier(hidden)

            loss = criterion(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(batch_x)
            train_correct += (logits.argmax(dim=1) == batch_y).sum().item()
            train_total += len(batch_y)

        train_loss /= train_total

        # Validate
        model.eval()
        classifier.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                hidden = model(batch_x)
                logits = classifier(hidden)

                loss = criterion(logits, batch_y)
                val_loss += loss.item() * len(batch_x)
                val_correct += (logits.argmax(dim=1) == batch_y).sum().item()
                val_total += len(batch_y)

        val_loss /= val_total
        val_acc = val_correct / val_total

        if (epoch + 1) % 5 == 0 or epoch == 0:
            train_acc = train_correct / train_total
            logger.info(
                "Epoch %d/%d — train_loss: %.4f train_acc: %.3f | "
                "val_loss: %.4f val_acc: %.3f",
                epoch + 1,
                gru_cfg.epochs,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                "model": model.state_dict(),
                "classifier": classifier.state_dict(),
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= gru_cfg.patience:
                logger.info(
                    "Early stopping at epoch %d (patience=%d)",
                    epoch + 1,
                    gru_cfg.patience,
                )
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state["model"])
        classifier.load_state_dict(best_state["classifier"])

    # Extract hidden states for LightGBM
    train_hidden = extract_hidden_states(model, train_seq, gru_cfg.batch_size, device)
    val_hidden = extract_hidden_states(model, val_seq, gru_cfg.batch_size, device)

    logger.info(
        "GRU hidden states — train: %s, val: %s",
        train_hidden.shape,
        val_hidden.shape,
    )

    return model, classifier, train_hidden, val_hidden


def extract_hidden_states(
    model: GRUExtractor,
    sequences: np.ndarray,
    batch_size: int = 64,
    device: torch.device | None = None,
) -> np.ndarray:
    """Extract hidden states from sequences using trained GRU.

    Args:
        model: Trained GRUExtractor.
        sequences: Input sequences of shape (n_samples, seq_len, input_size).
        batch_size: Batch size for inference.
        device: Device for computation.

    Returns:
        Hidden states of shape (n_samples, hidden_size).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    all_sequences = torch.from_numpy(sequences.copy()).float()

    hidden_states: list[np.ndarray] = []

    with torch.no_grad():
        for i in range(0, len(all_sequences), batch_size):
            batch = all_sequences[i : i + batch_size].to(device)
            hidden = model(batch)
            hidden_states.append(hidden.cpu().numpy())

    return np.concatenate(hidden_states, axis=0)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_gru_model(
    model: GRUExtractor,
    config: Config,
    path: str | Path,
) -> None:
    """Save GRU model weights and config to disk.

    Args:
        model: Trained GRUExtractor.
        config: Application configuration.
        path: Output path for model file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    gru_cfg = config.gru
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_size": gru_cfg.input_size,
        "hidden_size": gru_cfg.hidden_size,
        "num_layers": gru_cfg.num_layers,
        "dropout": gru_cfg.dropout,
        "sequence_length": gru_cfg.sequence_length,
    }
    torch.save(checkpoint, path)
    logger.info("GRU model saved: %s", path)


def load_gru_model(path: str | Path) -> tuple[GRUExtractor, dict[str, Any]]:
    """Load GRU model from disk.

    Args:
        path: Path to saved model file.

    Returns:
        Tuple of (model, checkpoint_metadata).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GRU model not found: {path}")

    checkpoint = torch.load(path, weights_only=False)

    model = GRUExtractor(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    metadata = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}

    logger.info(
        "GRU model loaded: %s (hidden_size=%d)", path, checkpoint["hidden_size"]
    )
    return model, metadata
