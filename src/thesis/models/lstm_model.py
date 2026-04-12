"""LSTM model training for sequential data."""

import logging
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from thesis.config.loader import Config

logger = logging.getLogger("thesis.models")


class LSTMClassifier(nn.Module):
    """Sequence classifier built on top of an LSTM encoder.

    The network consumes fixed-length OHLCV sequences and outputs logits for
    the three trading labels ``[-1, 0, 1]`` (encoded as ``[0, 1, 2]`` during
    training).
    """

    def __init__(
        self, input_size, hidden_size, num_layers, num_classes, dropout, bidirectional
    ):
        """Initialize LSTM sequence classifier layers.

        Args:
            input_size: Number of input features per timestep.
            hidden_size: Hidden dimension of the LSTM encoder.
            num_layers: Number of stacked LSTM layers.
            num_classes: Number of output classes.
            dropout: Dropout probability applied after the encoder.
            bidirectional: Whether to use a bidirectional LSTM encoder.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the classifier.

        Args:
            x: Input tensor with shape ``(batch, seq_len, input_size)``.

        Returns:
            Logits tensor with shape ``(batch, num_classes)``.
        """
        lstm_out, _ = self.lstm(x)
        # Use the final timestep representation for sequence classification.
        last_out = lstm_out[:, -1, :]
        last_out = self.dropout(last_out)
        out = self.fc(last_out)
        return out


def train_lstm(config: Config) -> None:
    """Train the LSTM base model and persist validation predictions.

    This routine prepares normalized OHLCV sequences, trains the LSTM with
    early stopping, saves the best checkpoint, stores normalization statistics
    for inference, and writes validation probabilities for stacking.

    Args:
        config: Loaded application configuration.
    """
    logger.info("Preparing LSTM data...")

    train_path = Path(config.paths.train_data)
    val_path = Path(config.paths.val_data)

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("Training/validation data not found.")

    # Load data
    train_df = pl.read_parquet(train_path)
    val_df = pl.read_parquet(val_path)

    # Prepare sequences
    seq_length = config.models["lstm"].sequence_length

    # Select OHLCV features for sequences
    ohlcv_cols = ["open", "high", "low", "close", "volume"]

    X_train, y_train, train_means, train_stds = _create_sequences(
        train_df, ohlcv_cols, seq_length
    )
    X_val, y_val, _, _ = _create_sequences(
        val_df, ohlcv_cols, seq_length, norm_stats=(train_means, train_stds)
    )

    logger.info(f"Training sequences: {X_train.shape}")
    logger.info(f"Validation sequences: {X_val.shape}")

    # Device
    device = torch.device(config.models["lstm"].device)
    logger.info(f"Using device: {device}")

    # Model
    input_size = len(ohlcv_cols)
    hidden_size = config.models["lstm"].hidden_size
    num_layers = config.models["lstm"].num_layers
    num_classes = 3
    dropout = config.models["lstm"].dropout
    bidirectional = config.models["lstm"].bidirectional

    model = LSTMClassifier(
        input_size, hidden_size, num_layers, num_classes, dropout, bidirectional
    )
    model = model.to(device)

    # Data loaders
    batch_size = config.models["lstm"].batch_size
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train),
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.models["lstm"].learning_rate,
        weight_decay=config.models["lstm"].weight_decay,
    )

    best_val_loss = float("inf")
    patience_counter = 0

    epochs = config.models["lstm"].epochs

    logger.info(f"Training for up to {epochs} epochs...")

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch + 1}/{epochs} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}"
            )

        # Early stopping
        if val_loss < best_val_loss - config.models["lstm"].min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            if config.models["lstm"].save_best:
                torch.save(model.state_dict(), config.models["lstm"].model_path)
        else:
            patience_counter += 1
            if patience_counter >= config.models["lstm"].patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    # Load best model
    model.load_state_dict(torch.load(config.models["lstm"].model_path))

    # Generate predictions
    logger.info("Generating LSTM predictions...")
    model.eval()

    with torch.no_grad():
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        outputs = model(X_val_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()

    # Save training normalization statistics for test predictions
    stats_path = Path(config.models["lstm"].model_path).parent / "lstm_norm_stats.npz"
    np.savez(stats_path, means=train_means, stds=train_stds)
    logger.info(f"Saved LSTM normalization stats: {stats_path}")

    # Save predictions with timestamps for alignment
    # Get timestamps aligned with sequences (seq_length offset)
    # Cast to same dtype as LightGBM (preserve timezone info)
    timestamps = val_df["timestamp"].to_numpy()[seq_length:]
    true_labels = val_df["label"].to_numpy()[seq_length:]

    preds_df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "true_label": true_labels,
            "pred_proba_class_1": probs[:, 2],  # Long
            "pred_proba_class_0": probs[:, 1],  # Hold
            "pred_proba_class_minus1": probs[:, 0],  # Short
        }
    )

    # Ensure timestamp has same dtype as LightGBM predictions
    if str(val_df["timestamp"].dtype) != str(preds_df["timestamp"].dtype):
        preds_df = preds_df.with_columns(
            pl.col("timestamp").cast(val_df["timestamp"].dtype)
        )

    preds_path = Path(config.models["lstm"].predictions_path)
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    preds_df.write_parquet(preds_path)
    logger.info(f"Saved LSTM predictions: {preds_path}")


def _create_sequences(
    df: pl.DataFrame,
    feature_cols: list[str],
    seq_length: int,
    norm_stats: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create sequences for LSTM.

    Args:
        df: Input dataframe containing features and label.
        feature_cols: Columns to use as features.
        seq_length: Sequence length.
        norm_stats: Optional ``(means, stds)`` from the training set.  When
            provided, these statistics are used instead of computing them from
            *df* to prevent lookahead bias on validation / test data.

    Returns:
        Tuple of ``(X, y, feature_means, feature_stds)`` arrays where ``X`` has
        shape ``(n_samples, seq_length, n_features)`` and ``y`` is class-encoded.
    """
    features = df.select(feature_cols).to_numpy()
    labels = df["label"].to_numpy()

    # Normalize features per column
    if norm_stats is not None:
        feature_means, feature_stds = norm_stats
    else:
        feature_means = features.mean(axis=0)
        feature_stds = features.std(axis=0)
    feature_stds = np.where(feature_stds == 0, 1.0, feature_stds)
    features = (features - feature_means) / feature_stds

    # Remap labels: -1, 0, 1 -> 0, 1, 2 for PyTorch CrossEntropyLoss
    label_map = {-1: 0, 0: 1, 1: 2}
    labels = np.array([label_map[int(label_val)] for label_val in labels])

    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i : i + seq_length])
        y.append(labels[i + seq_length])

    return np.array(X), np.array(y), feature_means, feature_stds
