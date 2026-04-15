"""GRU model training loop."""

import logging
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from thesis.config import Config
from thesis.gru.arch import GRUExtractor
from thesis.gru.dataset import SequenceDataset, prepare_sequences
from thesis.gru.inference import extract_hidden_states

logger = logging.getLogger("thesis.gru.train")


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
        Tuple of (trained_model, classifier_head, train_hidden, val_hidden).
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
