"""GRU model training loop."""

import copy
import logging
import time
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from torch.utils.data import DataLoader

from thesis.config import Config
from thesis.gru.arch import GRUExtractor
from thesis.gru.dataset import SequenceDataset, prepare_sequences
from thesis.gru.inference import extract_hidden_states


logger = logging.getLogger("thesis.gru.train")


def _build_model_and_classifier(
    config: Config,
    device: torch.device,
) -> tuple[GRUExtractor, nn.Linear]:
    """Build GRU model and classification head.

    Args:
        config: Application configuration.
        device: Target device for model placement.

    Returns:
        Tuple of (GRUExtractor, nn.Linear classifier).
    """
    gru_cfg = config.gru
    model = GRUExtractor(
        input_size=gru_cfg.input_size,
        hidden_size=gru_cfg.hidden_size,
        num_layers=gru_cfg.num_layers,
        dropout=gru_cfg.dropout,
    ).to(device)

    num_classes = config.labels.num_classes
    classifier = nn.Linear(gru_cfg.hidden_size, num_classes).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "GRU: %d params, %d layers, hidden=%d, device=%s",
        total_params,
        gru_cfg.num_layers,
        gru_cfg.hidden_size,
        device,
    )
    return model, classifier


def _train_epoch(
    model: GRUExtractor,
    classifier: nn.Linear,
    train_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch.

    Args:
        model: GRU feature extractor.
        classifier: Linear classification head.
        train_loader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Target device.

    Returns:
        Tuple of (average_loss, accuracy).
    """
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

    return train_loss / train_total, train_correct / train_total


def _validate_epoch(
    model: GRUExtractor,
    classifier: nn.Linear,
    val_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
) -> tuple[float, float]:
    """Run one validation epoch.

    Args:
        model: GRU feature extractor.
        classifier: Linear classification head.
        val_loader: Validation data loader.
        criterion: Loss function.
        device: Target device.

    Returns:
        Tuple of (average_loss, accuracy).
    """
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

    return val_loss / val_total, val_correct / val_total


def train_gru(
    config: Config,
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
) -> tuple[GRUExtractor, nn.Linear, np.ndarray, np.ndarray, list[dict[str, float]]]:
    """Train a GRU classifier and extract hidden-state features.

    The GRU backbone is trained with a temporary linear head using
    cross-entropy loss and early stopping on validation loss. The best
    checkpoint is restored before hidden states are extracted for downstream
    LightGBM training.

    Args:
        config: Application configuration containing GRU and labeling settings.
        train_df: Training split as a time-series DataFrame.
        val_df: Validation split as a time-series DataFrame.

    Returns:
        A tuple containing ``(model, classifier, train_hidden, val_hidden,
        history)`` where ``train_hidden`` and ``val_hidden`` are extracted GRU
        embeddings and ``history`` stores per-epoch train/validation metrics.
    """
    gru_cfg = config.gru
    gru_cols = config.gru.feature_cols
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

    if train_seq.shape[0] == 0:
        raise ValueError(
            f"GRU training: 0 sequences from train data "
            f"(sequence_length={gru_cfg.sequence_length}, "
            f"train rows={len(train_df)}). "
            "Ensure train_df has >= sequence_length rows."
        )
    if val_seq.shape[0] == 0:
        raise ValueError(
            f"GRU validation: 0 sequences from val data "
            f"(sequence_length={gru_cfg.sequence_length}, "
            f"val rows={len(val_df)}). "
            "Ensure val_df has >= sequence_length rows."
        )

    # Remap labels from {-1, 0, 1} to {0, 1, 2} for PyTorch CrossEntropyLoss
    if train_labels is not None:
        train_labels = (train_labels + 1).astype(np.int32)
    if val_labels is not None:
        val_labels = (val_labels + 1).astype(np.int32)

    # Create datasets & loaders
    train_dataset = SequenceDataset(train_seq, train_labels)
    val_dataset = SequenceDataset(val_seq, val_labels)

    # Note: shuffle=True for training loader shuffles which sequences are processed in each
    # epoch (not the sequence order itself). This is standard practice for mini-batch RNN
    # training to improve generalization. Val loader keeps shuffle=False to preserve order.
    train_loader = DataLoader(
        train_dataset,
        batch_size=gru_cfg.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=gru_cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classifier = _build_model_and_classifier(config, device)

    # Optimizer
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(classifier.parameters()),
        lr=gru_cfg.learning_rate,
    )
    criterion = nn.CrossEntropyLoss()

    # Training loop with early stopping
    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    patience_counter = 0
    history: list[dict[str, float]] = []
    stage_start = time.perf_counter()

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]GRU training"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TextColumn("[cyan]t_loss={task.fields[t_loss]:.4f}"),
        TextColumn("[green]t_acc={task.fields[t_acc]:.3f}"),
        TextColumn("•"),
        TextColumn("[cyan]v_loss={task.fields[v_loss]:.4f}"),
        TextColumn("[green]v_acc={task.fields[v_acc]:.3f}"),
        TimeElapsedColumn(),
        transient=False,
        console=Console(stderr=True),
    )

    with progress:
        task = progress.add_task(
            "epochs",
            total=gru_cfg.epochs,
            t_loss=0.0,
            t_acc=0.0,
            v_loss=0.0,
            v_acc=0.0,
        )

        for epoch in range(gru_cfg.epochs):
            train_loss, train_acc = _train_epoch(
                model, classifier, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc = _validate_epoch(
                model, classifier, val_loader, criterion, device
            )

            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": round(train_loss, 4),
                    "train_acc": round(train_acc, 4),
                    "val_loss": round(val_loss, 4),
                    "val_acc": round(val_acc, 4),
                }
            )

            progress.update(
                task,
                advance=1,
                t_loss=train_loss,
                t_acc=train_acc,
                v_loss=val_loss,
                v_acc=val_acc,
            )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_state = {
                    "model": copy.deepcopy(model.state_dict()),
                    "classifier": copy.deepcopy(classifier.state_dict()),
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= gru_cfg.patience:
                    if epoch + 1 < gru_cfg.epochs:
                        logger.info(
                            "Early stop at epoch %d (patience=%d)",
                            epoch + 1,
                            gru_cfg.patience,
                        )
                    break

    # Summary
    total_time = time.perf_counter() - stage_start
    best_val_acc = history[best_epoch - 1]["val_acc"] if history else 0.0
    logger.info(
        "GRU done: %d/%d epochs, best=e%d v_loss=%.4f v_acc=%.3f (%.1fs)",
        len(history),
        gru_cfg.epochs,
        best_epoch,
        best_val_loss,
        best_val_acc,
        total_time,
    )

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state["model"])
        classifier.load_state_dict(best_state["classifier"])

    # Extract hidden states for LightGBM
    train_hidden = extract_hidden_states(model, train_seq, gru_cfg.batch_size, device)
    val_hidden = extract_hidden_states(model, val_seq, gru_cfg.batch_size, device)

    logger.info(
        "GRU hidden states: train=%s, val=%s", train_hidden.shape, val_hidden.shape
    )

    return model, classifier, train_hidden, val_hidden, history
