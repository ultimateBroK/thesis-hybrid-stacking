"""GRU model inference and persistence."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from thesis.config import Config
from thesis.gru.arch import GRUExtractor

logger = logging.getLogger("thesis.gru.inference")


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
