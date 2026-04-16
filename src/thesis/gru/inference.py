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
    """
    Extracts the final hidden state for each input sequence using a trained GRUExtractor.

    Parameters:
        model (GRUExtractor): Trained GRUExtractor used to compute hidden states.
        sequences (np.ndarray): Input sequences with shape (n_samples, seq_len, input_size).
        batch_size (int): Number of samples processed per inference batch.
        device (torch.device | None): Computation device; if `None`, selects CUDA if available, otherwise CPU.

    Returns:
        np.ndarray: Array of shape (n_samples, hidden_size) containing the final hidden state for each sequence.
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
    """
    Save the GRU extractor's weights and related GRU configuration to disk.

    Parameters:
    \tmodel (GRUExtractor): Trained GRU extractor whose state_dict will be saved.
    \tconfig (Config): Application configuration; `config.gru` supplies GRU hyperparameters and `sequence_length` to include in the checkpoint.
    \tpath (str | Path): Destination file path for the saved checkpoint. The parent directory will be created if it does not exist.
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
    """
    Load a saved GRUExtractor and its associated metadata from disk.

    Parameters:
        path (str | Path): Filesystem path to the saved checkpoint file produced by `save_gru_model`.

    Returns:
        tuple[GRUExtractor, dict[str, Any]]: A tuple where the first element is a `GRUExtractor` instance
        initialized with the saved weights and set to evaluation mode, and the second element is a
        metadata dictionary containing all checkpoint entries except the model's `state_dict`.

    Raises:
        FileNotFoundError: If the provided `path` does not exist.
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
