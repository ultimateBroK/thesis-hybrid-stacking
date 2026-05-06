"""GRU inference — hidden-state extraction and probability prediction."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from thesis.stage_4_training._gru_arch import GRUExtractor


def extract_hidden_states(
    model: GRUExtractor,
    sequences: np.ndarray,
    batch_size: int = 64,
    device: torch.device | None = None,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> np.ndarray:
    """Extract final-layer hidden states for a batch of sequences.

    When ``mean`` and ``std`` are provided, the sequences are standardized
    using the same per-feature statistics that were computed during training.
    This ensures the model receives identically scaled input at inference
    time.

    Args:
        model: Trained GRU extractor used for forward passes.
        sequences: Input array with shape ``(n_samples, seq_len, input_size)``.
        batch_size: Number of samples per inference batch.
        device: Computation device. If ``None``, CUDA is used when available,
            otherwise CPU.
        mean: Per-feature mean with shape broadcastable to ``sequences``
            (typically ``(1, 1, n_features)``).  When ``None``, no
            standardization is applied.
        std: Per-feature standard deviation with the same shape convention
            as ``mean``.  Must be provided when ``mean`` is provided.

    Returns:
        Array of shape ``(n_samples, hidden_size)`` containing one hidden
        state vector per input sequence.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    # Apply the same standardization used during training
    data = sequences.copy()
    if mean is not None and std is not None:
        data = (data - mean) / std

    all_sequences = torch.from_numpy(data).float()

    hidden_states: list[np.ndarray] = []

    with torch.no_grad():
        for i in range(0, len(all_sequences), batch_size):
            batch = all_sequences[i : i + batch_size].to(device)
            hidden = model(batch)
            hidden_states.append(hidden.cpu().numpy())

    return np.concatenate(hidden_states, axis=0)


def predict_gru_proba(
    model: GRUExtractor,
    classifier: nn.Linear,
    sequences: np.ndarray,
    batch_size: int = 64,
    device: torch.device | None = None,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
    temperature: float | None = None,
) -> np.ndarray:
    """Predict class probabilities from a trained GRU backbone + classifier.

    Args:
        model: Trained GRU backbone.
        classifier: Classification head trained on top of the GRU hidden state.
        sequences: Input array with shape ``(n_samples, seq_len, input_size)``.
        batch_size: Number of samples per inference batch.
        device: Computation device. If ``None``, CUDA is used when available,
            otherwise CPU.
        mean: Optional training-set feature mean for standardization.
        std: Optional training-set feature std for standardization.
        temperature: Temperature scaling parameter for calibrated
            probabilities.  When ``None``, the value is read from
            ``model.temperature`` if available; otherwise ``T=1.0``
            (no scaling) is used.

    Returns:
        Array of shape ``(n_samples, n_classes)`` with softmax probabilities.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve temperature: explicit arg → model attribute → default 1.0.
    if temperature is None:
        temperature = getattr(model, "temperature", 1.0)

    model.eval()
    classifier.eval()

    data = sequences.copy()
    if mean is not None and std is not None:
        data = (data - mean) / std

    all_sequences = torch.from_numpy(data).float()
    probabilities: list[np.ndarray] = []

    with torch.no_grad():
        for i in range(0, len(all_sequences), batch_size):
            batch = all_sequences[i : i + batch_size].to(device)
            hidden = model(batch)
            logits = classifier(hidden)
            if temperature != 1.0:
                logits = logits / temperature
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())

    return np.concatenate(probabilities, axis=0)
