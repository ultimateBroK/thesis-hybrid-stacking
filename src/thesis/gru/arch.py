"""GRU neural network architecture."""

import torch
import torch.nn as nn


class GRUExtractor(nn.Module):
    """GRU-based feature extractor.

    Encodes a (batch, seq_len, input_size) sequence into a single
    (batch, hidden_size) hidden state vector.

    Args:
        input_size: Number of features per timestep.
        hidden_size: GRU hidden dimension.
        num_layers: Number of stacked GRU layers.
        dropout: Dropout between GRU layers (applied only if num_layers > 1).
    """

    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        """Initialize the GRU extractor module.

        Args:
            input_size: Number of expected features at each timestep.
            hidden_size: Dimensionality of the GRU hidden state.
            num_layers: Number of stacked GRU layers.
            dropout: Dropout probability applied between GRU layers when
                ``num_layers > 1``.

        Returns:
            None.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.input_norm = nn.LayerNorm(input_size)
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batched sequence into final hidden states.

        Args:
            x: Input tensor with shape ``(batch, seq_len, input_size)``.

        Returns:
            Final hidden state from the last GRU layer with shape
            ``(batch, hidden_size)``.
        """
        x = self.input_norm(x)
        _, hidden = self.gru(x)
        # hidden shape: (num_layers, batch, hidden_size)
        # Take last layer's hidden state
        return hidden[-1]
