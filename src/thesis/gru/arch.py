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
        """
        Initialize the GRUExtractor module and configure its GRU layer.
        
        Parameters:
            input_size (int): Number of expected features in the input at each time step.
            hidden_size (int): Dimensionality of the GRU hidden state.
            num_layers (int): Number of stacked GRU layers.
            dropout (float): Dropout probability applied between GRU layers when `num_layers` > 1; ignored (treated as 0.0) when `num_layers` == 1.
        
        Notes:
            Stores `hidden_size` and `num_layers` as attributes and constructs `self.gru` with `batch_first=True`.
        """
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
        """
        Encode a batched sequence and return the final hidden state from the last GRU layer.
        
        Parameters:
            x (torch.Tensor): Input tensor shaped (batch, seq_len, input_size).
        
        Returns:
            torch.Tensor: Final hidden state from the last GRU layer shaped (batch, hidden_size).
        """
        _, hidden = self.gru(x)
        # hidden shape: (num_layers, batch, hidden_size)
        # Take last layer's hidden state
        return hidden[-1]
