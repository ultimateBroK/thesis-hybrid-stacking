"""GRU architecture components — model and dropout layers.

Separates nn.Module definitions from training logic so the architecture
can be imported without pulling in the full training dependency chain.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VariationalDropout(nn.Module):
    """Variational (locked) dropout — same mask across all timesteps.

    Standard dropout draws an independent mask per element, breaking the
    temporal correlation that RNNs rely on.  Variational dropout generates
    *one* mask per sample and broadcasts it over the entire sequence so
    the same features are consistently dropped at every timestep.

    Args:
        p: Dropout probability (0.0 = no dropout).
    """

    def __init__(self, p: float = 0.1) -> None:
        """Initialise variational dropout with probability *p*."""
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply variational dropout.

        Args:
            x: Tensor with shape ``(batch, seq_len, features)``.

        Returns:
            Tensor with identical shape, dropout applied.
        """
        if not self.training or self.p == 0:
            return x
        # One mask per sample, broadcast over all timesteps.
        mask = torch.bernoulli(torch.full_like(x[:, :1, :], 1 - self.p)) / (1 - self.p)
        return x * mask


class GRUExtractor(nn.Module):
    """GRU-based feature extractor with learned attention pooling.

    Encodes a (batch, seq_len, input_size) sequence into a single
    (batch, hidden_size) vector via a 2-layer GRU followed by attention
    weighted-sum over all timesteps.  The attention layer learns which
    positions in the input sequence are most informative, rather than
    blindly using the final hidden state.

    The architecture applies LayerNorm, variational dropout, GRU encoding,
    attention pooling, and an optional projection for bidirectional outputs.

    Args:
        input_size: Number of features per timestep.
        hidden_size: GRU hidden dimension.
        num_layers: Number of stacked GRU layers.
        dropout: Dropout between GRU layers (applied only if num_layers > 1).
        variational_dropout: Variational dropout probability applied to the
            GRU input (0.0 = disabled).
    """

    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        variational_dropout: float = 0.1,
        bidirectional: bool = False,
    ) -> None:
        """Initialize the GRU extractor module.

        Args:
            input_size: Number of expected features at each timestep.
            hidden_size: Dimensionality of the GRU hidden state.
            num_layers: Number of stacked GRU layers.
            dropout: Dropout probability applied between GRU layers when
                ``num_layers > 1``.
            variational_dropout: Variational dropout probability applied to
                the GRU input before the recurrent computation.
            bidirectional: If True, use a bidirectional GRU so each timestep
                sees both past and future context.  When enabled the GRU
                output dimension doubles to ``hidden_size * 2`` and a linear
                projection layer reduces it back to ``hidden_size``.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.input_norm = nn.LayerNorm(input_size)
        self.var_drop = VariationalDropout(p=variational_dropout)
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        # Bidirectional GRU concatenates forward + backward → hidden_size * 2
        gru_output_dim = hidden_size * 2 if bidirectional else hidden_size
        # Learned attention scorer: maps each timestep hidden state to a
        # scalar score.
        self.attn_scorer = nn.Linear(gru_output_dim, 1)
        # Project bidirectional output back to hidden_size;
        # Identity when unidirectional.
        self.proj = (
            nn.Linear(gru_output_dim, hidden_size) if bidirectional else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batched sequence into an attention-weighted context vector.

        Args:
            x: Input tensor with shape ``(batch, seq_len, input_size)``.

        Returns:
            Attention-weighted context vector with shape
            ``(batch, hidden_size)``.
        """
        x = self.input_norm(x)
        x = self.var_drop(x)
        gru_out, _ = self.gru(x)  # (batch, seq_len, D) where D = hidden_size * ndir

        # Compute attention weights over the sequence dimension.
        # scores: (batch, seq_len, 1) → squeeze → softmax over seq_len
        attn_weights = torch.softmax(
            self.attn_scorer(gru_out), dim=1
        )  # (batch, seq_len, 1)

        # Weighted sum of hidden states: (batch, seq_len, D) → (batch, D)
        context = (gru_out * attn_weights).sum(dim=1)
        # Project to hidden_size when bidirectional (Identity when unidirectional)
        return self.proj(context)
