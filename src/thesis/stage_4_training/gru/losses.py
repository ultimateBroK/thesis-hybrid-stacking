"""Loss functions for GRU training.

FocalLoss handles class imbalance by down-weighting easy examples.
NT-Xent provides contrastive pretraining via cosine similarity.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812 – standard PyTorch abbreviation


class FocalLoss(nn.Module):
    """Focal Loss for multi-class classification.

    Down-weights easy (well-classified) examples so the model focuses
    on hard, misclassified samples.  This is critical when one class
    dominates (e.g. Hold ≈ 69 % in 3-class financial labeling).

    The loss is ``-alpha_t * (1 - p_t)^gamma * log(p_t)``, where ``p_t`` is
    the predicted probability of the correct class and ``alpha_t`` is an
    optional per-class weight.

    Args:
        gamma: Focusing parameter.  ``gamma=0`` reduces to standard
            cross-entropy.  Default ``2.0`` follows Lin et al.
        alpha: Per-class weights as a 1-D tensor of length
            ``num_classes``, or ``None`` for uniform weighting.
            Typically set to inverse class frequencies.
        num_classes: Number of target classes (default 3).

    Inputs use logits with shape ``(N, C)`` and targets with shape ``(N,)``;
    the output is a scalar tensor.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        num_classes: int = 3,
    ) -> None:
        """Initialise Focal Loss with *gamma*, optional *alpha* weights."""
        super().__init__()
        self.gamma = gamma
        self.num_classes = num_classes

        if alpha is not None:
            if alpha.dim() != 1 or alpha.size(0) != num_classes:
                msg = (
                    f"alpha must be a 1-D tensor of length {num_classes}, "
                    f"got shape {tuple(alpha.shape)}"
                )
                raise ValueError(msg)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: ``(N, C)`` raw scores.
            targets: ``(N,)`` integer class labels.
            sample_weight: Optional per-sample weights aligned to ``targets``.

        Returns:
            Scalar loss tensor.
        """
        # Softmax probabilities
        probs = F.softmax(logits, dim=-1)

        # Probability of the correct class per sample  →  (N,)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        p_t = (probs * targets_one_hot).sum(dim=-1).clamp(min=1e-8)

        # Standard cross-entropy (no reduction)  →  (N,)
        ce_loss = F.cross_entropy(logits, targets, reduction="none")

        # Focal modulating factor  (1 - p_t)^gamma
        focal_weight = (1.0 - p_t) ** self.gamma

        # Alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
        else:
            alpha_t = 1.0

        loss = alpha_t * focal_weight * ce_loss
        if sample_weight is not None:
            loss = loss * sample_weight.to(logits.device).float()
        return loss.mean()


def _nt_xent_loss(
    z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    """NT-Xent (InfoNCE) contrastive loss with cosine similarity.

    Computes the normalized temperature-scaled cross-entropy loss
    between two views of each sample. Each anchor in view 1 is
    paired with the corresponding sample in view 2 as a positive,
    while all other samples in the batch serve as negatives.

    Args:
        z1: First view embeddings, shape ``(N, D)``.
        z2: Second view embeddings, shape ``(N, D)``.
        temperature: Temperature scaling parameter (lower = harder
            positives, more uniform distribution). Default 0.1.

    Returns:
        Scalar loss tensor.
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    N = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # (2N, D)
    sim = z @ z.T / temperature  # (2N, 2N)

    # Mask self-similarity so each sample is not its own positive
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, float("-inf"))

    # Positive pairs: (i, i+N) for i in [0,N) and (i+N, i) for i in [0,N)
    labels = torch.cat([torch.arange(N, 2 * N), torch.arange(N)], dim=0).to(z.device)

    return F.cross_entropy(sim, labels)
