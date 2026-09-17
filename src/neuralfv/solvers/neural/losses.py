"""Rollout loss: mean-squared error averaged over every step of the unrolled
window, so early- and late-horizon accuracy are weighted equally rather than
letting the loss be dominated by the (typically easier) first step."""
from __future__ import annotations

import torch


def rollout_mse(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """predictions, targets: (batch, n_steps, N)."""
    return torch.mean((predictions - targets) ** 2)
