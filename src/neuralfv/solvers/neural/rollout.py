"""Multi-step rollout with noise injection ("pushforward"-style training).

Training on a single one-step prediction lets a network be locally accurate
but still blow up when its own small errors compound over a long rollout.
Unrolling the model for several steps during training, and perturbing the
intermediate states with noise scaled to the data itself, forces the network
to be self-correcting under its own error distribution -- this is what
actually earns the "steps past CFL limits for many steps" claim, not just a
single accurate jump.
"""
from __future__ import annotations

import torch

from neuralfv.solvers.neural.flux_net import ConservativeFluxNet


def rollout(
    model: ConservativeFluxNet,
    u0: torch.Tensor,
    dx: float,
    dt: float,
    n_steps: int,
    noise_std: float = 0.0,
) -> torch.Tensor:
    """Unroll the model n_steps forward. u0: (batch, N).
    Returns predictions of shape (batch, n_steps, N).

    When noise_std > 0 (training only), each intermediate state is perturbed
    before being fed back in, so the network sees inputs resembling its own
    compounding rollout error rather than only clean ground truth.
    """
    u = u0
    predictions = []
    for _ in range(n_steps):
        if noise_std > 0:
            scale = u.std(dim=-1, keepdim=True).clamp_min(1e-6)
            u = u + noise_std * scale * torch.randn_like(u)
        u = model(u, dx=dx, dt=dt)
        predictions.append(u)
    return torch.stack(predictions, dim=1)
