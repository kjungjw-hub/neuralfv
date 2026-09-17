"""ConservativeFluxNet: the core architectural upgrade over the original
2024 approach.

The original project trained a network to regress the next cell-average
directly. That has no structural reason to conserve mass — any conservation
has to be learned, approximately, from data. Here the network instead
predicts the *numerical flux* at each cell interface from a local window of
cell averages, and the state update is the discrete divergence of those
fluxes:

    u_i^{n+1} = u_i^n - (dt/dx) * (F_{i+1/2} - F_{i-1/2})

Since every predicted flux F_{i+1/2} is added to cell i and subtracted from
cell i+1, total mass sum(u)*dx is invariant under this update for *any*
network weights -- including a freshly-initialized, untrained network. That
claim is proven directly in tests/test_flux_net_conservation.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConservativeFluxNet(nn.Module):
    """A small stencil CNN over cell averages, predicting one scalar flux
    per interface. Fully convolutional and periodic-padded, so it applies to
    any grid resolution."""

    def __init__(self, stencil_half_width: int = 3, hidden_channels: int = 32):
        super().__init__()
        self.stencil_half_width = stencil_half_width
        kernel_size = 2 * stencil_half_width  # even kernel: centered on an interface, not a cell

        # A single stencil-gathering conv (periodic padding) feeds a
        # pointwise MLP with the local window around each interface.
        self._gather = nn.Conv1d(1, hidden_channels, kernel_size=kernel_size, padding=0, bias=True)
        self._mlp = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, 1, kernel_size=1),
        )

    def predict_fluxes(self, u: torch.Tensor) -> torch.Tensor:
        """u: (batch, N) cell averages -> (batch, N) fluxes, where output[i]
        is the flux at interface i+1/2 (between cell i and cell i+1)."""
        x = u.unsqueeze(1)  # (batch, 1, N)
        pad = self.stencil_half_width
        # Periodic padding, then a valid-mode conv centered on each interface:
        # window [i - pad + 1, ..., i + pad] straddles interface i+1/2.
        x_padded = F.pad(x, (pad - 1, pad), mode="circular")
        hidden = self._gather(x_padded)  # (batch, hidden, N)
        flux = self._mlp(hidden)  # (batch, 1, N)
        return flux.squeeze(1)

    def forward(self, u: torch.Tensor, dx: float, dt: float) -> torch.Tensor:
        """One conservative finite-volume update step."""
        flux_face = self.predict_fluxes(u)  # flux_face[i] at interface i+1/2
        flux_face_prev = torch.roll(flux_face, shifts=1, dims=-1)  # at interface i-1/2
        return u - (dt / dx) * (flux_face - flux_face_prev)
