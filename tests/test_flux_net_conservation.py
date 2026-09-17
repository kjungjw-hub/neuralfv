"""The single most important test in this repo: it proves that
ConservativeFluxNet exactly conserves the domain's total mass by
construction, on a randomly-initialized, UNTRAINED network. This decouples
"is the architecture correct" from "did training work" -- conservation is a
structural guarantee, not something the network has to learn."""
from __future__ import annotations

import torch

from neuralfv.solvers.neural.flux_net import ConservativeFluxNet


def test_untrained_network_exactly_conserves_mass():
    torch.manual_seed(0)
    net = ConservativeFluxNet(stencil_half_width=3, hidden_channels=16)
    net.eval()

    n_cells, dx, dt = 64, 0.1, 0.01
    u0 = torch.randn(1, n_cells)

    with torch.no_grad():
        mass_before = (u0.sum(dim=-1) * dx).item()
        u1 = net(u0, dx=dx, dt=dt)
        mass_after = (u1.sum(dim=-1) * dx).item()

    assert abs(mass_after - mass_before) < 1e-5


def test_conservation_holds_over_many_steps_and_batches():
    torch.manual_seed(1)
    net = ConservativeFluxNet(stencil_half_width=3, hidden_channels=16)
    net.eval()

    n_cells, dx, dt = 50, 0.2, 0.005
    u = torch.randn(4, n_cells)
    mass_initial = (u.sum(dim=-1) * dx).clone()

    with torch.no_grad():
        for _ in range(100):
            u = net(u, dx=dx, dt=dt)

    mass_final = u.sum(dim=-1) * dx
    assert torch.allclose(mass_final, mass_initial, atol=1e-4)


def test_output_shape_matches_input_at_multiple_resolutions():
    net = ConservativeFluxNet(stencil_half_width=3, hidden_channels=8)
    net.eval()
    for n_cells in (16, 33, 128):
        u = torch.randn(2, n_cells)
        with torch.no_grad():
            out = net(u, dx=1.0, dt=0.01)
        assert out.shape == u.shape
