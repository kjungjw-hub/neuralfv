from __future__ import annotations

import numpy as np
import torch

from neuralfv.data.datasets import RolloutWindowDataset
from neuralfv.solvers.neural.flux_net import ConservativeFluxNet
from neuralfv.solvers.neural.losses import rollout_mse
from neuralfv.solvers.neural.rollout import rollout


def test_rollout_output_shape():
    net = ConservativeFluxNet(stencil_half_width=3, hidden_channels=8)
    u0 = torch.randn(5, 32)
    preds = rollout(net, u0, dx=0.1, dt=0.01, n_steps=4, noise_std=0.0)
    assert preds.shape == (5, 4, 32)


def test_rollout_with_noise_injection_runs_and_differs_from_noiseless():
    torch.manual_seed(0)
    net = ConservativeFluxNet(stencil_half_width=3, hidden_channels=8)
    u0 = torch.randn(3, 32)
    torch.manual_seed(1)
    clean = rollout(net, u0, dx=0.1, dt=0.01, n_steps=3, noise_std=0.0)
    torch.manual_seed(1)
    noisy = rollout(net, u0, dx=0.1, dt=0.01, n_steps=3, noise_std=0.1)
    assert not torch.allclose(clean, noisy)


def test_gradients_flow_through_full_rollout():
    net = ConservativeFluxNet(stencil_half_width=3, hidden_channels=8)
    u0 = torch.randn(2, 16, requires_grad=False)
    targets = torch.randn(2, 3, 16)
    preds = rollout(net, u0, dx=0.1, dt=0.01, n_steps=3, noise_std=0.05)
    loss = rollout_mse(preds, targets)
    loss.backward()
    grads = [p.grad for p in net.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_rollout_window_dataset_indexing():
    trajectories = np.arange(2 * 6 * 4).reshape(2, 6, 4).astype(np.float32)
    dataset = RolloutWindowDataset(trajectories, rollout_length=2)
    assert len(dataset) == 2 * (5 - 2 + 1)
    u0, targets = dataset[0]
    assert torch.allclose(u0, torch.as_tensor(trajectories[0, 0]))
    assert torch.allclose(targets, torch.as_tensor(trajectories[0, 1:3]))
