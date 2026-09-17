"""Smoke test: a tiny training run should complete and reduce the loss.
This is what CI runs -- it exercises the full data -> model -> optimizer
pipeline, not just isolated units."""
from __future__ import annotations

import math

from neuralfv.training.train import train


def _tiny_config(pde: str) -> dict:
    return {
        "pde": pde,
        "advection_speed": 1.0,
        "domain_length": 6.283185307179586,
        "n_coarse": 16,
        "downsample_ratio": 4,
        "fine_cfl": 0.4,
        "n_fourier_modes": 4,
        "dt_large": 0.05,
        "n_large_steps": 4,
        "n_train_trajectories": 8,
        "n_val_trajectories": 4,
        "seed_train": 0,
        "seed_val": 1,
        "torch_seed": 0,
        "model": {"stencil_half_width": 2, "hidden_channels": 8},
        "training": {
            "batch_size": 4,
            "epochs": 6,
            "lr": 0.01,
            "rollout_length_final": 2,
            "rollout_curriculum_epochs": 3,
            "noise_std": 0.0,
        },
    }


def test_smoke_training_reduces_loss_advection():
    result = train(_tiny_config("advection"))
    losses = [h["train_loss"] for h in result["history"]]
    assert losses[-1] < losses[0]


def test_smoke_training_runs_burgers():
    result = train(_tiny_config("burgers"))
    assert len(result["history"]) == 6
    assert all(math.isfinite(h["train_loss"]) for h in result["history"])
