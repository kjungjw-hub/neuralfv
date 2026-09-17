"""Wraps precomputed coarse trajectories (from generate_dataset.py) into a
PyTorch Dataset of fixed-length rollout windows, for multi-step training."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class RolloutWindowDataset(Dataset):
    """Each item is (u0, targets) where targets has shape (rollout_length, N):
    the true coarse states dt_large, 2*dt_large, ..., rollout_length*dt_large
    after u0, all drawn from the same underlying trajectory."""

    def __init__(self, trajectories: np.ndarray, rollout_length: int):
        n_traj, n_steps_plus_one, _n_cells = trajectories.shape
        n_steps = n_steps_plus_one - 1
        if rollout_length > n_steps:
            raise ValueError(
                f"rollout_length={rollout_length} exceeds available steps={n_steps} per trajectory"
            )
        self.trajectories = torch.as_tensor(trajectories, dtype=torch.float32)
        self.rollout_length = rollout_length
        self.starts_per_traj = n_steps - rollout_length + 1
        self.n_traj = n_traj

    def __len__(self) -> int:
        return self.n_traj * self.starts_per_traj

    def __getitem__(self, idx: int):
        traj_idx, start = divmod(idx, self.starts_per_traj)
        window = self.trajectories[traj_idx, start : start + self.rollout_length + 1]
        return window[0], window[1:]
