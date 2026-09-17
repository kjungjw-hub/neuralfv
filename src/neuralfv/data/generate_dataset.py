"""Synthetic training data: run a fine, WENO5-resolved "truth" simulation and
block-average it down to a coarse grid at the large timestep the neural
solver will be trained to jump by. This produces dynamically-consistent
coarse cell-average trajectories -- the same "generate data by simulating
traditional numerical methods" idea from the original 2024 project, applied
at the resolution/timestep gap the network is meant to close.
"""
from __future__ import annotations

import numpy as np

from neuralfv.pdes.base import ConservationLaw
from neuralfv.solvers.classical.schemes import advance_to_time


def random_fourier_ic(domain_length: float, n_modes: int = 6, rng: np.random.Generator = None):
    """A smooth, random, periodic initial condition: a band-limited Fourier
    series with amplitudes decaying as 1/k so higher modes don't dominate."""
    rng = rng or np.random.default_rng()
    k = np.arange(1, n_modes + 1)
    cos_coeffs = rng.normal(size=n_modes) / k
    sin_coeffs = rng.normal(size=n_modes) / k
    phase = 2 * np.pi / domain_length

    def ic(x: np.ndarray) -> np.ndarray:
        angles = np.outer(x, k) * phase  # (len(x), n_modes)
        # Elementwise + sum rather than `@`: on macOS numpy's Accelerate BLAS
        # backend emits spurious divide-by-zero/overflow warnings from small
        # matmuls (values are unaffected, but the warnings are noise).
        return np.sum(np.cos(angles) * cos_coeffs, axis=-1) + np.sum(np.sin(angles) * sin_coeffs, axis=-1)

    return ic


def block_average(u_fine: np.ndarray, downsample_ratio: int) -> np.ndarray:
    n_coarse = u_fine.shape[-1] // downsample_ratio
    return u_fine.reshape(*u_fine.shape[:-1], n_coarse, downsample_ratio).mean(axis=-1)


def advance_fine_solution(u_fine: np.ndarray, dx_fine: float, dt_large: float,
                           pde: ConservationLaw, fine_cfl: float = 0.4) -> np.ndarray:
    """Advance the fine WENO5 "truth" solution by exactly dt_large."""
    return advance_to_time(u_fine, dx_fine, dt_large, pde, scheme="weno5", cfl=fine_cfl)


def generate_rollout_dataset(
    pde: ConservationLaw,
    domain_length: float,
    n_coarse: int,
    downsample_ratio: int,
    dt_large: float,
    n_large_steps: int,
    n_trajectories: int,
    n_fourier_modes: int = 6,
    fine_cfl: float = 0.4,
    seed: int = 0,
) -> np.ndarray:
    """Returns an array of shape (n_trajectories, n_large_steps + 1, n_coarse)
    of coarse cell-average trajectories, each advanced dt_large at a time."""
    rng = np.random.default_rng(seed)
    n_fine = n_coarse * downsample_ratio
    dx_fine = domain_length / n_fine
    x_fine = (np.arange(n_fine) + 0.5) * dx_fine

    trajectories = np.empty((n_trajectories, n_large_steps + 1, n_coarse))
    for traj_idx in range(n_trajectories):
        ic = random_fourier_ic(domain_length, n_fourier_modes, rng)
        u_fine = ic(x_fine)
        trajectories[traj_idx, 0] = block_average(u_fine, downsample_ratio)
        for step in range(n_large_steps):
            u_fine = advance_fine_solution(u_fine, dx_fine, dt_large, pde, fine_cfl)
            trajectories[traj_idx, step + 1] = block_average(u_fine, downsample_ratio)
    return trajectories
