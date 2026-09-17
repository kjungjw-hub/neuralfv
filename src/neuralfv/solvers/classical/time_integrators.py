"""Shu-Osher strong-stability-preserving 3rd-order Runge-Kutta (SSP-RK3).

Used as the time integrator for every classical scheme in this repo, so that
observed convergence order in space is not bottlenecked by time integration.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def ssp_rk3_step(u: np.ndarray, dt: float, rhs: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    u1 = u + dt * rhs(u)
    u2 = 0.75 * u + 0.25 * (u1 + dt * rhs(u1))
    u_new = (1.0 / 3.0) * u + (2.0 / 3.0) * (u2 + dt * rhs(u2))
    return u_new
