"""Spatial reconstructions that turn cell averages into left/right interface
states u_L, u_R at each face i+1/2. All are vectorized over a periodic grid
via np.roll, and all return arrays aligned so that index i is the interface
between cell i and cell i+1."""
from __future__ import annotations

import numpy as np


def first_order(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise-constant reconstruction (used by upwind / Lax-Friedrichs)."""
    u_left = u
    u_right = np.roll(u, -1)
    return u_left, u_right


def _minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    same_sign = (a * b) > 0
    return np.where(same_sign, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


def muscl(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise-linear MUSCL reconstruction with a minmod slope limiter."""
    u_im1 = np.roll(u, 1)
    u_ip1 = np.roll(u, -1)
    slope = _minmod(u - u_im1, u_ip1 - u)

    slope_ip1 = np.roll(slope, -1)
    u_left = u + 0.5 * slope  # value at right edge of cell i
    u_right = u_ip1 - 0.5 * slope_ip1  # value at left edge of cell i+1
    return u_left, u_right


def _weno5_biased(v1, v2, v3, v4, v5, eps: float = 1e-6) -> np.ndarray:
    """Classic Jiang-Shu WENO5 reconstruction, giving the value at the right
    edge of the cell holding v3, using the 5-point stencil (v1..v5)."""
    p0 = (2 * v1 - 7 * v2 + 11 * v3) / 6.0
    p1 = (-v2 + 5 * v3 + 2 * v4) / 6.0
    p2 = (2 * v3 + 5 * v4 - v5) / 6.0

    beta0 = (13.0 / 12.0) * (v1 - 2 * v2 + v3) ** 2 + 0.25 * (v1 - 4 * v2 + 3 * v3) ** 2
    beta1 = (13.0 / 12.0) * (v2 - 2 * v3 + v4) ** 2 + 0.25 * (v2 - v4) ** 2
    beta2 = (13.0 / 12.0) * (v3 - 2 * v4 + v5) ** 2 + 0.25 * (3 * v3 - 4 * v4 + v5) ** 2

    d0, d1, d2 = 0.1, 0.6, 0.3
    alpha0 = d0 / (eps + beta0) ** 2
    alpha1 = d1 / (eps + beta1) ** 2
    alpha2 = d2 / (eps + beta2) ** 2
    alpha_sum = alpha0 + alpha1 + alpha2

    w0, w1, w2 = alpha0 / alpha_sum, alpha1 / alpha_sum, alpha2 / alpha_sum
    return w0 * p0 + w1 * p1 + w2 * p2


def weno5(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """5th-order (in smooth regions) WENO reconstruction at every interface."""
    um2, um1 = np.roll(u, 2), np.roll(u, 1)
    up1, up2, up3 = np.roll(u, -1), np.roll(u, -2), np.roll(u, -3)

    u_left = _weno5_biased(um2, um1, u, up1, up2)
    # Mirror trick: the right-biased ("plus") reconstruction at the same
    # interface is the same formula applied to the spatially-reversed stencil.
    u_right = _weno5_biased(up3, up2, up1, u, um1)
    return u_left, u_right
