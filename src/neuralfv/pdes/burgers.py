"""Inviscid Burgers' equation: u_t + (u^2/2)_x = 0.

This is the PDE that actually carries the resume claim: it forms shocks in
finite time from smooth data, so "cut compute cost / step past CFL limits"
is a real, measurable statement rather than a toy one.
"""
from __future__ import annotations

import numpy as np

from neuralfv.pdes.base import ConservationLaw


class Burgers(ConservationLaw):
    def flux(self, u: np.ndarray) -> np.ndarray:
        return 0.5 * u**2

    def flux_derivative(self, u: np.ndarray) -> np.ndarray:
        return u

    def godunov_flux(self, u_left: np.ndarray, u_right: np.ndarray) -> np.ndarray:
        """Exact Godunov flux for the convex flux f(u) = u^2/2, sonic point
        at u = 0 (Toro, "Riemann Solvers and Numerical Methods for Fluid
        Dynamics", ch. 2)."""
        f_left = self.flux(u_left)
        f_right = self.flux(u_right)

        shock = u_left > u_right
        shock_flux = np.maximum(f_left, f_right)

        straddles_sonic_point = (u_left <= 0.0) & (0.0 <= u_right)
        rarefaction_flux = np.where(straddles_sonic_point, 0.0, np.minimum(f_left, f_right))

        return np.where(shock, shock_flux, rarefaction_flux)
