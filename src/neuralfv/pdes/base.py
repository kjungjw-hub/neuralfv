"""Common interface for 1D scalar hyperbolic conservation laws on a periodic domain."""
from __future__ import annotations

import numpy as np


class ConservationLaw:
    """u_t + f(u)_x = 0 on a periodic domain, defined by its flux f and f'."""

    def flux(self, u: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def flux_derivative(self, u: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def godunov_flux(self, u_left: np.ndarray, u_right: np.ndarray) -> np.ndarray:
        """Exact Riemann-solver numerical flux at an interface, given the
        reconstructed state just to the left (u_left) and just to the right
        (u_right) of that interface."""
        raise NotImplementedError

    def max_wave_speed(self, u: np.ndarray) -> float:
        return float(np.max(np.abs(self.flux_derivative(u))))

    def lax_friedrichs_flux(self, u_left: np.ndarray, u_right: np.ndarray) -> np.ndarray:
        """Global Lax-Friedrichs numerical flux; works for any flux function."""
        alpha = max(self.max_wave_speed(u_left), self.max_wave_speed(u_right))
        return 0.5 * (self.flux(u_left) + self.flux(u_right)) - 0.5 * alpha * (u_right - u_left)
