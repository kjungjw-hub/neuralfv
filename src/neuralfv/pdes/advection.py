"""Linear advection: u_t + a u_x = 0. Has a trivial exact solution, so it's
used to sanity-check the whole pipeline (data gen, training, metrics) before
moving to the shock-forming Burgers case."""
from __future__ import annotations

import numpy as np

from neuralfv.pdes.base import ConservationLaw


class LinearAdvection(ConservationLaw):
    def __init__(self, speed: float = 1.0):
        self.speed = speed

    def flux(self, u: np.ndarray) -> np.ndarray:
        return self.speed * u

    def flux_derivative(self, u: np.ndarray) -> np.ndarray:
        return np.full_like(u, self.speed)

    def godunov_flux(self, u_left: np.ndarray, u_right: np.ndarray) -> np.ndarray:
        # f is monotonic (f' = a everywhere), so the exact Riemann flux is
        # pure upwinding on the sign of the speed.
        return self.speed * (u_left if self.speed >= 0 else u_right)

    def exact_solution(self, x: np.ndarray, t: float, initial_condition, domain_length: float):
        x_upwind = (x - self.speed * t) % domain_length
        return initial_condition(x_upwind)
