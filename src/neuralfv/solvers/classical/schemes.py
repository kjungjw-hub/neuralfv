"""Four classical finite-volume schemes, all built from the same generic
conservative-update skeleton: reconstruct interface states -> compute a
numerical flux -> take the discrete divergence -> integrate in time with
SSP-RK3. They differ only in reconstruction order and Riemann solver, which
is exactly the axis neuralfv.solvers.neural.flux_net.ConservativeFluxNet
also upgrades.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from neuralfv.pdes.base import ConservationLaw
from neuralfv.solvers.classical import reconstructions as recon
from neuralfv.solvers.classical.time_integrators import ssp_rk3_step

Reconstructor = Callable[[np.ndarray], "tuple[np.ndarray, np.ndarray]"]
RiemannSolver = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _rhs(u: np.ndarray, dx: float, reconstruct: Reconstructor, riemann: RiemannSolver) -> np.ndarray:
    u_left, u_right = reconstruct(u)
    flux_face = riemann(u_left, u_right)  # flux_face[i] = flux at interface i+1/2
    flux_divergence = flux_face - np.roll(flux_face, 1)  # F[i+1/2] - F[i-1/2]
    return -flux_divergence / dx


def _make_stepper(reconstruct: Reconstructor, riemann_name: str):
    def step(u: np.ndarray, dx: float, dt: float, pde: ConservationLaw) -> np.ndarray:
        riemann = pde.godunov_flux if riemann_name == "godunov" else pde.lax_friedrichs_flux
        return ssp_rk3_step(u, dt, lambda v: _rhs(v, dx, reconstruct, riemann))

    return step


upwind_step = _make_stepper(recon.first_order, "godunov")
lax_friedrichs_step = _make_stepper(recon.first_order, "lax_friedrichs")
muscl_step = _make_stepper(recon.muscl, "godunov")
weno5_step = _make_stepper(recon.weno5, "godunov")

SCHEMES = {
    "upwind": upwind_step,
    "lax_friedrichs": lax_friedrichs_step,
    "muscl": muscl_step,
    "weno5": weno5_step,
}


def max_stable_dt(u: np.ndarray, dx: float, pde: ConservationLaw, cfl: float = 0.5) -> float:
    speed = pde.max_wave_speed(u)
    if speed == 0.0:
        return float("inf")
    return cfl * dx / speed


def run(
    u0: np.ndarray,
    dx: float,
    dt: float,
    n_steps: int,
    pde: ConservationLaw,
    scheme: str = "weno5",
) -> np.ndarray:
    """Integrate n_steps of size dt, returning trajectory of shape (n_steps+1, N)."""
    step_fn = SCHEMES[scheme]
    trajectory = np.empty((n_steps + 1, u0.shape[0]), dtype=u0.dtype)
    trajectory[0] = u0
    u = u0.copy()
    for n in range(n_steps):
        u = step_fn(u, dx, dt, pde)
        trajectory[n + 1] = u
    return trajectory


def advance_to_time(
    u: np.ndarray, dx: float, dt_target: float, pde: ConservationLaw,
    scheme: str = "weno5", cfl: float = 0.4,
) -> np.ndarray:
    """Advance exactly dt_target, subdividing into enough CFL-stable substeps
    (recomputed from the current state, since wave speed can change as the
    solution evolves -- e.g. Burgers shock formation). Used both to generate
    training data at a coarser cadence than the solver's own stable step, and
    to put a classical scheme on the same reporting cadence as the neural
    solver for apples-to-apples error/conservation comparisons."""
    step_fn = SCHEMES[scheme]
    dt_stable = max_stable_dt(u, dx, pde, cfl=cfl)
    n_substeps = max(1, int(np.ceil(dt_target / dt_stable)))
    dt_sub = dt_target / n_substeps
    for _ in range(n_substeps):
        u = step_fn(u, dx, dt_sub, pde)
    return u
