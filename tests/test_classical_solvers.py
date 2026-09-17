"""Correctness proofs for the classical finite-volume baselines: measured
spatial convergence order matches theory, and every scheme exactly conserves
mass on a periodic domain (to floating-point round-off)."""
from __future__ import annotations

import numpy as np
import pytest

from neuralfv.pdes.advection import LinearAdvection
from neuralfv.solvers.classical.schemes import max_stable_dt, run

DOMAIN_LENGTH = 2 * np.pi
PDE = LinearAdvection(speed=1.0)
INITIAL_CONDITION = np.sin


def _l2_error(scheme: str, n_cells: int, dt: float, n_steps: int) -> float:
    dx = DOMAIN_LENGTH / n_cells
    x = (np.arange(n_cells) + 0.5) * dx
    u0 = INITIAL_CONDITION(x)
    trajectory = run(u0, dx, dt, n_steps, PDE, scheme=scheme)
    exact = PDE.exact_solution(x, dt * n_steps, INITIAL_CONDITION, DOMAIN_LENGTH)
    return float(np.sqrt(np.mean((trajectory[-1] - exact) ** 2)))


def _observed_order(scheme: str, resolutions=(40, 80, 160)) -> float:
    # Fixed tiny dt (well under the CFL limit at every resolution) isolates
    # spatial error from SSP-RK3's own O(dt^3) temporal error.
    dt, n_steps = 1e-4, 50
    errors = [_l2_error(scheme, n, dt, n_steps) for n in resolutions]
    rates = [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
    return float(np.mean(rates))


@pytest.mark.parametrize(
    "scheme,expected_order,tolerance",
    [
        ("upwind", 1.0, 0.2),
        ("muscl", 1.6, 0.4),  # minmod clips slopes at smooth extrema; < 2 in L2 is expected
        ("weno5", 5.0, 0.7),
    ],
)
def test_convergence_order(scheme, expected_order, tolerance):
    order = _observed_order(scheme)
    assert abs(order - expected_order) < tolerance, f"{scheme}: observed order {order:.2f}"


@pytest.mark.parametrize("scheme", ["upwind", "lax_friedrichs", "muscl", "weno5"])
def test_exact_mass_conservation(scheme):
    n_cells = 80
    dx = DOMAIN_LENGTH / n_cells
    x = (np.arange(n_cells) + 0.5) * dx
    u0 = INITIAL_CONDITION(x)
    dt = max_stable_dt(u0, dx, PDE, cfl=0.5)
    trajectory = run(u0, dx, dt, n_steps=200, pde=PDE, scheme=scheme)

    mass_initial = u0.sum() * dx
    mass_final = trajectory[-1].sum() * dx
    assert abs(mass_final - mass_initial) < 1e-9
