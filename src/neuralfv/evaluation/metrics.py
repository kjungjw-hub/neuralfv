"""The four benchmark metrics used consistently across README, demo, and
write-up: accuracy-at-cost, wall-clock speedup at matched accuracy, effective
CFL number, and conservation error."""
from __future__ import annotations

import numpy as np

from neuralfv.pdes.base import ConservationLaw


def mass(u: np.ndarray, dx: float) -> np.ndarray:
    return u.sum(axis=-1) * dx


def relative_l2_error(u: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(u - reference) / np.linalg.norm(reference))


def conservation_drift(trajectory: np.ndarray, dx: float) -> np.ndarray:
    """|mass(t) - mass(0)| at every saved step of a trajectory of shape (T+1, N)."""
    m = mass(trajectory, dx)
    return np.abs(m - m[0])


def effective_cfl(
    dt_used: float, dx: float, pde: ConservationLaw, u_reference: np.ndarray, cfl: float = 0.5
) -> float:
    """How many multiples of the classical CFL-limited timestep dt_used is."""
    speed = pde.max_wave_speed(u_reference)
    if speed == 0.0:
        return float("inf")
    dt_cfl_limit = cfl * dx / speed
    return dt_used / dt_cfl_limit


def speedup_at_matched_accuracy(
    candidate_error: float,
    candidate_time: float,
    baseline_errors: np.ndarray,
    baseline_times: np.ndarray,
) -> float:
    """Interpolate the baseline's error-vs-time curve (assumed monotonically
    decreasing error with increasing time/resolution) to find the baseline
    time that reaches `candidate_error`, then report baseline_time / candidate_time.
    Returns NaN if the candidate is more accurate than every baseline point
    (extrapolation would be unreliable) or less accurate than all of them."""
    order = np.argsort(baseline_times)
    times, errors = baseline_times[order], baseline_errors[order]
    if candidate_error > errors.max() or candidate_error < errors.min():
        return float("nan")
    # errors is decreasing in time; interpolate log(time) vs log(error)
    matched_log_time = np.interp(np.log(candidate_error), np.log(errors[::-1]), np.log(times[::-1]))
    matched_time = float(np.exp(matched_log_time))
    return matched_time / candidate_time
