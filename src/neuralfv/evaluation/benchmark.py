"""End-to-end benchmark: loads a trained checkpoint, compares it against
classical finite-volume baselines across a resolution sweep, and computes
the four headline metrics (accuracy-at-cost, speedup at matched accuracy,
effective CFL, conservation drift). Every figure in the README/write-up
comes from a run of this script.

Usage: python -m neuralfv.evaluation.benchmark
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from neuralfv.data.generate_dataset import advance_fine_solution, block_average, random_fourier_ic
from neuralfv.evaluation import make_plots
from neuralfv.evaluation.metrics import (
    conservation_drift,
    effective_cfl,
    relative_l2_error,
    speedup_at_matched_accuracy,
)
from neuralfv.pdes.advection import LinearAdvection
from neuralfv.pdes.burgers import Burgers
from neuralfv.solvers.classical.schemes import advance_to_time, max_stable_dt
from neuralfv.solvers.classical.schemes import run as run_classical
from neuralfv.solvers.neural.flux_net import ConservativeFluxNet

PDE_BUILDERS = {
    "advection": lambda cfg: LinearAdvection(speed=cfg.get("advection_speed", 1.0)),
    "burgers": lambda cfg: Burgers(),
}
CLASSICAL_RESOLUTIONS = (16, 32, 64, 128, 256)
CLASSICAL_SCHEMES = ("upwind", "muscl", "weno5")


def load_checkpoint(path: str):
    ckpt = torch.load(path, weights_only=False)
    net = ConservativeFluxNet(**ckpt["model_cfg"])
    net.load_state_dict(ckpt["model_state"])
    net.eval()
    return net, ckpt["config"]


def fine_reference_trajectory(pde, cfg: dict, ic, n_eval_steps: int):
    n_fine = cfg["n_coarse"] * cfg["downsample_ratio"]
    dx_fine = cfg["domain_length"] / n_fine
    x_fine = (np.arange(n_fine) + 0.5) * dx_fine
    u_fine = ic(x_fine)
    fine_states = [u_fine.copy()]
    for _ in range(n_eval_steps):
        u_fine = advance_fine_solution(u_fine, dx_fine, cfg["dt_large"], pde, cfg.get("fine_cfl", 0.4))
        fine_states.append(u_fine.copy())
    return x_fine, np.array(fine_states)


def run_nn_rollout(net: ConservativeFluxNet, cfg: dict, u0_coarse: np.ndarray, n_eval_steps: int):
    dx, dt = cfg["domain_length"] / cfg["n_coarse"], cfg["dt_large"]
    u = torch.tensor(u0_coarse, dtype=torch.float32).unsqueeze(0)
    states = [u0_coarse.copy()]
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_eval_steps):
            u = net(u, dx=dx, dt=dt)
            states.append(u.numpy()[0].copy())
    elapsed = time.perf_counter() - t0
    return np.array(states), elapsed


def run_classical_matched_steps(pde, dx: float, dt_large: float, n_eval_steps: int, u0: np.ndarray,
                                 scheme: str = "weno5", cfl: float = 0.4):
    """Run a classical scheme on the SAME reporting cadence as the neural
    solver (n_eval_steps jumps of dt_large each, internally subdivided into
    CFL-stable substeps) so error/conservation curves are directly comparable
    at identical checkpoints."""
    u = u0.copy()
    states = [u0.copy()]
    for _ in range(n_eval_steps):
        u = advance_to_time(u, dx, dt_large, pde, scheme=scheme, cfl=cfl)
        states.append(u.copy())
    return np.array(states)


def run_classical_at_resolution(pde, domain_length: float, ic, scheme: str, n_cells: int, total_time: float):
    dx = domain_length / n_cells
    x = (np.arange(n_cells) + 0.5) * dx
    u0 = ic(x)
    dt_stable = max_stable_dt(u0, dx, pde, cfl=0.5)
    n_steps = max(1, int(np.ceil(total_time / dt_stable)))
    dt_actual = total_time / n_steps
    t0 = time.perf_counter()
    traj = run_classical(u0, dx, dt_actual, n_steps, pde, scheme=scheme)
    elapsed = time.perf_counter() - t0
    return x, traj, elapsed


def benchmark_pde(pde_name: str, checkpoint_path: str, n_eval_steps: int = 32,
                   n_test_ics: int = 6, test_seed: int = 999):
    net, cfg = load_checkpoint(checkpoint_path)
    pde = PDE_BUILDERS[pde_name](cfg)
    rng = np.random.default_rng(test_seed)

    domain_length, n_coarse, downsample_ratio = cfg["domain_length"], cfg["n_coarse"], cfg["downsample_ratio"]
    dx_coarse, dt_large = domain_length / n_coarse, cfg["dt_large"]
    total_time = n_eval_steps * dt_large
    step_times = np.arange(n_eval_steps + 1) * dt_large

    nn_final_errors, nn_times, nn_error_trajs, nn_conservation_trajs, effective_cfls = [], [], [], [], []
    weno_coarse_error_trajs, weno_coarse_conservation_trajs = [], []
    sweep_errors = {s: {n: [] for n in CLASSICAL_RESOLUTIONS} for s in CLASSICAL_SCHEMES}
    sweep_times = {s: {n: [] for n in CLASSICAL_RESOLUTIONS} for s in CLASSICAL_SCHEMES}
    snapshot = None

    for ic_idx in range(n_test_ics):
        ic = random_fourier_ic(domain_length, cfg.get("n_fourier_modes", 6), rng)
        x_fine, fine_traj = fine_reference_trajectory(pde, cfg, ic, n_eval_steps)
        ref_coarse_traj = block_average(fine_traj, downsample_ratio)

        nn_states, nn_time = run_nn_rollout(net, cfg, ref_coarse_traj[0], n_eval_steps)
        nn_error_traj = np.array(
            [relative_l2_error(nn_states[t], ref_coarse_traj[t]) for t in range(n_eval_steps + 1)]
        )
        nn_final_errors.append(nn_error_traj[-1])
        nn_error_trajs.append(nn_error_traj)
        nn_times.append(nn_time)
        nn_conservation_trajs.append(conservation_drift(nn_states, dx_coarse))
        effective_cfls.append(effective_cfl(dt_large, dx_coarse, pde, ref_coarse_traj[0], cfl=0.5))

        weno_coarse_traj = run_classical_matched_steps(
            pde, dx_coarse, dt_large, n_eval_steps, ref_coarse_traj[0], scheme="weno5", cfl=0.4
        )
        weno_coarse_error_trajs.append(
            np.array([relative_l2_error(weno_coarse_traj[t], ref_coarse_traj[t]) for t in range(n_eval_steps + 1)])
        )
        weno_coarse_conservation_trajs.append(conservation_drift(weno_coarse_traj, dx_coarse))

        for scheme in CLASSICAL_SCHEMES:
            for n_cells in CLASSICAL_RESOLUTIONS:
                ratio = fine_traj.shape[-1] // n_cells
                ref_final_at_res = block_average(fine_traj[-1], ratio)
                _, traj, elapsed = run_classical_at_resolution(
                    pde, domain_length, ic, scheme, n_cells, total_time
                )
                sweep_errors[scheme][n_cells].append(relative_l2_error(traj[-1], ref_final_at_res))
                sweep_times[scheme][n_cells].append(elapsed)

        if ic_idx == 0:
            _, upwind_coarse_traj, _ = run_classical_at_resolution(
                pde, domain_length, ic, "upwind", n_coarse, total_time
            )
            snapshot = {
                "x_fine": x_fine.tolist(),
                "u_fine_final": fine_traj[-1].tolist(),
                "x_coarse": ((np.arange(n_coarse) + 0.5) * dx_coarse).tolist(),
                "u_nn_final": nn_states[-1].tolist(),
                "u_upwind_final": upwind_coarse_traj[-1].tolist(),
            }

    nn_point = {"error": float(np.mean(nn_final_errors)), "time": float(np.mean(nn_times))}
    sweep_summary = {
        scheme: {
            "times": [float(np.mean(sweep_times[scheme][n])) for n in CLASSICAL_RESOLUTIONS],
            "errors": [float(np.mean(sweep_errors[scheme][n])) for n in CLASSICAL_RESOLUTIONS],
        }
        for scheme in CLASSICAL_SCHEMES
    }
    weno_times = np.array(sweep_summary["weno5"]["times"])
    weno_errors = np.array(sweep_summary["weno5"]["errors"])
    speedup = speedup_at_matched_accuracy(nn_point["error"], nn_point["time"], weno_errors, weno_times)

    results = {
        "pde": pde_name,
        "n_eval_steps": n_eval_steps,
        "dt_large": dt_large,
        "n_coarse": n_coarse,
        "n_test_ics": n_test_ics,
        "nn_point": nn_point,
        "classical_sweep": sweep_summary,
        "speedup_at_matched_accuracy_vs_weno5": None if np.isnan(speedup) else float(speedup),
        "effective_cfl": float(np.mean(effective_cfls)),
        "step_times": step_times.tolist(),
        "nn_error_trajectory": np.mean(np.stack(nn_error_trajs), axis=0).tolist(),
        "weno5_coarse_error_trajectory": np.mean(np.stack(weno_coarse_error_trajs), axis=0).tolist(),
        "conservation_drift_nn": np.mean(np.stack(nn_conservation_trajs), axis=0).tolist(),
        "conservation_drift_weno5_coarse": np.mean(np.stack(weno_coarse_conservation_trajs), axis=0).tolist(),
    }
    return results, snapshot


def make_all_plots(results: dict, snapshot: dict, figures_dir: Path) -> None:
    pde_label = results["pde"].capitalize()
    step_times = np.array(results["step_times"])

    make_plots.plot_pareto(
        results["classical_sweep"], results["nn_point"], pde_label,
        str(figures_dir / f"{results['pde']}_pareto.png"),
    )
    weno_label = f"WENO5 @ N={results['n_coarse']}"
    make_plots.plot_error_growth(
        step_times,
        {
            "nn": {"errors": np.array(results["nn_error_trajectory"]), "color": make_plots.COLOR_NN,
                   "label": "ConservativeFluxNet"},
            "weno5": {"errors": np.array(results["weno5_coarse_error_trajectory"]),
                      "color": make_plots.COLOR_WENO5, "label": weno_label},
        },
        pde_label, str(figures_dir / f"{results['pde']}_error_growth.png"),
    )
    make_plots.plot_conservation_drift(
        step_times,
        {
            "nn": {"drift": np.array(results["conservation_drift_nn"]), "color": make_plots.COLOR_NN,
                   "label": "ConservativeFluxNet"},
            "weno5": {"drift": np.array(results["conservation_drift_weno5_coarse"]),
                      "color": make_plots.COLOR_WENO5, "label": weno_label},
        },
        pde_label, str(figures_dir / f"{results['pde']}_conservation.png"),
    )
    make_plots.plot_snapshot(
        np.array(snapshot["x_fine"]), np.array(snapshot["u_fine_final"]),
        np.array(snapshot["x_coarse"]), np.array(snapshot["u_nn_final"]), np.array(snapshot["u_upwind_final"]),
        f"Upwind @ N={results['n_coarse']}", pde_label,
        str(figures_dir / f"{results['pde']}_snapshot.png"),
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    figures_dir = repo_root / "docs" / "report" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_path = repo_root / "results" / "metrics.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for pde_name, checkpoint in [
        ("advection", repo_root / "outputs" / "advection_cfn.pt"),
        ("burgers", repo_root / "outputs" / "burgers_cfn.pt"),
    ]:
        print(f"benchmarking {pde_name} ...")
        results, snapshot = benchmark_pde(pde_name, str(checkpoint))
        make_all_plots(results, snapshot, figures_dir)
        all_results[pde_name] = results
        summary = {k: v for k, v in results.items() if not k.endswith("trajectory") and "drift" not in k}
        print(json.dumps(summary, indent=2))

    results_path.write_text(json.dumps(all_results, indent=2))
    print(f"wrote {results_path}")


if __name__ == "__main__":
    main()
