"""Precompute NN-vs-classical solver trajectories for the interactive demo.

Precomputed (not live in-browser) so the demo's numbers always exactly match
what's reported in the README/write-up, and so it works with zero runtime ML
dependency in the browser -- see docs/report/report.md's "Demo design"
section for the full rationale.

Usage: PYTHONPATH=src python demo/export_demo_payloads.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from neuralfv.data.generate_dataset import block_average, random_fourier_ic
from neuralfv.evaluation.benchmark import (
    PDE_BUILDERS,
    fine_reference_trajectory,
    load_checkpoint,
    run_classical_matched_steps,
    run_nn_rollout,
)
from neuralfv.evaluation.metrics import relative_l2_error

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIO_SEEDS = {"advection": [101, 102], "burgers": [201, 202]}
N_EVAL_STEPS = 32


def build_scenario(pde_name: str, checkpoint_path: Path, seed: int) -> dict:
    net, cfg = load_checkpoint(str(checkpoint_path))
    pde = PDE_BUILDERS[pde_name](cfg)
    domain_length, n_coarse, downsample_ratio = cfg["domain_length"], cfg["n_coarse"], cfg["downsample_ratio"]
    dx_coarse, dt_large = domain_length / n_coarse, cfg["dt_large"]

    rng = np.random.default_rng(seed)
    ic = random_fourier_ic(domain_length, cfg.get("n_fourier_modes", 6), rng)
    _, fine_traj = fine_reference_trajectory(pde, cfg, ic, N_EVAL_STEPS)
    reference = block_average(fine_traj, downsample_ratio)

    nn_states, nn_time = run_nn_rollout(net, cfg, reference[0], N_EVAL_STEPS)
    upwind_states = run_classical_matched_steps(pde, dx_coarse, dt_large, N_EVAL_STEPS, reference[0], "upwind")
    weno5_states = run_classical_matched_steps(pde, dx_coarse, dt_large, N_EVAL_STEPS, reference[0], "weno5")

    x_coarse = ((np.arange(n_coarse) + 0.5) * dx_coarse).tolist()
    step_times = (np.arange(N_EVAL_STEPS + 1) * dt_large).tolist()

    return {
        "id": f"{pde_name}_{seed}",
        "pde": pde_name,
        "label": f"{pde_name.capitalize()} — seed {seed}",
        "x_coarse": x_coarse,
        "step_times": step_times,
        "series": {
            "reference": reference.tolist(),
            "nn": nn_states.tolist(),
            "upwind": upwind_states.tolist(),
            "weno5": weno5_states.tolist(),
        },
        "final_errors": {
            "nn": relative_l2_error(nn_states[-1], reference[-1]),
            "upwind": relative_l2_error(upwind_states[-1], reference[-1]),
            "weno5": relative_l2_error(weno5_states[-1], reference[-1]),
        },
        "nn_wall_time_seconds": nn_time,
    }


def main() -> None:
    scenarios = []
    for pde_name, seeds in SCENARIO_SEEDS.items():
        checkpoint = REPO_ROOT / "outputs" / f"{pde_name}_cfn.pt"
        for seed in seeds:
            print(f"building scenario {pde_name}/{seed} ...")
            scenarios.append(build_scenario(pde_name, checkpoint, seed))

    data_dir = REPO_ROOT / "demo" / "web" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "scenarios.json").write_text(json.dumps({"scenarios": scenarios}))

    metrics_path = REPO_ROOT / "results" / "metrics.json"
    if metrics_path.exists():
        (data_dir / "pareto.json").write_text(metrics_path.read_text())

    print(f"wrote {data_dir / 'scenarios.json'}")


if __name__ == "__main__":
    main()
