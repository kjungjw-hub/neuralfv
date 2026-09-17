"""Config-driven training entrypoint.

Usage: python -m neuralfv.training.train --config src/neuralfv/training/configs/burgers_cfn.yaml

Every number quoted in the README/write-up should trace back to a run of
this script with a committed config -- that is what "reproducible" means
for this repo.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from neuralfv.data.datasets import RolloutWindowDataset
from neuralfv.data.generate_dataset import generate_rollout_dataset
from neuralfv.pdes.advection import LinearAdvection
from neuralfv.pdes.burgers import Burgers
from neuralfv.solvers.neural.flux_net import ConservativeFluxNet
from neuralfv.solvers.neural.losses import rollout_mse
from neuralfv.solvers.neural.rollout import rollout

PDE_BUILDERS = {
    "advection": lambda cfg: LinearAdvection(speed=cfg.get("advection_speed", 1.0)),
    "burgers": lambda cfg: Burgers(),
}


def build_pde(cfg: dict):
    return PDE_BUILDERS[cfg["pde"]](cfg)


def build_trajectories(cfg: dict, n_trajectories: int, seed: int) -> np.ndarray:
    return generate_rollout_dataset(
        pde=build_pde(cfg),
        domain_length=cfg["domain_length"],
        n_coarse=cfg["n_coarse"],
        downsample_ratio=cfg["downsample_ratio"],
        dt_large=cfg["dt_large"],
        n_large_steps=cfg["n_large_steps"],
        n_trajectories=n_trajectories,
        n_fourier_modes=cfg.get("n_fourier_modes", 6),
        fine_cfl=cfg.get("fine_cfl", 0.4),
        seed=seed,
    )


def train(cfg: dict) -> dict:
    torch.manual_seed(cfg.get("torch_seed", 0))

    train_traj = build_trajectories(cfg, cfg["n_train_trajectories"], cfg["seed_train"])
    val_traj = build_trajectories(cfg, cfg["n_val_trajectories"], cfg["seed_val"])

    net = ConservativeFluxNet(**cfg["model"])
    train_cfg = cfg["training"]
    optimizer = torch.optim.Adam(net.parameters(), lr=train_cfg["lr"])

    dx = cfg["domain_length"] / cfg["n_coarse"]
    dt = cfg["dt_large"]
    final_rollout = train_cfg["rollout_length_final"]
    curriculum_epochs = train_cfg["rollout_curriculum_epochs"]

    history = []
    t_start = time.time()
    for epoch in range(train_cfg["epochs"]):
        rollout_length = min(final_rollout, 1 + epoch // curriculum_epochs)
        train_loader = DataLoader(
            RolloutWindowDataset(train_traj, rollout_length),
            batch_size=train_cfg["batch_size"],
            shuffle=True,
        )
        val_dataset = RolloutWindowDataset(val_traj, rollout_length)

        net.train()
        train_losses = []
        for u0, targets in train_loader:
            preds = rollout(net, u0, dx=dx, dt=dt, n_steps=rollout_length, noise_std=train_cfg["noise_std"])
            loss = rollout_mse(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        net.eval()
        with torch.no_grad():
            u0_val = val_dataset.trajectories[:, 0]
            targets_val = val_dataset.trajectories[:, 1 : 1 + rollout_length]
            preds_val = rollout(net, u0_val, dx=dx, dt=dt, n_steps=rollout_length, noise_std=0.0)
            val_loss = rollout_mse(preds_val, targets_val).item()

        train_loss = float(np.mean(train_losses))
        history.append(
            {"epoch": epoch, "rollout_length": rollout_length, "train_loss": train_loss, "val_loss": val_loss}
        )
        print(f"epoch {epoch:3d}  rollout={rollout_length:2d}  train={train_loss:.3e}  val={val_loss:.3e}")

    elapsed = time.time() - t_start
    return {"net": net, "history": history, "elapsed_seconds": elapsed, "config": cfg}


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    result = train(cfg)

    out_cfg = cfg["output"]
    checkpoint_path = Path(out_cfg["checkpoint"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": result["net"].state_dict(), "model_cfg": cfg["model"], "config": cfg},
        checkpoint_path,
    )

    log_path = Path(out_cfg["log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_payload = {"history": result["history"], "elapsed_seconds": result["elapsed_seconds"]}
    log_path.write_text(json.dumps(log_payload, indent=2))

    print(f"done in {result['elapsed_seconds']:.1f}s -- checkpoint: {checkpoint_path}, log: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
