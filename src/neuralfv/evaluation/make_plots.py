"""Publication-quality static PNG figures for the README and write-up.

Palette and mark specs follow the project's dataviz guidelines: fixed
categorical hue order (never cycled), 2px lines, >=8px end markers, direct
end-labels instead of a legend box, recessive hairline gridlines, and text
kept in ink tones rather than series colors.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

COLOR_UPWIND = "#2a78d6"  # categorical slot 1 (blue)
COLOR_MUSCL = "#eb6834"  # categorical slot 2 (orange)
COLOR_WENO5 = "#1baf7a"  # categorical slot 3 (aqua)
COLOR_NN = "#e34948"  # categorical slot 8 (red) -- reserved for the highlighted NN point
COLOR_REFERENCE = INK_PRIMARY

SCHEME_COLORS = {"upwind": COLOR_UPWIND, "muscl": COLOR_MUSCL, "weno5": COLOR_WENO5}
SCHEME_LABELS = {"upwind": "Upwind", "muscl": "MUSCL", "weno5": "WENO5"}


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(True, which="major", color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)


def plot_pareto(classical_sweep: dict, nn_point: dict, pde_name: str, save_path: str) -> None:
    """classical_sweep: {scheme_name: {"times": [...], "errors": [...]}}
    nn_point: {"time": float, "error": float}
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=150)
    _style_axes(ax)

    for scheme, data in classical_sweep.items():
        times, errors = np.array(data["times"]), np.array(data["errors"])
        order = np.argsort(times)
        ax.plot(
            times[order], errors[order],
            color=SCHEME_COLORS[scheme], linewidth=2, marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1, zorder=3,
        )
        ax.annotate(
            SCHEME_LABELS[scheme], xy=(times[order][-1], errors[order][-1]),
            xytext=(6, 0), textcoords="offset points", color=INK_SECONDARY,
            fontsize=9, va="center",
        )

    ax.scatter(
        [nn_point["time"]], [nn_point["error"]], marker="*", s=260,
        color=COLOR_NN, edgecolor=SURFACE, linewidth=1, zorder=5,
    )
    ax.annotate(
        "ConservativeFluxNet", xy=(nn_point["time"], nn_point["error"]),
        xytext=(8, 10), textcoords="offset points", color=INK_PRIMARY,
        fontsize=9, fontweight="bold",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Wall-clock time per rollout (s)")
    ax.set_ylabel("Relative L2 error vs. reference")
    ax.set_title(f"Accuracy vs. cost — {pde_name}", color=INK_PRIMARY, fontsize=12, loc="left")
    fig.tight_layout()
    fig.savefig(save_path, facecolor=SURFACE)
    plt.close(fig)


def plot_error_growth(step_times: np.ndarray, series: dict, pde_name: str, save_path: str) -> None:
    """series: {name: {"errors": array, "color": hex, "label": str}}"""
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    _style_axes(ax)

    for _name, data in series.items():
        ax.plot(step_times, data["errors"], color=data["color"], linewidth=2, zorder=3)
        ax.annotate(
            data["label"], xy=(step_times[-1], data["errors"][-1]), xytext=(6, 0),
            textcoords="offset points", color=INK_SECONDARY, fontsize=9, va="center",
        )

    ax.set_yscale("log")
    ax.set_xlabel("Simulated time")
    ax.set_ylabel("Relative L2 error vs. reference")
    ax.set_title(f"Rollout error growth — {pde_name}", color=INK_PRIMARY, fontsize=12, loc="left")
    fig.tight_layout()
    fig.savefig(save_path, facecolor=SURFACE)
    plt.close(fig)


def plot_conservation_drift(step_times: np.ndarray, series: dict, pde_name: str, save_path: str) -> None:
    """series: {name: {"drift": array, "color": hex, "label": str}}"""
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    _style_axes(ax)

    for _name, data in series.items():
        drift = np.maximum(data["drift"], 1e-17)  # avoid log(0)
        ax.plot(step_times, drift, color=data["color"], linewidth=2, zorder=3)
        ax.annotate(
            data["label"], xy=(step_times[-1], drift[-1]), xytext=(6, 0),
            textcoords="offset points", color=INK_SECONDARY, fontsize=9, va="center",
        )

    ax.set_yscale("log")
    ax.set_xlabel("Simulated time")
    ax.set_ylabel("|mass(t) - mass(0)|")
    ax.set_title(f"Conservation drift — {pde_name}", color=INK_PRIMARY, fontsize=12, loc="left")
    fig.tight_layout()
    fig.savefig(save_path, facecolor=SURFACE)
    plt.close(fig)


def plot_snapshot(
    x_reference: np.ndarray, u_reference: np.ndarray,
    x_coarse: np.ndarray, u_nn: np.ndarray, u_baseline: np.ndarray,
    baseline_label: str, pde_name: str, save_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    _style_axes(ax)

    ax.plot(x_reference, u_reference, color=COLOR_REFERENCE, linewidth=2, zorder=2, label="Reference (fine WENO5)")
    ax.plot(x_coarse, u_baseline, color=COLOR_UPWIND, linewidth=2, marker="o", markersize=5,
            markeredgecolor=SURFACE, zorder=3, label=baseline_label)
    ax.plot(x_coarse, u_nn, color=COLOR_NN, linewidth=2, marker="*", markersize=8,
            markeredgecolor=SURFACE, zorder=4, label="ConservativeFluxNet")

    ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9, loc="best")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_title(f"Solution snapshot — {pde_name}", color=INK_PRIMARY, fontsize=12, loc="left")
    fig.tight_layout()
    fig.savefig(save_path, facecolor=SURFACE)
    plt.close(fig)
