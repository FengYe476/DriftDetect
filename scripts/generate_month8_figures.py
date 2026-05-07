"""Generate Month 8 publication figures.

Loads paired baseline and SHARP rollout .npz files for Cheetah V3, then writes
four PNGs to results/figures/.

Figures
-------
1. fig1_obs_trajectories.png  : 4x2 panel of selected observation dims (seed 0).
2. fig2_mse_over_time.png     : per-step position and velocity MSE, mean +- std.
3. fig3_latent_drift.png      : per-step latent (deter) MSE, mean +- std.
4. fig4_cross_task_arch.png   : J_total reduction bar chart across tasks/archs.

Run from the repo root on the host where the rollouts live (RunPod):
    python scripts/generate_month8_figures.py
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = REPO_ROOT / "results" / "month8_controlled" / "rollouts_drifthead_nocorr"
SHARP_DIR = REPO_ROOT / "results" / "month8_controlled" / "rollouts_moment_match"
FIG_DIR = REPO_ROOT / "results" / "figures"

SEEDS = list(range(20))
N_STEPS = 200
ACTOR_HORIZON = 15

POS_DIMS = list(range(0, 8))
VEL_DIMS = list(range(8, 17))
DETER_SLICE = slice(1024, 1536)

OBS_PANELS = [
    (0, "root_x"),
    (1, "root_z"),
    (2, "thigh"),
    (4, "foot"),
    (8, "root_x_vel"),
    (9, "root_z_vel"),
    (10, "thigh_vel"),
    (12, "foot_vel"),
]

COLOR_TRUE = "black"
COLOR_BASELINE = "#E53935"
COLOR_SHARP = "#1E88E5"

LW_TRUE = 2.0
LW_BASE = 1.5
LW_SHARP = 1.5


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.color": "lightgray",
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
    })


def rollout_path(directory: Path, seed: int) -> Path:
    return directory / f"cheetah_v3_seed{seed}_v2.npz"


def load_seed(directory: Path, seed: int) -> dict[str, np.ndarray] | None:
    path = rollout_path(directory, seed)
    if not path.exists():
        return None
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def load_paired_seeds() -> tuple[list[int], list[dict[str, np.ndarray]], list[dict[str, np.ndarray]]]:
    seeds: list[int] = []
    baseline: list[dict[str, np.ndarray]] = []
    sharp: list[dict[str, np.ndarray]] = []
    for seed in SEEDS:
        b = load_seed(BASELINE_DIR, seed)
        s = load_seed(SHARP_DIR, seed)
        if b is None or s is None:
            continue
        seeds.append(seed)
        baseline.append(b)
        sharp.append(s)
    if not seeds:
        raise FileNotFoundError(
            f"No paired rollouts found under {BASELINE_DIR} and {SHARP_DIR}"
        )
    return seeds, baseline, sharp


def stack_field(rollouts: list[dict[str, np.ndarray]], field: str) -> np.ndarray:
    return np.stack([r[field][:N_STEPS] for r in rollouts], axis=0)


def add_horizon_line(ax, label: str | None = "Actor horizon") -> None:
    ax.axvline(ACTOR_HORIZON, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    if label is not None:
        ymin, ymax = ax.get_ylim()
        ax.text(
            ACTOR_HORIZON + 1.5,
            ymin + 0.92 * (ymax - ymin),
            label,
            color="gray",
            fontsize=9,
            alpha=0.85,
        )


def figure1_obs_trajectories(seed_data_baseline: dict, seed_data_sharp: dict) -> None:
    true_obs = seed_data_sharp["true_obs"][:N_STEPS]
    base_obs = seed_data_baseline["imagined_obs"][:N_STEPS]
    sharp_obs = seed_data_sharp["imagined_obs"][:N_STEPS]
    steps = np.arange(N_STEPS)

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
    axes = axes.flatten()

    for ax, (dim, name) in zip(axes, OBS_PANELS):
        ax.plot(steps, true_obs[:, dim], color=COLOR_TRUE, linewidth=LW_TRUE, label="True")
        ax.plot(steps, base_obs[:, dim], color=COLOR_BASELINE, linewidth=LW_BASE,
                linestyle="--", label="Baseline")
        ax.plot(steps, sharp_obs[:, dim], color=COLOR_SHARP, linewidth=LW_SHARP, label="SHARP")
        ax.set_title(f"{name} (dim {dim})")
        ax.axvline(ACTOR_HORIZON, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)

    for ax in axes[-4:]:
        ax.set_xlabel("Imagination step")
    for ax in axes[::4]:
        ax.set_ylabel("Value")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Long-Horizon Imagination: True vs Baseline vs SHARP", y=1.05)

    out = FIG_DIR / "fig1_obs_trajectories.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


def figure2_mse_over_time(baseline: list[dict], sharp: list[dict]) -> None:
    true_obs = stack_field(sharp, "true_obs")
    base_imag = stack_field(baseline, "imagined_obs")
    sharp_imag = stack_field(sharp, "imagined_obs")

    base_se = (base_imag - true_obs) ** 2
    sharp_se = (sharp_imag - true_obs) ** 2

    pos_base = base_se[:, :, POS_DIMS].mean(axis=2)
    pos_sharp = sharp_se[:, :, POS_DIMS].mean(axis=2)
    vel_base = base_se[:, :, VEL_DIMS].mean(axis=2)
    vel_sharp = sharp_se[:, :, VEL_DIMS].mean(axis=2)

    steps = np.arange(N_STEPS)
    fig, (ax_pos, ax_vel) = plt.subplots(1, 2, figsize=(13, 5))

    def plot_band(ax, data: np.ndarray, color: str, label: str, linestyle: str = "-") -> None:
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        ax.plot(steps, mean, color=color, linewidth=1.6, linestyle=linestyle, label=label)
        ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)

    plot_band(ax_pos, pos_base, COLOR_BASELINE, "Baseline", linestyle="--")
    plot_band(ax_pos, pos_sharp, COLOR_SHARP, "SHARP")
    ax_pos.set_title("Position MSE per step (dims 0-7)")
    ax_pos.set_xlabel("Imagination step")
    ax_pos.set_ylabel("MSE")
    ax_pos.axvline(ACTOR_HORIZON, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax_pos.legend(loc="upper left", frameon=False)

    plot_band(ax_vel, vel_base, COLOR_BASELINE, "Baseline", linestyle="--")
    plot_band(ax_vel, vel_sharp, COLOR_SHARP, "SHARP")
    ax_vel.set_title("Velocity MSE per step (dims 8-16)")
    ax_vel.set_xlabel("Imagination step")
    ax_vel.set_ylabel("MSE")
    ax_vel.axvline(ACTOR_HORIZON, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax_vel.legend(loc="upper left", frameon=False)

    n_seeds = len(baseline)
    fig.suptitle(f"Per-Step Observation MSE (mean +- std over {n_seeds} seeds)", y=1.02)

    out = FIG_DIR / "fig2_mse_over_time.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


def figure3_latent_drift(baseline: list[dict], sharp: list[dict]) -> None:
    def deter_mse(rollouts: list[dict]) -> np.ndarray:
        true_lat = stack_field(rollouts, "true_latent")[:, :, DETER_SLICE]
        imag_lat = stack_field(rollouts, "imagined_latent")[:, :, DETER_SLICE]
        return ((imag_lat - true_lat) ** 2).mean(axis=2)

    base_mse = deter_mse(baseline)
    sharp_mse = deter_mse(sharp)

    steps = np.arange(N_STEPS)
    fig, ax = plt.subplots(figsize=(10, 5.5))

    def band(data: np.ndarray, color: str, label: str, linestyle: str = "-") -> None:
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        ax.plot(steps, mean, color=color, linewidth=1.7, linestyle=linestyle, label=label)
        ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)

    band(base_mse, COLOR_BASELINE, "Baseline DreamerV3", linestyle="--")
    band(sharp_mse, COLOR_SHARP, "SHARP")

    ax.set_xlabel("Imagination step")
    ax.set_ylabel("Latent MSE (deter dims 1024:1536)")
    n_seeds = len(baseline)
    ax.set_title(f"Latent Drift Accumulation: Baseline vs SHARP (mean +- std, {n_seeds} seeds)")
    ax.axvline(ACTOR_HORIZON, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)

    ymin, ymax = ax.get_ylim()
    ax.text(
        ACTOR_HORIZON + 2.0,
        ymin + 0.93 * (ymax - ymin),
        "Actor horizon",
        color="gray",
        fontsize=9,
        alpha=0.85,
    )

    sharp_mean = sharp_mse.mean(axis=0)
    base_mean = base_mse.mean(axis=0)
    target = 150
    arrow_x = target
    arrow_y_target = sharp_mean[target]
    arrow_y_text = base_mean[target] * 0.55
    ax.annotate(
        "J_total: -89.6%",
        xy=(arrow_x, arrow_y_target),
        xytext=(arrow_x - 55, arrow_y_text),
        fontsize=11,
        color=COLOR_SHARP,
        arrowprops=dict(
            arrowstyle="->",
            color=COLOR_SHARP,
            lw=1.2,
            connectionstyle="arc3,rad=-0.15",
        ),
    )

    ax.legend(loc="upper left", frameon=False)

    out = FIG_DIR / "fig3_latent_drift.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


def figure4_cross_task_arch() -> None:
    labels = ["V3 Cheetah", "V3 Cartpole", "V4 Cheetah"]
    arch = ["RSSM", "RSSM", "Transformer"]
    values = [-89.6, -97.2, -64.1]
    colors = ["#1E88E5", "#43A047", "#FB8C00"]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.6, width=0.6)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value - 3.0,
            f"{value:.1f}%",
            ha="center",
            va="top",
            color="white",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("J_total Reduction (%)")
    ax.set_title("SHARP: Cross-Task and Cross-Architecture Drift Reduction")
    ax.set_ylim(-110, 5)
    ax.axhline(0, color="black", linewidth=0.8)

    for xi, label in zip(x, arch):
        ax.text(xi, 2.0, label, ha="center", va="bottom", fontsize=10, color="dimgray")

    ax.grid(axis="y", color="lightgray", alpha=0.3)
    ax.grid(axis="x", visible=False)

    out = FIG_DIR / "fig4_cross_task_arch.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    configure_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    seeds, baseline, sharp = load_paired_seeds()
    print(f"loaded {len(seeds)} paired seeds: {seeds}")

    if 0 in seeds:
        idx0 = seeds.index(0)
    else:
        idx0 = 0
        print(f"seed 0 not available, using seed {seeds[0]} for figure 1")
    figure1_obs_trajectories(baseline[idx0], sharp[idx0])

    figure2_mse_over_time(baseline, sharp)
    figure3_latent_drift(baseline, sharp)
    figure4_cross_task_arch()


if __name__ == "__main__":
    main()
