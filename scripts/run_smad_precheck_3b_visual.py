#!/usr/bin/env python3
"""SMAD pre-check 3b: visual sanity check for SFA/posterior/drift bases."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/driftdetect_matplotlib_cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from run_smad_precheck_3a_sfa import (  # noqa: E402
    DRIFT_TRIM_END_INCLUSIVE,
    DRIFT_TRIM_START,
    HORIZON,
    N_SEEDS,
    REPO_ROOT,
    ROLLOUT_DIR,
    VAR_THRESHOLD,
    discover_rollouts,
    fit_drift_basis,
    fit_posterior_basis,
    fit_sfa_basis,
    load_deter_rollouts,
    relative_path,
    resolve_path,
)


RANK = 3
FIGURE_DIR = REPO_ROOT / "results" / "figures"
VISUAL_OUTPUT = FIGURE_DIR / "smad_precheck_3b_visual.pdf"
DERIV_OUTPUT = FIGURE_DIR / "smad_precheck_3b_derivatives.pdf"


def main() -> None:
    args = parse_args()
    rollout_paths = discover_rollouts(args.rollout_dir)
    true_deter_all, imag_deter_all = load_deter_rollouts(rollout_paths)

    sfa_fit = fit_sfa_basis(
        true_deter_all,
        ranks=(RANK,),
        var_threshold=args.var_threshold,
    )
    posterior = fit_posterior_basis(
        true_deter_all,
        ranks=(RANK,),
        var_threshold=args.var_threshold,
    )
    drift = fit_drift_basis(
        true_deter_all,
        imag_deter_all,
        ranks=(RANK,),
        trim_start=args.trim_start,
        trim_end_inclusive=args.trim_end_inclusive,
    )

    bases = {
        "SFA": sfa_fit.bases_by_rank[RANK],
        "Posterior": posterior["bases_by_rank"][RANK],
        "Drift": drift["bases_by_rank"][RANK],
    }

    true_deter = true_deter_all[args.seed]
    imag_deter = imag_deter_all[args.seed]

    args.visual_output.parent.mkdir(parents=True, exist_ok=True)
    args.derivatives_output.parent.mkdir(parents=True, exist_ok=True)
    plot_projection_grid(
        true_deter,
        imag_deter,
        bases,
        output=args.visual_output,
    )
    derivative_variance = compute_derivative_variances(true_deter, bases)
    plot_derivative_variance(
        derivative_variance,
        output=args.derivatives_output,
    )

    print_summary(
        args=args,
        rollout_paths=rollout_paths,
        sfa_fit=sfa_fit,
        posterior=posterior,
        drift=drift,
        derivative_variance=derivative_variance,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot seed-0 temporal projections for U_sfa, U_posterior, and "
            "U_drift, plus first-derivative variance bars."
        )
    )
    parser.add_argument("--rollout-dir", type=Path, default=ROLLOUT_DIR)
    parser.add_argument("--visual-output", type=Path, default=VISUAL_OUTPUT)
    parser.add_argument("--derivatives-output", type=Path, default=DERIV_OUTPUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--var-threshold", type=float, default=VAR_THRESHOLD)
    parser.add_argument("--trim-start", type=int, default=DRIFT_TRIM_START)
    parser.add_argument(
        "--trim-end-inclusive",
        type=int,
        default=DRIFT_TRIM_END_INCLUSIVE,
    )
    args = parser.parse_args()
    if not 0 <= args.seed < N_SEEDS:
        raise ValueError(f"--seed must be in [0, {N_SEEDS - 1}].")
    if not 0 < args.var_threshold <= 1:
        raise ValueError("--var-threshold must be in (0, 1].")
    if not 0 <= args.trim_start <= args.trim_end_inclusive < HORIZON:
        raise ValueError("--trim-start/--trim-end-inclusive must be inside horizon.")

    args.rollout_dir = resolve_path(args.rollout_dir)
    args.visual_output = resolve_path(args.visual_output)
    args.derivatives_output = resolve_path(args.derivatives_output)
    return args


def plot_projection_grid(
    true_deter: np.ndarray,
    imag_deter: np.ndarray,
    bases: dict[str, np.ndarray],
    *,
    output: Path,
) -> None:
    steps = np.arange(true_deter.shape[0])
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True)
    colors = {
        "SFA": "#2c7fb8",
        "Posterior": "#f28e2b",
        "true": "#1f77b4",
        "imagined": "#d62728",
    }

    for col in range(RANK):
        projection = true_deter @ bases["SFA"][:, col]
        ax = axes[0, col]
        ax.plot(steps, projection, color=colors["SFA"], linewidth=1.8)
        ax.set_title(f"SFA dir {col}")
        ax.set_ylabel("projection")
        ax.grid(alpha=0.25)

    for col in range(RANK):
        projection = true_deter @ bases["Posterior"][:, col]
        ax = axes[1, col]
        ax.plot(steps, projection, color=colors["Posterior"], linewidth=1.8)
        ax.set_title(f"Posterior dir {col}")
        ax.set_ylabel("projection")
        ax.grid(alpha=0.25)

    for col in range(RANK):
        true_projection = true_deter @ bases["Drift"][:, col]
        imag_projection = imag_deter @ bases["Drift"][:, col]
        ax = axes[2, col]
        ax.plot(steps, true_projection, color=colors["true"], linewidth=1.8, label="true")
        ax.plot(
            steps,
            imag_projection,
            color=colors["imagined"],
            linewidth=1.8,
            label="imagined",
        )
        ax.set_title(f"Drift dir {col}")
        ax.set_xlabel("step")
        ax.set_ylabel("projection")
        ax.legend(frameon=False, fontsize=9)
        ax.grid(alpha=0.25)

    for ax in axes.flat:
        ax.set_xlim(0, HORIZON - 1)

    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def compute_derivative_variances(
    true_deter: np.ndarray,
    bases: dict[str, np.ndarray],
) -> dict[str, list[float]]:
    result = {}
    for name, basis in bases.items():
        projections = true_deter @ basis
        derivative = np.diff(projections, axis=0)
        result[name] = np.var(derivative, axis=0, ddof=1).tolist()
    return result


def plot_derivative_variance(
    derivative_variance: dict[str, list[float]],
    *,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    basis_names = ["SFA", "Posterior", "Drift"]
    offsets = np.array([-0.24, 0.0, 0.24])
    x = np.arange(len(basis_names))
    colors = ["#4e79a7", "#59a14f", "#e15759"]

    for direction in range(RANK):
        heights = [derivative_variance[name][direction] for name in basis_names]
        ax.bar(
            x + offsets[direction],
            heights,
            width=0.22,
            color=colors[direction],
            label=f"dir {direction}",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(basis_names)
    ax.set_ylabel("first-derivative variance")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def print_summary(
    *,
    args: argparse.Namespace,
    rollout_paths: list[Path],
    sfa_fit: object,
    posterior: dict[str, object],
    drift: dict[str, object],
    derivative_variance: dict[str, list[float]],
) -> None:
    print("SMAD Pre-check 3b: visual basis sanity check")
    print(f"Rollouts: {len(rollout_paths)} Cheetah v2 files")
    print(f"Representative seed: {args.seed}")
    print(
        "Basis metadata: "
        f"SFA whitened k={sfa_fit.retained_k}, "
        f"posterior PCA k={posterior['retained_k']}, "
        f"drift pool={tuple(drift['drift_pool'].shape)}"
    )
    print()
    print("First-derivative variance of true_deter projections")
    print("basis       dir0        dir1        dir2")
    print("---------   ---------   ---------   ---------")
    for name in ["SFA", "Posterior", "Drift"]:
        values = derivative_variance[name]
        print(
            f"{name:<9}   "
            f"{values[0]:<9.6f}   "
            f"{values[1]:<9.6f}   "
            f"{values[2]:<9.6f}"
        )
    print()
    print(f"Saved projection grid: {relative_path(args.visual_output)}")
    print(f"Saved derivative bars: {relative_path(args.derivatives_output)}")


if __name__ == "__main__":
    main()
