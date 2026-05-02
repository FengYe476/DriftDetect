#!/usr/bin/env python3
"""SMAD pre-check 3c: drift trim-window sensitivity for basis overlaps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from run_smad_precheck_3a_sfa import (
    HORIZON,
    LATENT_DIM,
    RANKS,
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
    subspace_overlap,
)


OUTPUT_PATH = REPO_ROOT / "results" / "tables" / "smad_precheck_3c_trim.json"
TRIM_WINDOWS = ((10, 190), (25, 175), (50, 150))
DETER_START = 1024
DETER_DIM = 512


def main() -> None:
    args = parse_args()
    rollout_paths = discover_rollouts(args.rollout_dir)
    true_deter, imag_deter = load_deter_rollouts(rollout_paths)

    sfa_fit = fit_sfa_basis(
        true_deter,
        ranks=args.ranks,
        var_threshold=args.var_threshold,
    )
    posterior = fit_posterior_basis(
        true_deter,
        ranks=args.ranks,
        var_threshold=args.var_threshold,
    )

    windows = {}
    for trim_start, trim_end in args.trim_windows:
        drift = fit_drift_basis(
            true_deter,
            imag_deter,
            ranks=args.ranks,
            trim_start=trim_start,
            trim_end_inclusive=trim_end,
        )
        window_key = format_window(trim_start, trim_end)
        windows[window_key] = {
            "trim_window_inclusive": [int(trim_start), int(trim_end)],
            "trimmed_steps_per_seed": int(drift["trimmed_steps_per_seed"]),
            "drift_pool_shape": list(drift["drift_pool"].shape),
            "drift_explained_variance_by_rank": {
                str(rank): {
                    "top_r_ratios": drift["pca"].explained_variance_ratio[
                        :rank
                    ].tolist(),
                    "cumulative": float(
                        np.sum(drift["pca"].explained_variance_ratio[:rank])
                    ),
                }
                for rank in args.ranks
            },
            "overlaps": compute_overlaps(
                ranks=args.ranks,
                sfa_bases=sfa_fit.bases_by_rank,
                posterior_bases=posterior["bases_by_rank"],
                drift_bases=drift["bases_by_rank"],
            ),
        }

    invariant = sfa_posterior_invariance(windows, args.ranks)
    result = {
        "analysis": "smad_precheck_3c_trim",
        "task": "cheetah_run",
        "rollout_dir": relative_path(args.rollout_dir),
        "rollout_files": [relative_path(path) for path in rollout_paths],
        "n_seeds": int(true_deter.shape[0]),
        "horizon": int(true_deter.shape[1]),
        "latent_dim": LATENT_DIM,
        "deter_dim": DETER_DIM,
        "deter_slice": [DETER_START, LATENT_DIM],
        "ranks": list(args.ranks),
        "trim_windows_inclusive": [list(window) for window in args.trim_windows],
        "var_threshold": float(args.var_threshold),
        "basis_estimation_scope": {
            "sfa": "Estimated once from full true posterior deter trajectories.",
            "posterior": "Estimated once from full true posterior deter trajectories.",
            "drift": "Re-estimated separately inside each trim window.",
        },
        "sfa": {
            "whitened_dimensions": int(sfa_fit.retained_k),
            "retained_variance_explained": float(
                np.sum(sfa_fit.pca.explained_variance_ratio[: sfa_fit.retained_k])
            ),
        },
        "posterior": {
            "retained_components": int(posterior["retained_k"]),
            "retained_variance_explained": float(
                np.sum(
                    posterior["pca"].explained_variance_ratio[
                        : posterior["retained_k"]
                    ]
                )
            ),
            "selected_pc_indices_by_rank": {
                str(rank): posterior["slow_order"][:rank].tolist()
                for rank in args.ranks
            },
        },
        "invariance_check": invariant,
        "windows": windows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    print_summary(result)
    print(f"\nSaved: {relative_path(args.output)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether low U_posterior/U_drift and U_sfa/U_drift overlaps "
            "depend on the drift PCA trim window."
        )
    )
    parser.add_argument("--rollout-dir", type=Path, default=ROLLOUT_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--var-threshold", type=float, default=VAR_THRESHOLD)
    parser.add_argument("--ranks", type=int, nargs="+", default=list(RANKS))
    args = parser.parse_args()
    if not 0 < args.var_threshold <= 1:
        raise ValueError("--var-threshold must be in (0, 1].")
    args.ranks = tuple(sorted(set(args.ranks)))
    if any(rank <= 0 or rank > DETER_DIM for rank in args.ranks):
        raise ValueError(f"--ranks must be in [1, {DETER_DIM}].")
    for trim_start, trim_end in TRIM_WINDOWS:
        if not 0 <= trim_start <= trim_end < HORIZON:
            raise ValueError(f"Invalid trim window [{trim_start}, {trim_end}].")
    args.trim_windows = TRIM_WINDOWS
    args.rollout_dir = resolve_path(args.rollout_dir)
    args.output = resolve_path(args.output)
    return args


def compute_overlaps(
    *,
    ranks: tuple[int, ...],
    sfa_bases: dict[int, np.ndarray],
    posterior_bases: dict[int, np.ndarray],
    drift_bases: dict[int, np.ndarray],
) -> dict[str, dict[str, float]]:
    overlaps = {}
    for rank in ranks:
        u_sfa = sfa_bases[rank]
        u_posterior = posterior_bases[rank]
        u_drift = drift_bases[rank]
        overlaps[str(rank)] = {
            "sfa_vs_posterior": subspace_overlap(u_sfa, u_posterior),
            "sfa_vs_drift": subspace_overlap(u_sfa, u_drift),
            "posterior_vs_drift": subspace_overlap(u_posterior, u_drift),
        }
    return overlaps


def sfa_posterior_invariance(
    windows: dict[str, dict[str, Any]],
    ranks: tuple[int, ...],
) -> dict[str, Any]:
    max_abs_deviation = 0.0
    by_rank = {}
    for rank in ranks:
        values = [
            window["overlaps"][str(rank)]["sfa_vs_posterior"]
            for window in windows.values()
        ]
        deviation = float(np.max(np.abs(np.asarray(values) - values[0])))
        by_rank[str(rank)] = {
            "values": [float(value) for value in values],
            "max_abs_deviation": deviation,
        }
        max_abs_deviation = max(max_abs_deviation, deviation)
    return {
        "sfa_vs_posterior_max_abs_deviation": float(max_abs_deviation),
        "sfa_vs_posterior_by_rank": by_rank,
        "passes": bool(max_abs_deviation <= 1e-12),
    }


def format_window(trim_start: int, trim_end: int) -> str:
    return f"[{trim_start},{trim_end}]"


def print_summary(result: dict[str, Any]) -> None:
    print("SMAD Pre-check 3c: trim window sensitivity")
    print(
        "Full-trajectory bases: "
        f"SFA k={result['sfa']['whitened_dimensions']}, "
        f"posterior PCA k={result['posterior']['retained_components']}"
    )
    invariant = result["invariance_check"]
    print(
        "SFA/posterior invariant across trim windows: "
        f"{'PASS' if invariant['passes'] else 'FAIL'} "
        f"(max deviation={invariant['sfa_vs_posterior_max_abs_deviation']:.3e})"
    )
    print()
    print_overlap_table(
        result,
        title="Trim window sensitivity: posterior/drift overlap",
        overlap_key="posterior_vs_drift",
    )
    print()
    print_overlap_table(
        result,
        title="Trim window sensitivity: SFA/drift overlap",
        overlap_key="sfa_vs_drift",
    )


def print_overlap_table(
    result: dict[str, Any],
    *,
    title: str,
    overlap_key: str,
) -> None:
    ranks = result["ranks"]
    print(title)
    print("          " + "   ".join(f"r={rank:<7}" for rank in ranks))
    for window_key, item in result["windows"].items():
        values = [
            item["overlaps"][str(rank)][overlap_key]
            for rank in ranks
        ]
        print(
            f"{window_key:<9} "
            + "   ".join(f"{value:<9.4f}" for value in values)
        )


if __name__ == "__main__":
    main()
