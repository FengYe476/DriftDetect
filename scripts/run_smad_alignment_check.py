#!/usr/bin/env python3
"""SMAD pre-check 1: posterior slow-subspace vs drift-subspace alignment."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ROLLOUT_DIR = REPO_ROOT / "results" / "rollouts"
OUTPUT_PATH = REPO_ROOT / "results" / "tables" / "smad_alignment_check.json"

ROLLOUT_RE = re.compile(r"cheetah_v3_seed(\d+)_v2\.npz$")
N_SEEDS = 20
HORIZON = 200
LATENT_DIM = 1536
DETER_START = 1024
DETER_DIM = 512
VAR_THRESHOLD = 0.95
DRIFT_TRIM = 25
DEFAULT_REPORTED_RANKS = (3, 5, 10)
DEFAULT_CURVE_RANKS = (1, 2, 3, 5, 10, 15, 20)


@dataclass(frozen=True)
class PCAFit:
    mean: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray
    singular_values: np.ndarray


def main() -> None:
    args = parse_args()
    rollout_paths = discover_rollouts(args.rollout_dir)
    true_deter, imag_deter = load_deter_rollouts(rollout_paths)

    posterior_pool = true_deter.reshape(-1, DETER_DIM)
    posterior_pca = fit_pca(posterior_pool)
    retained_k = n_components_for_variance(
        posterior_pca.explained_variance_ratio,
        threshold=args.var_threshold,
    )
    retained_components = posterior_pca.components[:retained_k]
    retained_variance = posterior_pca.explained_variance_ratio[:retained_k]

    projections = project_rollouts(true_deter, posterior_pca.mean, retained_components)
    autocorr = lag1_autocorr_by_pc(projections)
    slow_order = np.argsort(-autocorr)

    drift = imag_deter - true_deter
    drift_window = drift[:, args.trim : HORIZON - args.trim, :]
    drift_pool = drift_window.reshape(-1, DETER_DIM)
    drift_pca = fit_pca(drift_pool)

    max_rank = min(
        retained_k,
        drift_pca.components.shape[0],
        max(args.curve_ranks),
        DETER_DIM,
    )
    rank_results_raw = compute_rank_results(
        ranks=sorted(set(r for r in args.curve_ranks if r <= max_rank)),
        retained_components=retained_components,
        slow_order=slow_order,
        drift_components=drift_pca.components,
        drift_pool=drift_pool,
    )
    rank_results = {str(rank): values for rank, values in rank_results_raw.items()}

    reported_ranks = [rank for rank in args.reported_ranks if str(rank) in rank_results]
    drift_projection_r2 = drift_pool @ retained_components[slow_order[:2]].T

    result = {
        "analysis": "smad_alignment_check",
        "task": "cheetah_run",
        "rollout_dir": str(args.rollout_dir.relative_to(REPO_ROOT)),
        "rollout_files": [str(path.relative_to(REPO_ROOT)) for path in rollout_paths],
        "n_seeds": int(true_deter.shape[0]),
        "horizon": int(true_deter.shape[1]),
        "latent_dim": LATENT_DIM,
        "deter_dim": DETER_DIM,
        "deter_slice": [DETER_START, LATENT_DIM],
        "drift_trim_steps_each_side": int(args.trim),
        "drift_pool_shape": list(drift_pool.shape),
        "posterior_pca": {
            "var_threshold": float(args.var_threshold),
            "retained_components": int(retained_k),
            "retained_variance_explained": float(np.sum(retained_variance)),
            "explained_variance_ratio_retained": retained_variance.tolist(),
            "cumulative_variance_ratio_retained": np.cumsum(retained_variance).tolist(),
            "lag1_autocorrelation_retained": autocorr.tolist(),
            "slow_pc_order": slow_order.tolist(),
        },
        "drift_pca": {
            "explained_variance_ratio_top20": drift_pca.explained_variance_ratio[
                : min(20, drift_pca.explained_variance_ratio.size)
            ].tolist(),
            "cumulative_variance_ratio_top20": np.cumsum(
                drift_pca.explained_variance_ratio[
                    : min(20, drift_pca.explained_variance_ratio.size)
                ]
            ).tolist(),
        },
        "reported_ranks": reported_ranks,
        "curve_ranks": sorted(int(rank) for rank in rank_results),
        "rank_results": rank_results,
        "drift_projection_u_posterior_r2": drift_projection_r2.tolist(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    print_summary(result)
    print(f"\nSaved: {args.output.relative_to(REPO_ROOT)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify alignment between posterior slow modes and drift directions."
    )
    parser.add_argument("--rollout-dir", type=Path, default=ROLLOUT_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--var-threshold", type=float, default=VAR_THRESHOLD)
    parser.add_argument("--trim", type=int, default=DRIFT_TRIM)
    parser.add_argument(
        "--reported-ranks",
        type=int,
        nargs="+",
        default=list(DEFAULT_REPORTED_RANKS),
    )
    parser.add_argument(
        "--curve-ranks",
        type=int,
        nargs="+",
        default=list(DEFAULT_CURVE_RANKS),
    )
    args = parser.parse_args()
    if not 0 < args.var_threshold <= 1:
        raise ValueError("--var-threshold must be in (0, 1].")
    if not 0 <= args.trim < HORIZON // 2:
        raise ValueError(f"--trim must be in [0, {HORIZON // 2}).")
    args.rollout_dir = resolve_path(args.rollout_dir)
    args.output = resolve_path(args.output)
    return args


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def discover_rollouts(rollout_dir: Path) -> list[Path]:
    by_seed: dict[int, Path] = {}
    for path in rollout_dir.glob("cheetah_v3_seed*_v2.npz"):
        match = ROLLOUT_RE.fullmatch(path.name)
        if match is None:
            continue
        by_seed[int(match.group(1))] = path

    missing = [seed for seed in range(N_SEEDS) if seed not in by_seed]
    if missing:
        raise FileNotFoundError(
            f"Missing Cheetah v2 rollout files for seeds: {missing}"
        )
    return [by_seed[seed] for seed in range(N_SEEDS)]


def load_deter_rollouts(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    true_deters = []
    imag_deters = []
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            true_latent = require_array(data, "true_latent", path)
            imag_latent = require_array(data, "imagined_latent", path)
        validate_latent(true_latent, "true_latent", path)
        validate_latent(imag_latent, "imagined_latent", path)
        true_deters.append(true_latent[:, DETER_START:].astype(np.float64, copy=False))
        imag_deters.append(imag_latent[:, DETER_START:].astype(np.float64, copy=False))

    true_deter = np.stack(true_deters, axis=0)
    imag_deter = np.stack(imag_deters, axis=0)
    expected = (N_SEEDS, HORIZON, DETER_DIM)
    if true_deter.shape != expected or imag_deter.shape != expected:
        raise ValueError(
            f"Expected deter rollouts {expected}, got "
            f"{true_deter.shape} and {imag_deter.shape}."
        )
    return true_deter, imag_deter


def require_array(data: np.lib.npyio.NpzFile, key: str, path: Path) -> np.ndarray:
    if key not in data:
        raise KeyError(f"{path} does not contain {key!r}.")
    return np.asarray(data[key])


def validate_latent(array: np.ndarray, key: str, path: Path) -> None:
    expected = (HORIZON, LATENT_DIM)
    if array.shape != expected:
        raise ValueError(f"{path}:{key} expected {expected}, got {array.shape}.")


def fit_pca(matrix: np.ndarray) -> PCAFit:
    matrix = np.asarray(matrix, dtype=np.float64)
    mean = matrix.mean(axis=0, keepdims=True)
    centered = matrix - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    explained_variance = singular_values**2 / max(matrix.shape[0] - 1, 1)
    total = explained_variance.sum()
    if total <= 0:
        explained_ratio = np.zeros_like(explained_variance)
        explained_ratio[0] = 1.0
    else:
        explained_ratio = explained_variance / total
    return PCAFit(
        mean=mean.reshape(-1),
        components=vt,
        explained_variance_ratio=explained_ratio,
        singular_values=singular_values,
    )


def n_components_for_variance(ratios: np.ndarray, threshold: float) -> int:
    cumulative = np.cumsum(ratios)
    return int(np.searchsorted(cumulative, threshold, side="left") + 1)


def project_rollouts(
    rollouts: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
) -> np.ndarray:
    centered = rollouts - mean.reshape(1, 1, -1)
    return centered @ components.T


def lag1_autocorr_by_pc(projections: np.ndarray) -> np.ndarray:
    n_seeds, _, n_pcs = projections.shape
    values = np.empty((n_seeds, n_pcs), dtype=np.float64)
    for seed in range(n_seeds):
        for pc in range(n_pcs):
            values[seed, pc] = lag1_autocorr(projections[seed, :, pc])
    return np.nanmean(values, axis=0)


def lag1_autocorr(series: np.ndarray) -> float:
    x0 = np.asarray(series[:-1], dtype=np.float64)
    x1 = np.asarray(series[1:], dtype=np.float64)
    x0 = x0 - x0.mean()
    x1 = x1 - x1.mean()
    denom = np.sqrt(np.dot(x0, x0) * np.dot(x1, x1))
    if denom <= 0:
        return 0.0
    return float(np.dot(x0, x1) / denom)


def compute_rank_results(
    ranks: list[int],
    retained_components: np.ndarray,
    slow_order: np.ndarray,
    drift_components: np.ndarray,
    drift_pool: np.ndarray,
) -> dict[int, dict[str, Any]]:
    total_energy = float(np.sum(drift_pool**2))
    if total_energy <= 0:
        raise ValueError("Total drift energy is zero; cannot compute projection ratio.")

    results: dict[int, dict[str, Any]] = {}
    for rank in ranks:
        selected = slow_order[:rank]
        u_posterior = retained_components[selected].T
        u_drift = drift_components[:rank].T
        overlap = float(np.linalg.norm(u_posterior.T @ u_drift, ord="fro") ** 2 / rank)
        projected = drift_pool @ u_posterior
        variance_ratio = float(np.sum(projected**2) / total_energy)
        results[int(rank)] = {
            "r": int(rank),
            "selected_pc_indices": selected.tolist(),
            "overlap": float(np.clip(overlap, 0.0, 1.0)),
            "variance_projection_ratio": variance_ratio,
            "verdict": verdict_for_overlap(overlap),
        }
    return results


def verdict_for_overlap(overlap: float) -> str:
    if overlap > 0.7:
        return "STRONG GO - U_posterior is a good basis for SMAD"
    if overlap >= 0.4:
        return "WEAK GO - proceed with caution, monitor during Phase 2"
    return "NO GO with U_posterior - use U_drift directly as basis"


def print_summary(result: dict[str, Any]) -> None:
    pca = result["posterior_pca"]
    print("SMAD Pre-check 1: U_posterior vs drift subspace")
    print(f"Rollouts: {result['n_seeds']} seeds, horizon={result['horizon']}")
    print(
        "Posterior PCA: "
        f"k={pca['retained_components']} PCs, "
        f"variance={pca['retained_variance_explained']:.4f}"
    )
    print(f"Drift pool: {tuple(result['drift_pool_shape'])}")
    print()
    print("r   overlap   var_ratio   verdict")
    print("--  --------  ---------   -------")
    for rank in result["reported_ranks"]:
        item = result["rank_results"][str(rank)]
        print(
            f"{rank:<2d}  "
            f"{item['overlap']:.4f}    "
            f"{item['variance_projection_ratio']:.4f}      "
            f"{item['verdict']}"
        )


if __name__ == "__main__":
    main()
