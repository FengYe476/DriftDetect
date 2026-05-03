#!/usr/bin/env python3
"""Numerical check for the toy mismatch theorem conjecture.

This script is intentionally NumPy-only. The existing toy RSSM implementation is
PyTorch-based, but the theorem check only needs a controlled linear system where
posterior and drift covariance operators can be compared directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "results" / "tables" / "toy_mismatch_theorem_check.json"
DEFAULT_EIGENVALUES = np.asarray(
    [0.99, 0.97, 0.95, 0.90, 0.50, 0.40, 0.30, 0.20, 0.10, 0.05],
    dtype=np.float64,
)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    eigenvectors = orthogonal_matrix(rng, args.dim)
    A_true = eigenvectors @ np.diag(DEFAULT_EIGENVALUES) @ eigenvectors.T
    perturbation = make_noncommuting_perturbation(
        eigenvectors,
        rank=args.rank,
        rng=rng,
    )
    true_rollouts = generate_true_rollouts(
        A_true,
        seed=args.seed + 1,
        n_rollouts=args.n_rollouts,
        burn_in=args.burn_in,
        posterior_steps=args.posterior_steps,
        horizon=args.horizon,
        process_noise_std=args.process_noise_std,
    )

    capacity_gap_sweep = run_capacity_gap_sweep(
        args=args,
        A_true=A_true,
        perturbation=perturbation,
        true_rollouts=true_rollouts,
    )
    noise_structure_sweep = run_noise_structure_sweep(
        args=args,
        A_true=A_true,
        perturbation=perturbation,
        eigenvectors=eigenvectors,
        true_rollouts=true_rollouts,
    )
    support_summary = summarize_support(capacity_gap_sweep, noise_structure_sweep)

    output = {
        "analysis": "toy_mismatch_theorem_check",
        "candidate_theorem": "B_prime",
        "notes": {
            "interpretation": (
                "Numerical support for a conjecture, not a proof. The script "
                "checks whether posterior covariance and accumulated drift "
                "covariance produce different leading subspaces under controlled "
                "linear dynamics."
            ),
            "random_rank3_overlap_floor": float(args.rank / args.dim),
            "noncommuting_perturbation": (
                "In the A_true eigenbasis, the perturbation maps the top-r slow "
                "coordinates into the remaining faster coordinates."
            ),
            "structured_noise_orientation": (
                "Structured noise eigenvalues are applied in reverse A_true "
                "eigenbasis order, so largest noise variance falls on fast "
                "directions."
            ),
        },
        "config": {
            "seed": int(args.seed),
            "dim": int(args.dim),
            "rank": int(args.rank),
            "true_eigenvalues": DEFAULT_EIGENVALUES.tolist(),
            "n_rollouts": int(args.n_rollouts),
            "burn_in": int(args.burn_in),
            "posterior_steps": int(args.posterior_steps),
            "horizon": int(args.horizon),
            "start_stride": int(args.start_stride),
            "process_noise_std": float(args.process_noise_std),
            "inference_noise_std": float(args.inference_noise_std),
            "epsilon_max": float(args.epsilon_max),
            "epsilon_steps": int(args.epsilon_steps),
            "noise_epsilon": float(args.noise_epsilon),
        },
        "capacity_gap_sweep": capacity_gap_sweep,
        "noise_structure_sweep": noise_structure_sweep,
        "support_summary": support_summary,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(to_jsonable(output), indent=2) + "\n")

    print_capacity_table(capacity_gap_sweep)
    print_noise_table(noise_structure_sweep)
    print_support_summary(support_summary)
    print(f"\nSaved: {relative_path(args.output)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Numerically check the toy posterior/drift mismatch theorem."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--n_rollouts", type=int, default=800)
    parser.add_argument("--burn_in", type=int, default=80)
    parser.add_argument("--posterior_steps", type=int, default=80)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--start_stride", type=int, default=4)
    parser.add_argument("--process_noise_std", type=float, default=0.02)
    parser.add_argument("--inference_noise_std", type=float, default=0.10)
    parser.add_argument("--epsilon_max", type=float, default=0.30)
    parser.add_argument("--epsilon_steps", type=int, default=10)
    parser.add_argument("--noise_epsilon", type=float, default=0.10)
    args = parser.parse_args()

    args.output = resolve_path(args.output)
    if args.dim != int(DEFAULT_EIGENVALUES.size):
        raise ValueError(
            f"This check is parameterized for dim={DEFAULT_EIGENVALUES.size}; "
            f"got --dim={args.dim}."
        )
    if not 1 <= args.rank < args.dim:
        raise ValueError(f"--rank must be in [1, {args.dim - 1}], got {args.rank}.")
    if args.n_rollouts <= 0:
        raise ValueError("--n_rollouts must be positive.")
    if args.burn_in < 0:
        raise ValueError("--burn_in must be non-negative.")
    if args.posterior_steps <= 0:
        raise ValueError("--posterior_steps must be positive.")
    if args.horizon <= 0:
        raise ValueError("--horizon must be positive.")
    if args.start_stride <= 0:
        raise ValueError("--start_stride must be positive.")
    if args.process_noise_std < 0:
        raise ValueError("--process_noise_std must be non-negative.")
    if args.inference_noise_std < 0:
        raise ValueError("--inference_noise_std must be non-negative.")
    if args.epsilon_max < 0:
        raise ValueError("--epsilon_max must be non-negative.")
    if args.epsilon_steps < 2:
        raise ValueError("--epsilon_steps must be at least 2.")
    if args.noise_epsilon < 0:
        raise ValueError("--noise_epsilon must be non-negative.")
    return args


def run_capacity_gap_sweep(
    *,
    args: argparse.Namespace,
    A_true: np.ndarray,
    perturbation: np.ndarray,
    true_rollouts: np.ndarray,
) -> list[dict[str, Any]]:
    epsilons = np.linspace(0.0, args.epsilon_max, args.epsilon_steps)
    inference_cov = make_noise_covariance(
        condition="isotropic",
        dim=args.dim,
        eigenvectors=None,
        sigma=args.inference_noise_std,
    )
    rows = []
    for epsilon in epsilons:
        A_model = contractive_model_matrix(A_true, perturbation, float(epsilon))
        overlap = evaluate_overlap(
            A_model=A_model,
            true_rollouts=true_rollouts,
            inference_cov=inference_cov,
            rank=args.rank,
            posterior_steps=args.posterior_steps,
            horizon=args.horizon,
            start_stride=args.start_stride,
            seed=args.seed + 1000,
        )
        rows.append(
            {
                "epsilon": float(epsilon),
                "capacity_gap_spectral_norm": float(
                    np.linalg.norm(A_model - A_true, ord=2)
                ),
                "spectral_radius": spectral_radius(A_model),
                "overlap": float(overlap),
            }
        )
    return rows


def run_noise_structure_sweep(
    *,
    args: argparse.Namespace,
    A_true: np.ndarray,
    perturbation: np.ndarray,
    eigenvectors: np.ndarray,
    true_rollouts: np.ndarray,
) -> list[dict[str, Any]]:
    A_model = contractive_model_matrix(A_true, perturbation, args.noise_epsilon)
    conditions = noise_conditions(args.dim)
    rows = []
    for index, condition in enumerate(conditions):
        covariance = make_noise_covariance(
            condition=condition["name"],
            dim=args.dim,
            eigenvectors=eigenvectors,
            sigma=args.inference_noise_std,
        )
        overlap = evaluate_overlap(
            A_model=A_model,
            true_rollouts=true_rollouts,
            inference_cov=covariance,
            rank=args.rank,
            posterior_steps=args.posterior_steps,
            horizon=args.horizon,
            start_stride=args.start_stride,
            seed=args.seed + 2000 + index,
        )
        rows.append(
            {
                "condition": condition["name"],
                "noise_eigenvalues": condition["eigenvalues"].tolist(),
                "spectral_radius": spectral_radius(A_model),
                "overlap": float(overlap),
            }
        )
    return rows


def evaluate_overlap(
    *,
    A_model: np.ndarray,
    true_rollouts: np.ndarray,
    inference_cov: np.ndarray,
    rank: int,
    posterior_steps: int,
    horizon: int,
    start_stride: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    n_rollouts, total_steps, dim = true_rollouts.shape
    if total_steps < posterior_steps + horizon + 1:
        raise ValueError("true_rollouts does not cover posterior_steps + horizon.")
    inference_noise = rng.multivariate_normal(
        mean=np.zeros(dim, dtype=np.float64),
        cov=inference_cov,
        size=(n_rollouts, total_steps),
    )

    posterior_latents = true_rollouts[:, :posterior_steps] + inference_noise[
        :, :posterior_steps
    ]
    U_posterior = pca_basis(posterior_latents.reshape(-1, dim), rank=rank)

    drift_chunks = []
    starts = range(0, posterior_steps, start_stride)
    for start in starts:
        imagined = true_rollouts[:, start] + inference_noise[:, start]
        for horizon_step in range(1, horizon + 1):
            imagined = imagined @ A_model.T
            true_future = true_rollouts[:, start + horizon_step]
            drift_chunks.append(imagined - true_future)
    drift_matrix = np.concatenate(drift_chunks, axis=0)
    U_drift = pca_basis(drift_matrix, rank=rank)
    return subspace_overlap(U_posterior, U_drift)


def generate_true_rollouts(
    A_true: np.ndarray,
    *,
    seed: int,
    n_rollouts: int,
    burn_in: int,
    posterior_steps: int,
    horizon: int,
    process_noise_std: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dim = A_true.shape[0]
    total_kept = posterior_steps + horizon + 1
    state = rng.normal(size=(n_rollouts, dim))
    kept = []
    for step in range(burn_in + total_kept):
        state = state @ A_true.T
        if process_noise_std > 0:
            state = state + rng.normal(
                scale=process_noise_std,
                size=(n_rollouts, dim),
            )
        if step >= burn_in:
            kept.append(state.copy())
    return np.stack(kept, axis=1)


def make_noncommuting_perturbation(
    eigenvectors: np.ndarray,
    *,
    rank: int,
    rng: np.random.Generator,
) -> np.ndarray:
    dim = eigenvectors.shape[0]
    mixing = rng.normal(size=(dim - rank, rank))
    mixing = mixing / np.linalg.norm(mixing, ord=2)
    perturbation_eig = np.zeros((dim, dim), dtype=np.float64)
    perturbation_eig[rank:, :rank] = mixing
    return eigenvectors @ perturbation_eig @ eigenvectors.T


def contractive_model_matrix(
    A_true: np.ndarray,
    perturbation: np.ndarray,
    epsilon: float,
    max_radius: float = 0.995,
) -> np.ndarray:
    A_model = A_true + float(epsilon) * perturbation
    radius = spectral_radius(A_model)
    if radius > max_radius:
        A_model = A_model * (max_radius / radius)
    return A_model


def make_noise_covariance(
    *,
    condition: str,
    dim: int,
    eigenvectors: np.ndarray | None,
    sigma: float,
) -> np.ndarray:
    if condition == "isotropic":
        return float(sigma) ** 2 * np.eye(dim, dtype=np.float64)
    values = noise_condition_map(dim)[condition]
    if eigenvectors is None:
        raise ValueError("eigenvectors are required for structured noise.")
    orientation = eigenvectors[:, ::-1]
    return float(sigma) ** 2 * orientation @ np.diag(values) @ orientation.T


def noise_conditions(dim: int) -> list[dict[str, np.ndarray]]:
    mapping = noise_condition_map(dim)
    return [
        {"name": "isotropic", "eigenvalues": np.ones(dim, dtype=np.float64)},
        {"name": "mild_structured", "eigenvalues": mapping["mild_structured"]},
        {"name": "structured", "eigenvalues": mapping["structured"]},
        {"name": "strong_structured", "eigenvalues": mapping["strong_structured"]},
    ]


def noise_condition_map(dim: int) -> dict[str, np.ndarray]:
    return {
        "mild_structured": np.geomspace(1.0, 0.1, dim),
        "structured": np.geomspace(1.0, 0.001, dim),
        "strong_structured": np.geomspace(1.0, 0.0003, dim),
    }


def pca_basis(matrix: np.ndarray, *, rank: int) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return vt[:rank].T


def subspace_overlap(U_a: np.ndarray, U_b: np.ndarray) -> float:
    rank = U_a.shape[1]
    return float(np.linalg.norm(U_a.T @ U_b, ord="fro") ** 2 / rank)


def spectral_radius(matrix: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def summarize_support(
    capacity_gap_sweep: list[dict[str, Any]],
    noise_structure_sweep: list[dict[str, Any]],
) -> dict[str, Any]:
    epsilons = np.asarray([row["epsilon"] for row in capacity_gap_sweep])
    capacity_overlaps = np.asarray([row["overlap"] for row in capacity_gap_sweep])
    endpoint_drop = float(capacity_overlaps[0] - capacity_overlaps[-1])
    pearson = safe_corr(epsilons, capacity_overlaps)

    noise_by_name = {row["condition"]: float(row["overlap"]) for row in noise_structure_sweep}
    isotropic = noise_by_name["isotropic"]
    structured_values = [
        value for name, value in noise_by_name.items() if name != "isotropic"
    ]
    structured_below_isotropic = all(value < isotropic for value in structured_values)
    verdict = (
        "supports_conjecture"
        if endpoint_drop > 0.0 and pearson < 0.0 and structured_below_isotropic
        else "mixed_or_negative"
    )
    return {
        "capacity_endpoint_drop": endpoint_drop,
        "capacity_pearson_epsilon_vs_overlap": pearson,
        "capacity_support": bool(endpoint_drop > 0.0 and pearson < 0.0),
        "isotropic_overlap": isotropic,
        "min_structured_overlap": float(min(structured_values)),
        "structured_below_isotropic": bool(structured_below_isotropic),
        "noise_support": bool(structured_below_isotropic),
        "verdict": verdict,
    }


def orthogonal_matrix(rng: np.random.Generator, dim: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(dim, dim)))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q * signs


def print_capacity_table(rows: list[dict[str, Any]]) -> None:
    print("Capacity gap sweep:", flush=True)
    print("| epsilon | spectral_radius | overlap |", flush=True)
    print("|---:|---:|---:|", flush=True)
    for row in rows:
        print(
            f"| {row['epsilon']:.3f} | {row['spectral_radius']:.4f} | "
            f"{row['overlap']:.4f} |",
            flush=True,
        )


def print_noise_table(rows: list[dict[str, Any]]) -> None:
    print("\nNoise structure sweep:", flush=True)
    print("| condition | overlap |", flush=True)
    print("|---|---:|", flush=True)
    for row in rows:
        print(f"| {row['condition']} | {row['overlap']:.4f} |", flush=True)


def print_support_summary(summary: dict[str, Any]) -> None:
    print("\nSupport summary:", flush=True)
    print(
        f"  capacity endpoint drop: {summary['capacity_endpoint_drop']:.4f}",
        flush=True,
    )
    print(
        "  pearson(epsilon, overlap): "
        f"{summary['capacity_pearson_epsilon_vs_overlap']:.4f}",
        flush=True,
    )
    print(
        "  isotropic vs min structured overlap: "
        f"{summary['isotropic_overlap']:.4f} vs "
        f"{summary['min_structured_overlap']:.4f}",
        flush=True,
    )
    print(f"  verdict: {summary['verdict']}", flush=True)


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return relative_path(value)
    return value


if __name__ == "__main__":
    main()
