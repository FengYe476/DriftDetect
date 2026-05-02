#!/usr/bin/env python3
"""Cartpole v2 posterior/drift mismatch check for SMAD."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scipy.linalg import eigh

    SCIPY_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - optional SFA dependency path
    eigh = None
    SCIPY_IMPORT_ERROR = exc

try:
    from run_smad_precheck_3a_sfa import fit_sfa_basis  # noqa: E402

    SFA_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - optional comparison path
    fit_sfa_basis = None
    SFA_IMPORT_ERROR = exc


REPO_ROOT = Path(__file__).resolve().parents[1]
PREFERRED_ROLLOUT_DIR = REPO_ROOT / "results" / "rollouts_cartpole"
FALLBACK_ROLLOUT_DIR = REPO_ROOT / "results" / "rollouts"
OUTPUT_PATH = REPO_ROOT / "results" / "tables" / "cartpole_mismatch_check.json"

ROLLOUT_RE = re.compile(r"cartpole_v3_seed(\d+)_v2\.npz$")
ROLLOUT_PATTERN = "cartpole_v3_seed*_v2.npz"
N_SEEDS = 20
HORIZON = 200
LATENT_DIM = 1536
DETER_START = 1024
DETER_DIM = 512
VAR_THRESHOLD = 0.95
RANKS = (3, 5, 10)
DRIFT_TRIM_START = 25
DRIFT_TRIM_END_INCLUSIVE = 175
EPS = 1e-12
CHEETAH_REFERENCE = {
    "3": 0.13,
    "5": 0.17,
    "10": 0.28,
}
SFA_FIT_ERRORS: list[str] = []


@dataclass(frozen=True)
class PCAFit:
    mean: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    singular_values: np.ndarray


def main() -> None:
    args = parse_args()
    rollout_paths = discover_rollouts(args.rollout_dir)
    true_deter, imag_deter = load_deter_rollouts(rollout_paths)

    posterior = fit_posterior_basis(
        true_deter,
        ranks=args.ranks,
        var_threshold=args.var_threshold,
    )
    drift = fit_drift_basis(
        true_deter,
        imag_deter,
        ranks=args.ranks,
        trim_start=args.trim_start,
        trim_end_inclusive=args.trim_end_inclusive,
    )

    sfa_fit = maybe_fit_sfa(
        true_deter,
        ranks=args.ranks,
        var_threshold=args.var_threshold,
    )

    overlaps = compute_overlaps(
        ranks=args.ranks,
        posterior_bases=posterior["bases_by_rank"],
        drift_bases=drift["bases_by_rank"],
        sfa_bases=sfa_fit.bases_by_rank if sfa_fit is not None else None,
    )

    result = {
        "analysis": "cartpole_mismatch_check",
        "task": "dmc_cartpole_swingup",
        "rollout_dir": relative_path(args.rollout_dir),
        "rollout_files": [relative_path(path) for path in rollout_paths],
        "n_seeds": int(true_deter.shape[0]),
        "horizon": int(true_deter.shape[1]),
        "latent_dim": LATENT_DIM,
        "deter_dim": DETER_DIM,
        "deter_slice": [DETER_START, LATENT_DIM],
        "ranks": list(args.ranks),
        "var_threshold": float(args.var_threshold),
        "drift_trim_window_inclusive": [
            int(args.trim_start),
            int(args.trim_end_inclusive),
        ],
        "cheetah_reference_posterior_vs_drift": CHEETAH_REFERENCE,
        "key_question": {
            "cross_task_threshold": 0.4,
            "task_specific_threshold": 0.6,
            "conclusion": conclude_mismatch(overlaps),
        },
        "overlaps": overlaps,
        "posterior": posterior_metadata(posterior),
        "drift": drift_metadata(drift),
        "sfa": sfa_result_metadata(sfa_fit),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    print_summary(result)
    print(f"\nSaved: {relative_path(args.output)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Cartpole U_posterior, U_drift, and optional U_sfa bases."
        )
    )
    parser.add_argument(
        "--rollout-dir",
        type=Path,
        default=PREFERRED_ROLLOUT_DIR,
        help=(
            "Directory containing Cartpole v2 NPZ files. Defaults to "
            "results/rollouts_cartpole and falls back to results/rollouts "
            "when the preferred directory is absent."
        ),
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--var-threshold", type=float, default=VAR_THRESHOLD)
    parser.add_argument("--ranks", type=int, nargs="+", default=list(RANKS))
    parser.add_argument("--trim-start", type=int, default=DRIFT_TRIM_START)
    parser.add_argument(
        "--trim-end-inclusive",
        type=int,
        default=DRIFT_TRIM_END_INCLUSIVE,
    )
    args = parser.parse_args()
    if not 0 < args.var_threshold <= 1:
        raise ValueError("--var-threshold must be in (0, 1].")
    args.ranks = tuple(sorted(set(args.ranks)))
    if any(rank <= 0 or rank > DETER_DIM for rank in args.ranks):
        raise ValueError(f"--ranks must be in [1, {DETER_DIM}].")
    if not 0 <= args.trim_start <= args.trim_end_inclusive < HORIZON:
        raise ValueError("--trim-start/--trim-end-inclusive must be inside horizon.")

    requested_rollout_dir = resolve_path(args.rollout_dir)
    args.rollout_dir = resolve_rollout_dir(requested_rollout_dir)
    args.output = resolve_path(args.output)
    return args


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_rollout_dir(path: Path) -> Path:
    if path.exists():
        return path
    if path == PREFERRED_ROLLOUT_DIR and FALLBACK_ROLLOUT_DIR.exists():
        return FALLBACK_ROLLOUT_DIR
    return path


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def discover_rollouts(rollout_dir: Path) -> list[Path]:
    by_seed: dict[int, Path] = {}
    for path in rollout_dir.glob(ROLLOUT_PATTERN):
        match = ROLLOUT_RE.fullmatch(path.name)
        if match is None:
            continue
        by_seed[int(match.group(1))] = path

    missing = [seed for seed in range(N_SEEDS) if seed not in by_seed]
    if missing:
        raise FileNotFoundError(
            f"Missing Cartpole v2 rollout files for seeds: {missing} "
            f"in {rollout_dir}."
        )
    return [by_seed[seed] for seed in range(N_SEEDS)]


def load_deter_rollouts(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    true_deters = []
    imag_deters = []
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            true_deter = extract_deter(data, path, direct_keys=("true_deter",))
            imag_deter = extract_deter(
                data,
                path,
                direct_keys=("imagined_deter", "imag_deter"),
                latent_keys=("imagined_latent", "imag_latent"),
            )
        validate_deter(true_deter, "true_deter", path)
        validate_deter(imag_deter, "imagined_deter", path)
        true_deters.append(true_deter.astype(np.float64, copy=False))
        imag_deters.append(imag_deter.astype(np.float64, copy=False))

    true_deter = np.stack(true_deters, axis=0)
    imag_deter = np.stack(imag_deters, axis=0)
    expected = (N_SEEDS, HORIZON, DETER_DIM)
    if true_deter.shape != expected or imag_deter.shape != expected:
        raise ValueError(
            f"Expected deter rollouts {expected}, got "
            f"{true_deter.shape} and {imag_deter.shape}."
        )
    return true_deter, imag_deter


def extract_deter(
    data: np.lib.npyio.NpzFile,
    path: Path,
    *,
    direct_keys: tuple[str, ...],
    latent_keys: tuple[str, ...] = ("true_latent", "posterior_latent"),
) -> np.ndarray:
    for key in direct_keys:
        if key in data:
            return np.asarray(data[key])
    for key in latent_keys:
        if key in data:
            latent = np.asarray(data[key])
            validate_latent(latent, key, path)
            return latent[:, DETER_START : DETER_START + DETER_DIM]
    keys = tuple(data.files)
    raise KeyError(
        f"{path} does not contain deter keys {direct_keys} or latent keys "
        f"{latent_keys}; found {keys}."
    )


def validate_latent(array: np.ndarray, key: str, path: Path) -> None:
    expected = (HORIZON, LATENT_DIM)
    if array.shape != expected:
        raise ValueError(f"{path}:{key} expected {expected}, got {array.shape}.")


def validate_deter(array: np.ndarray, key: str, path: Path) -> None:
    expected = (HORIZON, DETER_DIM)
    if array.shape != expected:
        raise ValueError(f"{path}:{key} expected {expected}, got {array.shape}.")


def fit_posterior_basis(
    true_deter: np.ndarray,
    *,
    ranks: tuple[int, ...],
    var_threshold: float,
) -> dict[str, Any]:
    posterior_pool = true_deter.reshape(-1, DETER_DIM)
    pca = fit_pca(posterior_pool)
    retained_k = max(
        n_components_for_variance(
            pca.explained_variance_ratio,
            threshold=var_threshold,
        ),
        max(ranks),
    )
    if max(ranks) > retained_k:
        raise ValueError(
            f"Requested rank {max(ranks)} but posterior PCA retained only {retained_k}."
        )

    retained_components = pca.components[:retained_k]
    projections = project_rollouts(true_deter, pca.mean, retained_components)
    autocorr = lag1_autocorr_by_pc(projections)
    slow_order = np.argsort(-autocorr)

    bases_by_rank = {}
    orthonormality_by_rank = {}
    for rank in ranks:
        basis = retained_components[slow_order[:rank]].T
        error = orthonormality_error(basis)
        assert_orthonormal(basis, f"U_posterior r={rank}", error)
        bases_by_rank[rank] = basis
        orthonormality_by_rank[rank] = error

    return {
        "pca": pca,
        "retained_k": retained_k,
        "retained_components": retained_components,
        "projections": projections,
        "autocorr": autocorr,
        "slow_order": slow_order,
        "bases_by_rank": bases_by_rank,
        "orthonormality_by_rank": orthonormality_by_rank,
    }


def fit_drift_basis(
    true_deter: np.ndarray,
    imag_deter: np.ndarray,
    *,
    ranks: tuple[int, ...],
    trim_start: int,
    trim_end_inclusive: int,
) -> dict[str, Any]:
    drift = imag_deter - true_deter
    drift_window = drift[:, trim_start : trim_end_inclusive + 1, :]
    drift_pool = drift_window.reshape(-1, DETER_DIM)
    pca = fit_pca(drift_pool)

    if max(ranks) > pca.components.shape[0]:
        raise ValueError(
            f"Requested rank {max(ranks)} but drift PCA produced "
            f"{pca.components.shape[0]} components."
        )

    bases_by_rank = {}
    orthonormality_by_rank = {}
    for rank in ranks:
        basis = pca.components[:rank].T
        error = orthonormality_error(basis)
        assert_orthonormal(basis, f"U_drift r={rank}", error)
        bases_by_rank[rank] = basis
        orthonormality_by_rank[rank] = error

    return {
        "pca": pca,
        "drift_pool": drift_pool,
        "trimmed_steps_per_seed": drift_window.shape[1],
        "bases_by_rank": bases_by_rank,
        "orthonormality_by_rank": orthonormality_by_rank,
    }


def maybe_fit_sfa(
    true_deter: np.ndarray,
    *,
    ranks: tuple[int, ...],
    var_threshold: float,
) -> Any | None:
    SFA_FIT_ERRORS.clear()
    try:
        return fit_sfa_basis_cartpole(
            true_deter,
            ranks=ranks,
            var_threshold=var_threshold,
        )
    except Exception as exc:
        SFA_FIT_ERRORS.append(f"local SFA failed: {exc!r}")
    if fit_sfa_basis is None:
        return None
    try:
        return fit_sfa_basis(true_deter, ranks=ranks, var_threshold=var_threshold)
    except Exception as exc:
        SFA_FIT_ERRORS.append(f"precheck helper SFA failed: {exc!r}")
        return None


@dataclass(frozen=True)
class SFABasisFit:
    pca: PCAFit
    retained_k: int
    retained_components: np.ndarray
    retained_eigenvalues: np.ndarray
    derivative_covariance: np.ndarray
    slow_eigenvalues: np.ndarray
    slow_vectors: np.ndarray
    raw_filters: np.ndarray
    bases_by_rank: dict[int, np.ndarray]
    raw_orthonormality_by_rank: dict[int, dict[str, float]]
    final_orthonormality_by_rank: dict[int, dict[str, float]]


def fit_sfa_basis_cartpole(
    true_deter: np.ndarray,
    *,
    ranks: tuple[int, ...],
    var_threshold: float,
) -> SFABasisFit:
    if eigh is None:
        raise ImportError(f"scipy.linalg.eigh is unavailable: {SCIPY_IMPORT_ERROR!r}")

    z_rollouts, _, _, zero_std_dims = zscore_rollouts(true_deter)
    if zero_std_dims:
        print(f"Warning: {zero_std_dims} deter dims had near-zero std during z-score.")

    z_pool = z_rollouts.reshape(-1, DETER_DIM)
    pca = fit_pca(z_pool)
    retained_k = max(
        n_components_for_variance(
            pca.explained_variance_ratio,
            threshold=var_threshold,
        ),
        max(ranks),
    )
    retained_components = pca.components[:retained_k]
    retained_eigenvalues = pca.explained_variance[:retained_k]
    safe_eigenvalues = np.maximum(retained_eigenvalues, EPS)
    inv_sqrt_eigenvalues = 1.0 / np.sqrt(safe_eigenvalues)

    centered_z = z_rollouts - pca.mean.reshape(1, 1, -1)
    delta_z = centered_z[:, 1:, :] - centered_z[:, :-1, :]
    delta_pool = delta_z.reshape(-1, DETER_DIM)
    delta_whitened = (delta_pool @ retained_components.T) * inv_sqrt_eigenvalues
    derivative_covariance = covariance(delta_whitened)
    derivative_covariance = symmetrize(derivative_covariance)
    slow_eigenvalues, slow_vectors = eigh(derivative_covariance)

    raw_filters = retained_components.T @ (
        inv_sqrt_eigenvalues[:, None] * slow_vectors
    )

    bases_by_rank: dict[int, np.ndarray] = {}
    raw_orthonormality_by_rank = {}
    final_orthonormality_by_rank = {}
    for rank in ranks:
        raw_basis = raw_filters[:, :rank]
        basis = orthonormalize_columns(raw_basis)
        bases_by_rank[rank] = basis
        raw_orthonormality_by_rank[rank] = orthonormality_error(raw_basis)
        final_error = orthonormality_error(basis)
        final_orthonormality_by_rank[rank] = final_error
        assert_orthonormal(basis, f"U_sfa r={rank}", final_error)

    return SFABasisFit(
        pca=pca,
        retained_k=retained_k,
        retained_components=retained_components,
        retained_eigenvalues=retained_eigenvalues,
        derivative_covariance=derivative_covariance,
        slow_eigenvalues=slow_eigenvalues,
        slow_vectors=slow_vectors,
        raw_filters=raw_filters,
        bases_by_rank=bases_by_rank,
        raw_orthonormality_by_rank=raw_orthonormality_by_rank,
        final_orthonormality_by_rank=final_orthonormality_by_rank,
    )


def zscore_rollouts(
    rollouts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    pool = rollouts.reshape(-1, rollouts.shape[-1])
    mean = pool.mean(axis=0)
    std = pool.std(axis=0)
    zero_mask = std < EPS
    safe_std = np.where(zero_mask, 1.0, std)
    z = (rollouts - mean.reshape(1, 1, -1)) / safe_std.reshape(1, 1, -1)
    return z, mean, safe_std, int(np.sum(zero_mask))


def covariance(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    return centered.T @ centered / max(centered.shape[0] - 1, 1)


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    return (matrix + matrix.T) * 0.5


def orthonormalize_columns(matrix: np.ndarray) -> np.ndarray:
    q, r = np.linalg.qr(matrix, mode="reduced")
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q * signs.reshape(1, -1)


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
        explained_variance=explained_variance,
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


def orthonormality_error(matrix: np.ndarray) -> dict[str, float]:
    gram = matrix.T @ matrix
    diff = gram - np.eye(gram.shape[0])
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "frobenius": float(np.linalg.norm(diff, ord="fro")),
    }


def assert_orthonormal(
    matrix: np.ndarray,
    name: str,
    error: dict[str, float] | None = None,
    *,
    atol: float = 1e-10,
) -> None:
    if error is None:
        error = orthonormality_error(matrix)
    if error["max_abs"] > atol:
        raise RuntimeError(
            f"{name} is not orthonormal; max_abs error={error['max_abs']:.3e}."
        )


def subspace_overlap(u_a: np.ndarray, u_b: np.ndarray) -> float:
    if u_a.shape != u_b.shape:
        raise ValueError(f"Basis shape mismatch: {u_a.shape} vs {u_b.shape}.")
    rank = u_a.shape[1]
    overlap = np.trace((u_a @ u_a.T) @ (u_b @ u_b.T)) / rank
    return float(np.clip(overlap, 0.0, 1.0))


def compute_overlaps(
    *,
    ranks: tuple[int, ...],
    posterior_bases: dict[int, np.ndarray],
    drift_bases: dict[int, np.ndarray],
    sfa_bases: dict[int, np.ndarray] | None,
) -> dict[str, dict[str, float]]:
    overlaps = {}
    for rank in ranks:
        u_posterior = posterior_bases[rank]
        u_drift = drift_bases[rank]
        item = {
            "posterior_vs_drift": subspace_overlap(u_posterior, u_drift),
        }
        if sfa_bases is not None:
            u_sfa = sfa_bases[rank]
            item["sfa_vs_posterior"] = subspace_overlap(u_sfa, u_posterior)
            item["sfa_vs_drift"] = subspace_overlap(u_sfa, u_drift)
        overlaps[str(rank)] = item
    return overlaps


def pca_spectrum_summary(pca: PCAFit, retained_k: int | None = None) -> dict[str, Any]:
    top_n = min(20, pca.explained_variance_ratio.size)
    summary = {
        "n_components_total": int(pca.components.shape[0]),
        "components_for_95_variance": int(
            n_components_for_variance(pca.explained_variance_ratio, VAR_THRESHOLD)
        ),
        "explained_variance_ratio_top20": pca.explained_variance_ratio[
            :top_n
        ].tolist(),
        "cumulative_variance_ratio_top20": np.cumsum(
            pca.explained_variance_ratio[:top_n]
        ).tolist(),
        "eigenvalues_top20": pca.explained_variance[:top_n].tolist(),
    }
    if retained_k is not None:
        summary.update(
            {
                "retained_components": int(retained_k),
                "retained_variance_explained": float(
                    np.sum(pca.explained_variance_ratio[:retained_k])
                ),
                "retained_eigenvalue_min": float(
                    np.min(pca.explained_variance[:retained_k])
                ),
                "retained_eigenvalue_max": float(
                    np.max(pca.explained_variance[:retained_k])
                ),
            }
        )
    return summary


def posterior_metadata(posterior: dict[str, Any]) -> dict[str, Any]:
    autocorr = posterior["autocorr"]
    slow_order = posterior["slow_order"]
    n_top = min(20, autocorr.size)
    return {
        "method": "PCA directions ranked by mean within-seed lag-1 autocorrelation.",
        "pca": pca_spectrum_summary(posterior["pca"], posterior["retained_k"]),
        "retained_components": int(posterior["retained_k"]),
        "lag1_autocorrelation_top20_by_slow_order": autocorr[
            slow_order[:n_top]
        ].tolist(),
        "slow_pc_order_top20": slow_order[:n_top].tolist(),
        "selected_pc_indices_by_rank": {
            str(rank): slow_order[:rank].tolist()
            for rank in sorted(posterior["bases_by_rank"])
        },
        "orthonormality_by_rank": {
            str(rank): posterior["orthonormality_by_rank"][rank]
            for rank in sorted(posterior["bases_by_rank"])
        },
    }


def drift_metadata(drift: dict[str, Any]) -> dict[str, Any]:
    pca = drift["pca"]
    return {
        "method": "PCA on imagined_deter - true_deter over the trimmed window.",
        "drift_pool_shape": list(drift["drift_pool"].shape),
        "trimmed_steps_per_seed": int(drift["trimmed_steps_per_seed"]),
        "pca": pca_spectrum_summary(pca),
        "explained_variance_by_rank": {
            str(rank): {
                "top_r_ratios": pca.explained_variance_ratio[:rank].tolist(),
                "cumulative": float(np.sum(pca.explained_variance_ratio[:rank])),
            }
            for rank in sorted(drift["bases_by_rank"])
        },
        "orthonormality_by_rank": {
            str(rank): drift["orthonormality_by_rank"][rank]
            for rank in sorted(drift["bases_by_rank"])
        },
    }


def sfa_result_metadata(sfa_fit: Any | None) -> dict[str, Any]:
    src_files = discover_src_sfa_files()
    metadata = {
        "estimated": sfa_fit is not None,
        "src_sfa_files": [relative_path(path) for path in src_files],
        "method_source": "scripts/run_cartpole_mismatch_check.py",
        "helper_source": "scripts/run_smad_precheck_3a_sfa.py"
        if fit_sfa_basis is not None
        else None,
    }
    if sfa_fit is None:
        metadata["reason"] = (
            "; ".join(SFA_FIT_ERRORS)
            or (
                f"SFA helper import failed: {SFA_IMPORT_ERROR!r}"
                if SFA_IMPORT_ERROR is not None
                else "No SFA helper is available."
            )
        )
        return metadata
    metadata.update(sfa_metadata_cartpole(sfa_fit))
    return metadata


def sfa_metadata_cartpole(sfa_fit: SFABasisFit) -> dict[str, Any]:
    n_slow = min(20, sfa_fit.slow_eigenvalues.size)
    n_fast = min(10, sfa_fit.slow_eigenvalues.size)
    return {
        "standardization": "Per-dimension z-score over pooled posterior deter samples.",
        "whitening_pca": pca_spectrum_summary(sfa_fit.pca, sfa_fit.retained_k),
        "whitened_dimensions": int(sfa_fit.retained_k),
        "retained_at_least_max_rank": True,
        "derivative_covariance_shape": list(sfa_fit.derivative_covariance.shape),
        "slowest_derivative_eigenvalues_top20": sfa_fit.slow_eigenvalues[
            :n_slow
        ].tolist(),
        "fastest_derivative_eigenvalues_top10": sfa_fit.slow_eigenvalues[
            -n_fast:
        ].tolist(),
        "condition_summary": {
            "slowest_eigenvalue": float(sfa_fit.slow_eigenvalues[0]),
            "fastest_eigenvalue": float(sfa_fit.slow_eigenvalues[-1]),
            "fastest_to_slowest_ratio": float(
                sfa_fit.slow_eigenvalues[-1]
                / max(sfa_fit.slow_eigenvalues[0], EPS)
            ),
        },
        "basis_mapping": (
            "Raw SFA filters use PCA whitening inverse sqrt eigenvalue scaling; "
            "columns are QR-orthonormalized before Euclidean subspace overlap."
        ),
        "orthonormality_by_rank": {
            str(rank): {
                "raw_filters": sfa_fit.raw_orthonormality_by_rank[rank],
                "final_basis": sfa_fit.final_orthonormality_by_rank[rank],
            }
            for rank in sorted(sfa_fit.bases_by_rank)
        },
    }


def discover_src_sfa_files() -> list[Path]:
    src_dir = REPO_ROOT / "src"
    if not src_dir.exists():
        return []
    candidates = []
    for path in src_dir.rglob("*.py"):
        name = path.name.lower()
        if "sfa" in name or "slow" in name:
            candidates.append(path)
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        if "Slow Feature Analysis" in text or "fit_sfa" in text or "U_sfa" in text:
            candidates.append(path)
    return sorted(set(candidates))


def conclude_mismatch(overlaps: dict[str, dict[str, float]]) -> str:
    values = [item["posterior_vs_drift"] for item in overlaps.values()]
    if all(value < 0.4 for value in values):
        return "cross-task mismatch: all Cartpole posterior/drift overlaps are < 0.4"
    if all(value > 0.6 for value in values):
        return "task-specific mismatch: all Cartpole posterior/drift overlaps are > 0.6"
    return "mixed/ambiguous: Cartpole posterior/drift overlaps fall between the decision bands"


def print_summary(result: dict[str, Any]) -> None:
    print("Cartpole mismatch check: U_posterior vs U_drift")
    print(
        f"Rollouts: {result['n_seeds']} seeds, horizon={result['horizon']}, "
        f"dir={result['rollout_dir']}"
    )
    print(
        "Posterior PCA: "
        f"k={result['posterior']['retained_components']} PCs, "
        "variance="
        f"{result['posterior']['pca']['retained_variance_explained']:.4f}"
    )
    print(
        "Drift pool: "
        f"{tuple(result['drift']['drift_pool_shape'])}, "
        f"trim={tuple(result['drift_trim_window_inclusive'])} inclusive"
    )
    if result["sfa"]["estimated"]:
        print(
            "SFA whitening PCA: "
            f"k={result['sfa']['whitened_dimensions']} PCs, "
            "variance="
            f"{result['sfa']['whitening_pca']['retained_variance_explained']:.4f}"
        )
    else:
        print(f"SFA: skipped ({result['sfa']['reason']})")
    print()
    if result["sfa"]["estimated"]:
        print("r   posterior/drift   SFA/posterior   SFA/drift   Cheetah ref")
        print("--  ---------------   -------------   ---------   -----------")
    else:
        print("r   posterior/drift   Cheetah ref")
        print("--  ---------------   -----------")

    for rank in result["ranks"]:
        item = result["overlaps"][str(rank)]
        cheetah = result["cheetah_reference_posterior_vs_drift"][str(rank)]
        if result["sfa"]["estimated"]:
            print(
                f"{rank:<2d}  "
                f"{item['posterior_vs_drift']:<15.4f}   "
                f"{item['sfa_vs_posterior']:<13.4f}   "
                f"{item['sfa_vs_drift']:<9.4f}   "
                f"{cheetah:<11.2f}"
            )
        else:
            print(
                f"{rank:<2d}  "
                f"{item['posterior_vs_drift']:<15.4f}   "
                f"{cheetah:<11.2f}"
            )
    print()
    print(result["key_question"]["conclusion"])


if __name__ == "__main__":
    main()
