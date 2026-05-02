#!/usr/bin/env python3
"""V4 Cheetah posterior/drift mismatch check for SMAD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from run_cartpole_mismatch_check import (  # noqa: E402
    assert_orthonormal,
    fit_pca,
    fit_sfa_basis_cartpole,
    lag1_autocorr_by_pc,
    n_components_for_variance,
    orthonormality_error,
    project_rollouts,
    sfa_metadata_cartpole,
    subspace_overlap,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ROLLOUT_DIR = REPO_ROOT / "results" / "rollouts_v4"
OUTPUT_PATH = REPO_ROOT / "results" / "tables" / "v4_mismatch_check.json"
V3_CHEETAH_PATH = REPO_ROOT / "results" / "tables" / "smad_alignment_check.json"
V3_CARTPOLE_PATH = REPO_ROOT / "results" / "tables" / "cartpole_mismatch_check.json"

MIN_ROLLOUTS = 20
HORIZON = 200
LATENT_DIM = 512
VAR_THRESHOLD = 0.95
RANKS = (3, 5, 10)
DRIFT_TRIM_START = 25
DRIFT_TRIM_END_INCLUSIVE = 175
EPS = 1e-12

TRUE_LATENT_KEYS = ("true_latent", "true_latents")
IMAGINED_LATENT_KEYS = (
    "imagined_latent",
    "imagined_latents",
    "imag_latent",
    "imag_latents",
)
V3_CHEETAH_FALLBACK = {"3": 0.13, "5": 0.17, "10": 0.28}
V3_CARTPOLE_FALLBACK = {"3": 0.06, "5": 0.08, "10": 0.12}


def main() -> None:
    args = parse_args()
    rollout_paths = discover_rollouts(args.rollout_dir)
    print_found_rollouts(rollout_paths)
    true_latents, imagined_latents, key_debug = load_latent_rollouts(rollout_paths)

    posterior = fit_posterior_basis(
        true_latents,
        ranks=args.ranks,
        var_threshold=args.var_threshold,
    )
    drift = fit_drift_basis(
        true_latents,
        imagined_latents,
        ranks=args.ranks,
        var_threshold=args.var_threshold,
        trim_start=args.trim_start,
        trim_end_inclusive=args.trim_end_inclusive,
    )
    sfa_fit, sfa_error = maybe_fit_sfa(
        true_latents,
        ranks=args.ranks,
        var_threshold=args.var_threshold,
    )

    overlaps = compute_overlaps(
        ranks=args.ranks,
        posterior_bases=posterior["bases_by_rank"],
        drift_bases=drift["bases_by_rank"],
        sfa_bases=sfa_fit.bases_by_rank if sfa_fit is not None else None,
    )
    comparison = build_comparison_table(args.ranks, overlaps)
    conclusion = conclude_v4(overlaps)

    result = {
        "analysis": "v4_mismatch_check",
        "task": "cheetah-run",
        "rollout_dir": relative_path(args.rollout_dir),
        "rollout_files": [relative_path(path) for path in rollout_paths],
        "available_keys_by_file": key_debug,
        "n_rollouts": int(true_latents.shape[0]),
        "horizon": int(true_latents.shape[1]),
        "latent_dim": int(true_latents.shape[2]),
        "latent_type": "DreamerV4 packed tokenizer latent flattened from (8, 64)",
        "ranks": list(args.ranks),
        "var_threshold": float(args.var_threshold),
        "drift_trim_window_inclusive": [
            int(args.trim_start),
            int(args.trim_end_inclusive),
        ],
        "key_question": {
            "cross_architecture_threshold": 0.4,
            "v3_rssm_specific_threshold": 0.6,
            "conclusion": conclusion,
        },
        "overlaps": overlaps,
        "posterior": posterior_metadata(posterior),
        "drift": drift_metadata(drift),
        "sfa": sfa_result_metadata(sfa_fit, sfa_error),
        "comparison_table": comparison,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    print_summary(result)
    print(f"\nSaved: {relative_path(args.output)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare V4 Cheetah U_posterior, U_drift, and optional U_sfa bases."
        )
    )
    parser.add_argument("--rollout-dir", type=Path, default=ROLLOUT_DIR)
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
    if any(rank <= 0 or rank > LATENT_DIM for rank in args.ranks):
        raise ValueError(f"--ranks must be in [1, {LATENT_DIM}].")
    if not 0 <= args.trim_start <= args.trim_end_inclusive < HORIZON:
        raise ValueError("--trim-start/--trim-end-inclusive must be inside horizon.")
    args.rollout_dir = resolve_path(args.rollout_dir)
    args.output = resolve_path(args.output)
    return args


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def discover_rollouts(rollout_dir: Path) -> list[Path]:
    if not rollout_dir.exists():
        raise FileNotFoundError(
            f"V4 rollout directory not found. Expected path: {rollout_dir}"
        )
    paths = sorted(rollout_dir.glob("*.npz"))
    if len(paths) < MIN_ROLLOUTS:
        found = ", ".join(path.name for path in paths) or "<none>"
        raise FileNotFoundError(
            f"Expected at least {MIN_ROLLOUTS} V4 rollout .npz files in "
            f"{rollout_dir}, found {len(paths)}: {found}"
        )
    return paths


def print_found_rollouts(paths: list[Path]) -> None:
    print("Found V4 rollout files:")
    for path in paths:
        print(f"  {relative_path(path)}")
    print()


def load_latent_rollouts(
    paths: list[Path],
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, Any]]]:
    true_latents = []
    imagined_latents = []
    key_debug: dict[str, dict[str, Any]] = {}
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            available_keys = list(data.files)
            print(f"{relative_path(path)} keys: {available_keys}")
            true_key, true_latent = require_first_array(data, TRUE_LATENT_KEYS, path)
            imagined_key, imagined_latent = require_first_array(
                data,
                IMAGINED_LATENT_KEYS,
                path,
            )
        validate_latent(true_latent, f"{path.name}:{true_key}")
        validate_latent(imagined_latent, f"{path.name}:{imagined_key}")
        true_latents.append(true_latent.astype(np.float64, copy=False))
        imagined_latents.append(imagined_latent.astype(np.float64, copy=False))
        key_debug[relative_path(path)] = {
            "available_keys": available_keys,
            "true_latent_key": true_key,
            "imagined_latent_key": imagined_key,
        }
    print()
    return (
        np.stack(true_latents, axis=0),
        np.stack(imagined_latents, axis=0),
        key_debug,
    )


def require_first_array(
    data: np.lib.npyio.NpzFile,
    keys: tuple[str, ...],
    path: Path,
) -> tuple[str, np.ndarray]:
    for key in keys:
        if key in data:
            return key, np.asarray(data[key])
    raise KeyError(
        f"{path} does not contain any of {keys}; available keys={data.files}."
    )


def validate_latent(array: np.ndarray, name: str) -> None:
    expected = (HORIZON, LATENT_DIM)
    if array.shape != expected:
        raise ValueError(f"{name} expected shape {expected}, got {array.shape}.")


def fit_posterior_basis(
    true_latents: np.ndarray,
    *,
    ranks: tuple[int, ...],
    var_threshold: float,
) -> dict[str, Any]:
    posterior_pool = true_latents.reshape(-1, LATENT_DIM)
    pca = fit_pca(posterior_pool)
    raw_k95 = n_components_for_variance(
        pca.explained_variance_ratio,
        threshold=var_threshold,
    )
    retained_k = max(raw_k95, max(ranks))
    retained_components = pca.components[:retained_k]
    projections = project_rollouts(true_latents, pca.mean, retained_components)
    autocorr = lag1_autocorr_by_pc(projections)
    slow_order = np.argsort(-autocorr)

    bases_by_rank = {}
    orthonormality_by_rank = {}
    for rank in ranks:
        basis = retained_components[slow_order[:rank]].T
        error = orthonormality_error(basis)
        assert_orthonormal(basis, f"U_posterior_v4 r={rank}", error)
        bases_by_rank[rank] = basis
        orthonormality_by_rank[rank] = error

    return {
        "pca": pca,
        "raw_k95": raw_k95,
        "retained_k": retained_k,
        "retained_components": retained_components,
        "autocorr": autocorr,
        "slow_order": slow_order,
        "bases_by_rank": bases_by_rank,
        "orthonormality_by_rank": orthonormality_by_rank,
    }


def fit_drift_basis(
    true_latents: np.ndarray,
    imagined_latents: np.ndarray,
    *,
    ranks: tuple[int, ...],
    var_threshold: float,
    trim_start: int,
    trim_end_inclusive: int,
) -> dict[str, Any]:
    drift = imagined_latents - true_latents
    drift_window = drift[:, trim_start : trim_end_inclusive + 1, :]
    drift_pool = drift_window.reshape(-1, LATENT_DIM)
    pca = fit_pca(drift_pool)
    raw_k95 = n_components_for_variance(
        pca.explained_variance_ratio,
        threshold=var_threshold,
    )

    bases_by_rank = {}
    orthonormality_by_rank = {}
    for rank in ranks:
        basis = pca.components[:rank].T
        error = orthonormality_error(basis)
        assert_orthonormal(basis, f"U_drift_v4 r={rank}", error)
        bases_by_rank[rank] = basis
        orthonormality_by_rank[rank] = error

    return {
        "pca": pca,
        "raw_k95": raw_k95,
        "drift_pool": drift_pool,
        "trimmed_steps_per_rollout": drift_window.shape[1],
        "bases_by_rank": bases_by_rank,
        "orthonormality_by_rank": orthonormality_by_rank,
    }


def maybe_fit_sfa(
    true_latents: np.ndarray,
    *,
    ranks: tuple[int, ...],
    var_threshold: float,
) -> tuple[Any | None, str | None]:
    try:
        return (
            fit_sfa_basis_cartpole(
                true_latents,
                ranks=ranks,
                var_threshold=var_threshold,
            ),
            None,
        )
    except Exception as exc:
        return None, repr(exc)


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


def posterior_metadata(posterior: dict[str, Any]) -> dict[str, Any]:
    pca = posterior["pca"]
    slow_order = posterior["slow_order"]
    autocorr = posterior["autocorr"]
    n_top = min(20, autocorr.size)
    return {
        "method": (
            "PCA on pooled true V4 latents, then PCs ranked by mean "
            "within-rollout lag-1 autocorrelation."
        ),
        "pca": pca_diagnostics(
            pca,
            raw_k95=posterior["raw_k95"],
            retained_k=posterior["retained_k"],
        ),
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
        "method": "PCA on imagined_latent - true_latent over the trimmed window.",
        "drift_pool_shape": list(drift["drift_pool"].shape),
        "trimmed_steps_per_rollout": int(drift["trimmed_steps_per_rollout"]),
        "pca": pca_diagnostics(
            pca,
            raw_k95=drift["raw_k95"],
            retained_k=max(RANKS),
        ),
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


def pca_diagnostics(pca: Any, *, raw_k95: int, retained_k: int) -> dict[str, Any]:
    ratios = pca.explained_variance_ratio
    top_n = min(20, ratios.size)
    total_variance = float(np.sum(pca.explained_variance))
    top1 = float(ratios[0]) if ratios.size else 0.0
    warnings = {
        "near_zero_total_variance": bool(total_variance <= EPS),
        "top1_variance_ratio_gt_0_90": bool(top1 > 0.90),
        "raw_k95_lt_3": bool(raw_k95 < 3),
    }
    return {
        "n_components_total": int(pca.components.shape[0]),
        "components_for_95_variance_raw": int(raw_k95),
        "retained_components": int(retained_k),
        "retained_variance_explained": float(np.sum(ratios[:retained_k])),
        "total_variance": total_variance,
        "explained_variance_ratio_top20": ratios[:top_n].tolist(),
        "cumulative_variance_ratio_top20": np.cumsum(ratios[:top_n]).tolist(),
        "eigenvalues_top20": pca.explained_variance[:top_n].tolist(),
        "warnings": warnings,
    }


def sfa_result_metadata(sfa_fit: Any | None, sfa_error: str | None) -> dict[str, Any]:
    if sfa_fit is None:
        return {
            "estimated": False,
            "method_source": "scripts/run_cartpole_mismatch_check.py",
            "reason": sfa_error or "SFA helper unavailable.",
        }
    metadata = {
        "estimated": True,
        "method_source": "scripts/run_cartpole_mismatch_check.py",
    }
    metadata.update(sfa_metadata_cartpole(sfa_fit))
    return metadata


def build_comparison_table(
    ranks: tuple[int, ...],
    overlaps: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    return {
        "V3 Cheetah": load_v3_cheetah_comparison(ranks),
        "V3 Cartpole": load_v3_cartpole_comparison(ranks),
        "V4 Cheetah": {
            str(rank): overlaps[str(rank)]["posterior_vs_drift"]
            for rank in ranks
        },
    }


def load_v3_cheetah_comparison(ranks: tuple[int, ...]) -> dict[str, float]:
    if V3_CHEETAH_PATH.exists():
        data = json.loads(V3_CHEETAH_PATH.read_text())
        rank_results = data.get("rank_results", {})
        values = {
            str(rank): float(rank_results[str(rank)]["overlap"])
            for rank in ranks
            if str(rank) in rank_results
        }
        if len(values) == len(ranks):
            return values
    return {str(rank): V3_CHEETAH_FALLBACK[str(rank)] for rank in ranks}


def load_v3_cartpole_comparison(ranks: tuple[int, ...]) -> dict[str, float]:
    if V3_CARTPOLE_PATH.exists():
        data = json.loads(V3_CARTPOLE_PATH.read_text())
        overlaps = data.get("overlaps", {})
        values = {
            str(rank): float(overlaps[str(rank)]["posterior_vs_drift"])
            for rank in ranks
            if str(rank) in overlaps
        }
        if len(values) == len(ranks):
            return values
    return {str(rank): V3_CARTPOLE_FALLBACK[str(rank)] for rank in ranks}


def conclude_v4(overlaps: dict[str, dict[str, float]]) -> str:
    values = [item["posterior_vs_drift"] for item in overlaps.values()]
    if all(value < 0.4 for value in values):
        return "cross-architecture mismatch: all V4 posterior/drift overlaps are < 0.4"
    if all(value > 0.6 for value in values):
        return "V3/RSSM-specific mismatch: all V4 posterior/drift overlaps are > 0.6"
    return "mixed/ambiguous: V4 posterior/drift overlaps fall between the decision bands"


def print_summary(result: dict[str, Any]) -> None:
    posterior_pca = result["posterior"]["pca"]
    drift_pca = result["drift"]["pca"]
    print("V4 Cheetah mismatch check: U_posterior vs U_drift")
    print(
        f"Rollouts: {result['n_rollouts']}, horizon={result['horizon']}, "
        f"latent_dim={result['latent_dim']}"
    )
    print(
        "Posterior PCA: "
        f"k95={posterior_pca['components_for_95_variance_raw']}, "
        f"retained={posterior_pca['retained_components']}, "
        f"variance={posterior_pca['retained_variance_explained']:.4f}, "
        f"top1={posterior_pca['explained_variance_ratio_top20'][0]:.4f}"
    )
    print(f"Posterior PCA warnings: {posterior_pca['warnings']}")
    print(
        "Drift PCA: "
        f"k95={drift_pca['components_for_95_variance_raw']}, "
        f"drift_pool={tuple(result['drift']['drift_pool_shape'])}, "
        f"trim={tuple(result['drift_trim_window_inclusive'])} inclusive"
    )
    print(f"Drift PCA warnings: {drift_pca['warnings']}")
    if result["sfa"]["estimated"]:
        print(
            "SFA whitening PCA: "
            f"k={result['sfa']['whitened_dimensions']}, "
            "variance="
            f"{result['sfa']['whitening_pca']['retained_variance_explained']:.4f}"
        )
    else:
        print(f"SFA: skipped ({result['sfa']['reason']})")

    print()
    if result["sfa"]["estimated"]:
        print("r   posterior/drift   SFA/posterior   SFA/drift")
        print("--  ---------------   -------------   ---------")
        for rank in result["ranks"]:
            item = result["overlaps"][str(rank)]
            print(
                f"{rank:<2d}  "
                f"{item['posterior_vs_drift']:<15.4f}   "
                f"{item['sfa_vs_posterior']:<13.4f}   "
                f"{item['sfa_vs_drift']:<9.4f}"
            )
    else:
        print("r   posterior/drift")
        print("--  ---------------")
        for rank in result["ranks"]:
            item = result["overlaps"][str(rank)]
            print(f"{rank:<2d}  {item['posterior_vs_drift']:<15.4f}")

    print()
    print_cross_architecture_table(result)
    print()
    print(result["key_question"]["conclusion"])


def print_cross_architecture_table(result: dict[str, Any]) -> None:
    ranks = result["ranks"]
    comparison = result["comparison_table"]
    print("Cross-architecture mismatch comparison:")
    print("| Setting | " + " | ".join(f"r={rank}" for rank in ranks) + " |")
    print("|---|" + "|".join("---:" for _ in ranks) + "|")
    for setting in ("V3 Cheetah", "V3 Cartpole", "V4 Cheetah"):
        values = [comparison[setting][str(rank)] for rank in ranks]
        print(
            f"| {setting} | "
            + " | ".join(f"{value:.2f}" for value in values)
            + " |"
        )


if __name__ == "__main__":
    main()
