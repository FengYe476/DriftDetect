#!/usr/bin/env python3
"""SMAD pre-check 4: cross-validated post-hoc U_drift damping."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt


REPO_ROOT = Path(__file__).resolve().parents[1]
ROLLOUT_DIR = REPO_ROOT / "results" / "rollouts"
OUTPUT_PATH = REPO_ROOT / "results" / "tables" / "smad_precheck_4_xval.json"

ROLLOUT_RE = re.compile(r"cheetah_v3_seed(\d+)_v2\.npz$")
N_SEEDS = 20
N_SPLITS = 5
ESTIMATION_SEEDS_PER_SPLIT = 10
HORIZON = 200
LATENT_DIM = 1536
DETER_START = 1024
DETER_DIM = 512
RANK = 10
ETAS = (0.20, 0.50)
VAR_THRESHOLD = 0.95
TRIM_START = 25
TRIM_END_INCLUSIVE = 175

BANDS = {
    "dc_trend": (0.00, 0.02),
    "very_low": (0.02, 0.05),
    "low": (0.05, 0.10),
    "mid": (0.10, 0.20),
    "high": (0.20, 0.50),
}


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

    split_results = []
    for split_idx in range(args.n_splits):
        est_seeds, val_seeds = make_split(split_idx)
        u_drift, drift_pca = estimate_u_drift(
            true_deter[est_seeds],
            imag_deter[est_seeds],
            rank=args.rank,
            trim_start=args.trim_start,
            trim_end_inclusive=args.trim_end_inclusive,
        )

        est_eval = evaluate_seed_set(
            true_deter[est_seeds],
            imag_deter[est_seeds],
            u_drift=u_drift,
            etas=args.etas,
            trim_start=args.trim_start,
            trim_end_inclusive=args.trim_end_inclusive,
            var_threshold=args.var_threshold,
        )
        val_eval = evaluate_seed_set(
            true_deter[val_seeds],
            imag_deter[val_seeds],
            u_drift=u_drift,
            etas=args.etas,
            trim_start=args.trim_start,
            trim_end_inclusive=args.trim_end_inclusive,
            var_threshold=args.var_threshold,
        )

        split_results.append(
            {
                "split": int(split_idx),
                "estimation_seeds": est_seeds.tolist(),
                "validation_seeds": val_seeds.tolist(),
                "u_drift_estimation": {
                    "rank": int(args.rank),
                    "drift_pool_shape": [
                        int(est_seeds.size * (args.trim_end_inclusive - args.trim_start + 1)),
                        DETER_DIM,
                    ],
                    "explained_variance_ratio_top10": drift_pca.explained_variance_ratio[
                        : args.rank
                    ].tolist(),
                    "cumulative_explained_variance_top10": float(
                        np.sum(drift_pca.explained_variance_ratio[: args.rank])
                    ),
                },
                "estimation_eval": est_eval,
                "validation_eval": val_eval,
            }
        )

    summary = summarize_splits(split_results, args.etas)
    result = {
        "analysis": "smad_precheck_4_xval",
        "task": "cheetah_run",
        "purpose": (
            "Estimate U_drift on one split half and evaluate post-hoc damping "
            "on held-out rollouts to test whether the Month 4 effect transfers."
        ),
        "rollout_dir": relative_path(args.rollout_dir),
        "rollout_files": [relative_path(path) for path in rollout_paths],
        "n_seeds": N_SEEDS,
        "n_splits": int(args.n_splits),
        "seeds_per_split": {
            "estimation": ESTIMATION_SEEDS_PER_SPLIT,
            "validation": N_SEEDS - ESTIMATION_SEEDS_PER_SPLIT,
        },
        "split_rng": "numpy.random.RandomState(split_idx)",
        "horizon": HORIZON,
        "latent_dim": LATENT_DIM,
        "deter_dim": DETER_DIM,
        "deter_slice": [DETER_START, LATENT_DIM],
        "rank": int(args.rank),
        "etas": list(args.etas),
        "trim_window_inclusive": [args.trim_start, args.trim_end_inclusive],
        "frequency_eval": {
            "pca_fit_scope": (
                "Fit separately on each estimation or validation set's true deter "
                "trajectories; no all-seed PCA is used."
            ),
            "var_threshold": float(args.var_threshold),
            "bands": {name: list(bounds) for name, bounds in BANDS.items()},
            "filter": {"type": "butterworth", "order": 4, "fs": 1.0},
            "total_mse_definition": "sum of the five per-band MSE values",
        },
        "damping": {
            "formula": (
                "imagined_damped[t] = imagined_deter[t] - eta * "
                "U_drift_est @ (U_drift_est.T @ "
                "(imagined_deter[t] - true_deter[t]))"
            )
        },
        "splits": split_results,
        "summary": summary,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    print_summary(result)
    print(f"\nSaved: {relative_path(args.output)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-validate post-hoc damping along U_drift estimated from split halves."
    )
    parser.add_argument("--rollout-dir", type=Path, default=ROLLOUT_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument("--rank", type=int, default=RANK)
    parser.add_argument("--etas", type=float, nargs="+", default=list(ETAS))
    parser.add_argument("--var-threshold", type=float, default=VAR_THRESHOLD)
    parser.add_argument("--trim-start", type=int, default=TRIM_START)
    parser.add_argument(
        "--trim-end-inclusive",
        type=int,
        default=TRIM_END_INCLUSIVE,
    )
    args = parser.parse_args()
    if args.n_splits <= 0:
        raise ValueError("--n-splits must be positive.")
    if not 1 <= args.rank <= DETER_DIM:
        raise ValueError(f"--rank must be in [1, {DETER_DIM}].")
    if any(eta < 0 for eta in args.etas):
        raise ValueError("--etas must be non-negative.")
    if not 0 < args.var_threshold <= 1:
        raise ValueError("--var-threshold must be in (0, 1].")
    if not 0 <= args.trim_start <= args.trim_end_inclusive < HORIZON:
        raise ValueError("--trim-start/--trim-end-inclusive must be inside horizon.")

    args.rollout_dir = resolve_path(args.rollout_dir)
    args.output = resolve_path(args.output)
    args.etas = tuple(float(eta) for eta in args.etas)
    return args


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


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


def make_split(split_idx: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(split_idx)
    permuted = rng.permutation(N_SEEDS)
    estimation = np.sort(permuted[:ESTIMATION_SEEDS_PER_SPLIT])
    validation = np.sort(permuted[ESTIMATION_SEEDS_PER_SPLIT:])
    return estimation, validation


def estimate_u_drift(
    true_deter: np.ndarray,
    imag_deter: np.ndarray,
    *,
    rank: int,
    trim_start: int,
    trim_end_inclusive: int,
) -> tuple[np.ndarray, PCAFit]:
    drift = imag_deter - true_deter
    drift_pool = drift[:, trim_start : trim_end_inclusive + 1, :].reshape(
        -1, DETER_DIM
    )
    drift_pca = fit_pca(drift_pool)
    u_drift = drift_pca.components[:rank].T
    assert_orthonormal(u_drift, "U_drift_est")
    return u_drift, drift_pca


def evaluate_seed_set(
    true_deter: np.ndarray,
    imag_deter: np.ndarray,
    *,
    u_drift: np.ndarray,
    etas: tuple[float, ...],
    trim_start: int,
    trim_end_inclusive: int,
    var_threshold: float,
) -> dict[str, Any]:
    eval_pca = fit_pca(true_deter.reshape(-1, DETER_DIM))
    eval_k = n_components_for_variance(
        eval_pca.explained_variance_ratio,
        threshold=var_threshold,
    )
    eval_components = eval_pca.components[:eval_k]

    true_projected = project_rollouts(true_deter, eval_pca.mean, eval_components)
    imag_projected = project_rollouts(imag_deter, eval_pca.mean, eval_components)
    true_bands = filter_all_bands(true_projected)
    imag_bands = filter_all_bands(imag_projected)
    original_band_mse = band_mse_over_window(
        true_bands,
        imag_bands,
        trim_start=trim_start,
        trim_end_inclusive=trim_end_inclusive,
    )
    original_total = float(sum(original_band_mse.values()))
    raw_original_mse = raw_mse_over_window(
        true_deter,
        imag_deter,
        trim_start=trim_start,
        trim_end_inclusive=trim_end_inclusive,
    )

    eta_results = {}
    for eta in etas:
        damped_deter = apply_posthoc_damping(true_deter, imag_deter, u_drift, eta)
        damped_projected = project_rollouts(
            damped_deter,
            eval_pca.mean,
            eval_components,
        )
        damped_bands = filter_all_bands(damped_projected)
        damped_band_mse = band_mse_over_window(
            true_bands,
            damped_bands,
            trim_start=trim_start,
            trim_end_inclusive=trim_end_inclusive,
        )
        damped_total = float(sum(damped_band_mse.values()))
        raw_damped_mse = raw_mse_over_window(
            true_deter,
            damped_deter,
            trim_start=trim_start,
            trim_end_inclusive=trim_end_inclusive,
        )
        eta_results[eta_key(eta)] = {
            "eta": float(eta),
            "band_mse_original": original_band_mse,
            "band_mse_damped": damped_band_mse,
            "dc_trend_mse_original": float(original_band_mse["dc_trend"]),
            "dc_trend_mse_damped": float(damped_band_mse["dc_trend"]),
            "total_mse_original": original_total,
            "total_mse_damped": damped_total,
            "dc_trend_reduction_pct": percent_reduction(
                original_band_mse["dc_trend"],
                damped_band_mse["dc_trend"],
            ),
            "total_reduction_pct": percent_reduction(original_total, damped_total),
            "raw_deter_mse_original": float(raw_original_mse),
            "raw_deter_mse_damped": float(raw_damped_mse),
            "raw_deter_reduction_pct": percent_reduction(
                raw_original_mse,
                raw_damped_mse,
            ),
        }

    return {
        "n_seeds": int(true_deter.shape[0]),
        "frequency_pca": {
            "retained_components": int(eval_k),
            "retained_variance_explained": float(
                np.sum(eval_pca.explained_variance_ratio[:eval_k])
            ),
            "explained_variance_ratio_top20": eval_pca.explained_variance_ratio[
                : min(20, eval_pca.explained_variance_ratio.size)
            ].tolist(),
        },
        "original": {
            "band_mse": original_band_mse,
            "dc_trend_mse": float(original_band_mse["dc_trend"]),
            "total_mse": original_total,
            "raw_deter_mse": float(raw_original_mse),
        },
        "etas": eta_results,
    }


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
    return (rollouts - mean.reshape(1, 1, -1)) @ components.T


def apply_posthoc_damping(
    true_deter: np.ndarray,
    imag_deter: np.ndarray,
    u_drift: np.ndarray,
    eta: float,
) -> np.ndarray:
    drift = imag_deter - true_deter
    projected_drift = (drift @ u_drift) @ u_drift.T
    return imag_deter - float(eta) * projected_drift


def filter_all_bands(projected: np.ndarray) -> dict[str, np.ndarray]:
    return {
        band: filter_projected(projected, low=low, high=high)
        for band, (low, high) in BANDS.items()
    }


def filter_projected(projected: np.ndarray, low: float, high: float) -> np.ndarray:
    out = np.empty_like(projected, dtype=np.float64)
    for seed in range(projected.shape[0]):
        for dim in range(projected.shape[2]):
            out[seed, :, dim] = bandpass(projected[seed, :, dim], low=low, high=high)
    return out


def bandpass(signal_1d: np.ndarray, low: float, high: float) -> np.ndarray:
    signal_1d = np.asarray(signal_1d, dtype=np.float64)
    nyquist = 0.5
    if low == 0 and high >= nyquist:
        return signal_1d.copy()
    if high <= 0:
        return np.full_like(signal_1d, signal_1d.mean())

    high = min(high, nyquist)
    if low == 0:
        sos = butter(4, high, btype="lowpass", output="sos", fs=1.0)
    elif high >= nyquist:
        sos = butter(4, low, btype="highpass", output="sos", fs=1.0)
    else:
        sos = butter(4, (low, high), btype="bandpass", output="sos", fs=1.0)
    padlen = min(signal_1d.size - 1, 3 * (2 * sos.shape[0] + 1))
    return sosfiltfilt(sos, signal_1d, padlen=padlen)


def band_mse_over_window(
    true_bands: dict[str, np.ndarray],
    imag_bands: dict[str, np.ndarray],
    *,
    trim_start: int,
    trim_end_inclusive: int,
) -> dict[str, float]:
    window = slice(trim_start, trim_end_inclusive + 1)
    return {
        band: float(np.mean((imag_bands[band][:, window, :] - true_bands[band][:, window, :]) ** 2))
        for band in BANDS
    }


def raw_mse_over_window(
    true_deter: np.ndarray,
    imag_deter: np.ndarray,
    *,
    trim_start: int,
    trim_end_inclusive: int,
) -> float:
    diff = imag_deter[:, trim_start : trim_end_inclusive + 1, :] - true_deter[
        :, trim_start : trim_end_inclusive + 1, :
    ]
    return float(np.mean(diff**2))


def percent_reduction(original: float, damped: float) -> float:
    if original == 0:
        return 0.0 if damped == 0 else float("-inf")
    return float((original - damped) / original * 100.0)


def assert_orthonormal(matrix: np.ndarray, name: str, *, atol: float = 1e-10) -> None:
    gram = matrix.T @ matrix
    error = float(np.max(np.abs(gram - np.eye(gram.shape[0]))))
    if error > atol:
        raise RuntimeError(f"{name} is not orthonormal; max_abs error={error:.3e}.")


def summarize_splits(
    split_results: list[dict[str, Any]],
    etas: tuple[float, ...],
) -> dict[str, Any]:
    summary = {}
    for eta in etas:
        key = eta_key(eta)
        est_dc = np.asarray(
            [
                split["estimation_eval"]["etas"][key]["dc_trend_reduction_pct"]
                for split in split_results
            ],
            dtype=np.float64,
        )
        est_total = np.asarray(
            [
                split["estimation_eval"]["etas"][key]["total_reduction_pct"]
                for split in split_results
            ],
            dtype=np.float64,
        )
        val_dc = np.asarray(
            [
                split["validation_eval"]["etas"][key]["dc_trend_reduction_pct"]
                for split in split_results
            ],
            dtype=np.float64,
        )
        val_total = np.asarray(
            [
                split["validation_eval"]["etas"][key]["total_reduction_pct"]
                for split in split_results
            ],
            dtype=np.float64,
        )
        summary[key] = {
            "eta": float(eta),
            "estimation": reduction_summary(est_dc, est_total),
            "validation": reduction_summary(val_dc, val_total),
            "transfer_ratio": {
                "dc_trend": safe_ratio(float(np.mean(val_dc)), float(np.mean(est_dc))),
                "total": safe_ratio(
                    float(np.mean(val_total)),
                    float(np.mean(est_total)),
                ),
            },
        }
    return summary


def reduction_summary(dc: np.ndarray, total: np.ndarray) -> dict[str, float]:
    return {
        "dc_trend_reduction_pct_mean": float(np.mean(dc)),
        "dc_trend_reduction_pct_std": float(np.std(dc, ddof=1)),
        "total_reduction_pct_mean": float(np.mean(total)),
        "total_reduction_pct_std": float(np.std(total, ddof=1)),
    }


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def eta_key(eta: float) -> str:
    return f"{eta:.2f}"


def print_summary(result: dict[str, Any]) -> None:
    print(f"SMAD Pre-check 4: Cross-validation damping test (r={result['rank']})")
    for eta in result["etas"]:
        key = eta_key(float(eta))
        print()
        print(f"eta={float(eta):.2f}:")
        print("Split  Est dc_trend%  Est total%  Val dc_trend%  Val total%")
        for split in result["splits"]:
            est = split["estimation_eval"]["etas"][key]
            val = split["validation_eval"]["etas"][key]
            print(
                f"{split['split']:<5d}  "
                f"{est['dc_trend_reduction_pct']:<13.2f}  "
                f"{est['total_reduction_pct']:<10.2f}  "
                f"{val['dc_trend_reduction_pct']:<13.2f}  "
                f"{val['total_reduction_pct']:<10.2f}"
            )
        summary = result["summary"][key]
        print(
            "Mean   "
            f"{summary['estimation']['dc_trend_reduction_pct_mean']:<13.2f}  "
            f"{summary['estimation']['total_reduction_pct_mean']:<10.2f}  "
            f"{summary['validation']['dc_trend_reduction_pct_mean']:<13.2f}  "
            f"{summary['validation']['total_reduction_pct_mean']:<10.2f}"
        )

    print()
    print("Transfer ratio (Val/Est):")
    for eta in result["etas"]:
        key = eta_key(float(eta))
        transfer = result["summary"][key]["transfer_ratio"]
        print(
            f"eta={float(eta):.2f}: "
            f"dc_trend={transfer['dc_trend']:.3f}  "
            f"total={transfer['total']:.3f}"
        )


if __name__ == "__main__":
    main()
