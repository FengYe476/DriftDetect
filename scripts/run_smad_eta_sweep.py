#!/usr/bin/env python3
"""SMAD pre-check 2: post-hoc eta sweep with a U_drift basis."""

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
OUTPUT_PATH = REPO_ROOT / "results" / "tables" / "smad_eta_sweep.json"

ROLLOUT_RE = re.compile(r"cheetah_v3_seed(\d+)_v2\.npz$")
N_SEEDS = 20
HORIZON = 200
LATENT_DIM = 1536
DETER_START = 1024
DETER_DIM = 512
VAR_THRESHOLD = 0.95
TRIM_START = 25
TRIM_END = 175
RANKS = (3, 5, 10)
ETAS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)
GO_DC_REDUCTION_PCT = 20.0
MAX_TOTAL_INCREASE_PCT = 10.0

BANDS = {
    "dc_trend": (0.0, 0.02),
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
    true_latent, imag_latent = load_rollouts(rollout_paths)
    true_deter = true_latent[..., DETER_START:].astype(np.float64, copy=False)
    imag_deter = imag_latent[..., DETER_START:].astype(np.float64, copy=False)
    imag_stoch = imag_latent[..., :DETER_START].astype(np.float64, copy=False)

    drift = imag_deter - true_deter
    drift_pool = drift[:, args.trim_start : args.trim_end, :].reshape(-1, DETER_DIM)
    drift_pca = fit_pca(drift_pool)

    eval_pca = fit_pca(true_deter.reshape(-1, DETER_DIM))
    eval_k = n_components_for_variance(
        eval_pca.explained_variance_ratio,
        threshold=args.var_threshold,
    )
    eval_components = eval_pca.components[:eval_k]
    true_projected = project_rollouts(true_deter, eval_pca.mean, eval_components)
    true_bands = filter_all_bands(true_projected)

    rank_results: dict[str, Any] = {}
    global_candidates = []
    for rank in args.ranks:
        u_drift = drift_pca.components[:rank].T
        rank_items = []
        baseline = None
        for eta in args.etas:
            damped_deter = apply_posthoc_damping(true_deter, drift, u_drift, eta)
            damped_latent = np.concatenate([imag_stoch, damped_deter], axis=-1)
            if damped_latent.shape != (N_SEEDS, HORIZON, LATENT_DIM):
                raise RuntimeError(f"Unexpected damped latent shape {damped_latent.shape}.")

            damped_projected = project_rollouts(
                damped_deter,
                eval_pca.mean,
                eval_components,
            )
            damped_bands = filter_all_bands(damped_projected)
            band_mse = band_mse_over_window(
                true_bands,
                damped_bands,
                trim_start=args.trim_start,
                trim_end=args.trim_end,
            )
            total_mse = raw_total_mse(
                true_deter,
                damped_deter,
                trim_start=args.trim_start,
                trim_end=args.trim_end,
            )
            total_band_mse = float(sum(band_mse.values()))

            if baseline is None:
                baseline = {
                    "dc_trend_mse": band_mse["dc_trend"],
                    "total_mse": total_mse,
                }
            dc_change_pct = pct_change(band_mse["dc_trend"], baseline["dc_trend_mse"])
            total_change_pct = pct_change(total_mse, baseline["total_mse"])
            pass_total = bool(total_change_pct <= args.max_total_increase_pct)
            item = {
                "eta": float(eta),
                "band_mse": {band: float(value) for band, value in band_mse.items()},
                "dc_trend_mse": float(band_mse["dc_trend"]),
                "total_mse": float(total_mse),
                "total_band_mse": total_band_mse,
                "dc_trend_change_pct": float(dc_change_pct),
                "dc_trend_reduction_pct": float(-dc_change_pct),
                "total_change_pct": float(total_change_pct),
                "passes_total_guardrail": pass_total,
            }
            rank_items.append(item)
            if pass_total:
                global_candidates.append((rank, item))

        candidate = select_rank_candidate(rank_items)
        rank_results[str(rank)] = {
            "r": int(rank),
            "u_drift_explained_variance_ratio": drift_pca.explained_variance_ratio[
                :rank
            ].tolist(),
            "u_drift_cumulative_explained_variance": float(
                np.sum(drift_pca.explained_variance_ratio[:rank])
            ),
            "eta_results": rank_items,
            "eta_candidate": candidate,
        }

    overall_candidate = select_overall_candidate(global_candidates)
    phase2_go = (
        overall_candidate is not None
        and overall_candidate["dc_trend_reduction_pct"] >= args.go_dc_reduction_pct
    )

    result = {
        "analysis": "smad_eta_sweep",
        "task": "cheetah_run",
        "approximation": (
            "Post-hoc linear damping of cumulative drift along U_drift; "
            "does not re-run RSSM img_step and is not the real SMAD intervention."
        ),
        "rollout_dir": str(args.rollout_dir.relative_to(REPO_ROOT)),
        "rollout_files": [str(path.relative_to(REPO_ROOT)) for path in rollout_paths],
        "n_seeds": N_SEEDS,
        "horizon": HORIZON,
        "latent_dim": LATENT_DIM,
        "deter_dim": DETER_DIM,
        "deter_slice": [DETER_START, LATENT_DIM],
        "trim_window": [args.trim_start, args.trim_end],
        "drift_pool_shape": list(drift_pool.shape),
        "bands": {name: list(bounds) for name, bounds in BANDS.items()},
        "filter": {"type": "butterworth", "order": 4, "fs": 1.0},
        "frequency_eval_pca": {
            "var_threshold": float(args.var_threshold),
            "retained_components": int(eval_k),
            "retained_variance_explained": float(
                np.sum(eval_pca.explained_variance_ratio[:eval_k])
            ),
        },
        "ranks": list(args.ranks),
        "etas": list(args.etas),
        "rank_results": rank_results,
        "eta_star": overall_candidate,
        "phase2_go": bool(phase2_go),
        "phase2_verdict": phase2_verdict(phase2_go, overall_candidate),
        "selection_rule": {
            "candidate": (
                "Maximize dc_trend reduction relative to eta=0 subject to total_mse "
                "not increasing by more than 10%."
            ),
            "go_threshold": "dc_trend reduction >= 20% and total guardrail passes",
            "max_total_increase_pct": float(args.max_total_increase_pct),
            "go_dc_reduction_pct": float(args.go_dc_reduction_pct),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print_summary(result)
    print(f"\nSaved: {args.output.relative_to(REPO_ROOT)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a post-hoc SMAD eta sweep using U_drift from Cheetah rollouts."
    )
    parser.add_argument("--rollout-dir", type=Path, default=ROLLOUT_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--trim-start", type=int, default=TRIM_START)
    parser.add_argument("--trim-end", type=int, default=TRIM_END)
    parser.add_argument("--var-threshold", type=float, default=VAR_THRESHOLD)
    parser.add_argument("--ranks", type=int, nargs="+", default=list(RANKS))
    parser.add_argument("--etas", type=float, nargs="+", default=list(ETAS))
    parser.add_argument(
        "--max-total-increase-pct",
        type=float,
        default=MAX_TOTAL_INCREASE_PCT,
    )
    parser.add_argument(
        "--go-dc-reduction-pct",
        type=float,
        default=GO_DC_REDUCTION_PCT,
    )
    args = parser.parse_args()
    if not 0 <= args.trim_start < args.trim_end <= HORIZON:
        raise ValueError("--trim-start/--trim-end must define a valid horizon window.")
    if not 0 < args.var_threshold <= 1:
        raise ValueError("--var-threshold must be in (0, 1].")
    if any(rank <= 0 or rank > DETER_DIM for rank in args.ranks):
        raise ValueError(f"--ranks must be in [1, {DETER_DIM}].")
    if any(eta < 0 for eta in args.etas):
        raise ValueError("--etas must be non-negative.")
    args.rollout_dir = resolve_path(args.rollout_dir)
    args.output = resolve_path(args.output)
    args.ranks = tuple(args.ranks)
    args.etas = tuple(args.etas)
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
        raise FileNotFoundError(f"Missing Cheetah v2 rollout seeds: {missing}")
    return [by_seed[seed] for seed in range(N_SEEDS)]


def load_rollouts(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    true_latents = []
    imag_latents = []
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            true_latent = require_array(data, "true_latent", path)
            imag_latent = require_array(data, "imagined_latent", path)
        validate_latent(true_latent, "true_latent", path)
        validate_latent(imag_latent, "imagined_latent", path)
        true_latents.append(true_latent.astype(np.float64, copy=False))
        imag_latents.append(imag_latent.astype(np.float64, copy=False))
    return np.stack(true_latents, axis=0), np.stack(imag_latents, axis=0)


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
    return (rollouts - mean.reshape(1, 1, -1)) @ components.T


def apply_posthoc_damping(
    true_deter: np.ndarray,
    drift: np.ndarray,
    u_drift: np.ndarray,
    eta: float,
) -> np.ndarray:
    projected_drift = (drift @ u_drift) @ u_drift.T
    return true_deter + drift - float(eta) * projected_drift


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
    damped_bands: dict[str, np.ndarray],
    *,
    trim_start: int,
    trim_end: int,
) -> dict[str, float]:
    window = slice(trim_start, trim_end)
    result = {}
    for band in BANDS:
        diff = damped_bands[band][:, window, :] - true_bands[band][:, window, :]
        result[band] = float(np.mean(diff**2))
    return result


def raw_total_mse(
    true_deter: np.ndarray,
    damped_deter: np.ndarray,
    *,
    trim_start: int,
    trim_end: int,
) -> float:
    diff = damped_deter[:, trim_start:trim_end, :] - true_deter[:, trim_start:trim_end, :]
    return float(np.mean(diff**2))


def pct_change(value: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0 if value == 0 else float("inf")
    return float((value / baseline - 1.0) * 100.0)


def select_rank_candidate(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    passing = [item for item in items if item["passes_total_guardrail"]]
    if not passing:
        return None
    best = max(passing, key=lambda item: (item["dc_trend_reduction_pct"], -item["eta"]))
    return {
        "eta": best["eta"],
        "dc_trend_reduction_pct": best["dc_trend_reduction_pct"],
        "total_change_pct": best["total_change_pct"],
        "dc_trend_mse": best["dc_trend_mse"],
        "total_mse": best["total_mse"],
    }


def select_overall_candidate(candidates: list[tuple[int, dict[str, Any]]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    rank, best = max(
        candidates,
        key=lambda pair: (pair[1]["dc_trend_reduction_pct"], -pair[0], -pair[1]["eta"]),
    )
    return {
        "r": int(rank),
        "eta": best["eta"],
        "dc_trend_reduction_pct": best["dc_trend_reduction_pct"],
        "total_change_pct": best["total_change_pct"],
        "dc_trend_mse": best["dc_trend_mse"],
        "total_mse": best["total_mse"],
    }


def phase2_verdict(phase2_go: bool, candidate: dict[str, Any] | None) -> str:
    if phase2_go and candidate is not None:
        return (
            "SMAD Phase 2 GO with U_drift basis at "
            f"r={candidate['r']}, eta={candidate['eta']:.2f}. "
            "Scope: 1 configuration, damping+anchor combined."
        )
    return "SMAD Phase 2 NO GO. Pivot to pure diagnostics + theory + toy."


def print_summary(result: dict[str, Any]) -> None:
    print("SMAD Pre-check 2: post-hoc eta sweep with U_drift basis")
    print("Approximation: post-hoc damping of cumulative drift; RSSM is not re-run.")
    print(f"Frequency PCA retained {result['frequency_eval_pca']['retained_components']} PCs")
    print(f"Trim window: {tuple(result['trim_window'])}")
    print()
    for rank in result["ranks"]:
        print(f"r={rank}")
        print("eta   dc_trend_mse   total_mse   dc_change%   total_change%   pass")
        print("----  ------------   ---------   ----------   -------------   ----")
        for item in result["rank_results"][str(rank)]["eta_results"]:
            print(
                f"{item['eta']:<4.2f}  "
                f"{item['dc_trend_mse']:<12.6f}   "
                f"{item['total_mse']:<9.6f}   "
                f"{item['dc_trend_change_pct']:<10.2f}   "
                f"{item['total_change_pct']:<13.2f}   "
                f"{'PASS' if item['passes_total_guardrail'] else 'FAIL'}"
            )
        candidate = result["rank_results"][str(rank)]["eta_candidate"]
        if candidate is None:
            print("candidate: none")
        else:
            print(
                "candidate: "
                f"eta={candidate['eta']:.2f}, "
                f"dc_reduction={candidate['dc_trend_reduction_pct']:.2f}%, "
                f"total_change={candidate['total_change_pct']:.2f}%"
            )
        print()
    print(f"eta*: {result['eta_star']}")
    print(f"Verdict: {result['phase2_verdict']}")


if __name__ == "__main__":
    main()
