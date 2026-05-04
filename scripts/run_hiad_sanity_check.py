#!/usr/bin/env python3
"""HIAD sanity check: test whether Delta z_t PCA aligns with U_drift.

This is the Month 8 Week 1 Day 5 decision-point script. The result determines
whether HIAD development continues or the project pivots to Path A.

Usage:
    python scripts/run_hiad_sanity_check.py \
        --rollout_dir results/rollouts/ \
        --u_fixed results/smad/U_drift_cheetah_r10.npy \
        --rank 10 \
        --window_sizes 25 50 100 200 \
        --output results/tables/hiad_sanity_check.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.smad.U_estimation import DETER_DIM, DETER_START  # noqa: E402
from src.smad.hiad import SelfConsistencyDriftEstimator  # noqa: E402


ROLLOUT_DIR = REPO_ROOT / "results" / "rollouts"
U_FIXED_PATH = REPO_ROOT / "results" / "smad" / "U_drift_cheetah_r10.npy"
OUTPUT_PATH = REPO_ROOT / "results" / "tables" / "hiad_sanity_check.json"
ROLLOUT_RE = re.compile(r"cheetah_v3_seed(\d+)_v2\.npz$")
N_SEEDS = 20
DEFAULT_WINDOW_SIZES = (25, 50, 100, 200)
PROCEED_THRESHOLD = 0.7
MARGINAL_THRESHOLD = 0.3


def main() -> None:
    args = parse_args()
    rollout_paths = discover_rollouts(args.rollout_dir)
    U_fixed = load_fixed_basis(args.u_fixed, args.rank)

    results_by_window: dict[str, dict[str, Any]] = {}
    per_seed_by_window: dict[str, list[dict[str, Any]]] = {}
    for window_size in args.window_sizes:
        result, per_seed = evaluate_window_size(
            rollout_paths=rollout_paths,
            U_fixed=U_fixed,
            rank=args.rank,
            window_size=window_size,
        )
        results_by_window[str(window_size)] = result
        per_seed_by_window[str(window_size)] = per_seed

    full_result, full_per_seed = evaluate_full_rollout(
        rollout_paths=rollout_paths,
        U_fixed=U_fixed,
        rank=args.rank,
    )
    results_by_window["full"] = full_result
    per_seed_by_window["full"] = full_per_seed

    verdict = verdict_for_overlap(full_result["mean_overlap"])
    output = {
        "u_fixed_path": display_path(args.u_fixed),
        "rollout_dir": display_path(args.rollout_dir),
        "rollout_files": [display_path(path) for path in rollout_paths],
        "rank": int(args.rank),
        "n_seeds": len(rollout_paths),
        "segment_mode": "non_overlapping",
        "results_by_window": results_by_window,
        "per_seed_default_window": per_seed_by_window.get("25", []),
        "verdict": verdict,
        "threshold_used": {
            "proceed": PROCEED_THRESHOLD,
            "marginal": MARGINAL_THRESHOLD,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")

    print_summary(
        result=output,
        default_per_seed=per_seed_by_window.get("25", []),
        output_path=args.output,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the HIAD Delta z_t PCA vs U_drift sanity check."
    )
    parser.add_argument("--rollout_dir", type=Path, default=ROLLOUT_DIR)
    parser.add_argument("--u_fixed", type=Path, default=U_FIXED_PATH)
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument(
        "--window_sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_WINDOW_SIZES),
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    args.rollout_dir = resolve_path(args.rollout_dir)
    args.u_fixed = resolve_path(args.u_fixed)
    args.output = resolve_path(args.output)
    if args.rank <= 0:
        raise ValueError("--rank must be positive.")
    for window_size in args.window_sizes:
        if window_size <= 0:
            raise ValueError("--window_sizes entries must be positive.")
        if window_size < args.rank:
            raise ValueError(
                f"window size {window_size} must be >= rank {args.rank}."
            )
    return args


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def display_path(path: Path) -> str:
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


def load_fixed_basis(path: Path, rank: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Fixed U basis not found: {display_path(path)}")
    U = np.load(path)
    if U.ndim != 2 or U.shape[0] != DETER_DIM:
        raise ValueError(f"U_fixed must have shape ({DETER_DIM}, r), got {U.shape}.")
    if U.shape[1] < rank:
        raise ValueError(f"U_fixed rank {U.shape[1]} is smaller than requested {rank}.")
    return U[:, :rank].astype(np.float64, copy=False)


def evaluate_window_size(
    *,
    rollout_paths: list[Path],
    U_fixed: np.ndarray,
    rank: int,
    window_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    all_overlaps = []
    per_seed = []
    effective_window_sizes = []
    for path in rollout_paths:
        seed = seed_from_path(path)
        deltas = load_imagined_deter_deltas(path)
        effective_window = min(window_size, deltas.shape[0])
        if effective_window < rank:
            raise ValueError(
                f"{path.name}: effective window {effective_window} < rank {rank}."
            )
        overlaps = []
        for start in range(0, deltas.shape[0] - effective_window + 1, effective_window):
            segment = deltas[start : start + effective_window]
            overlaps.append(overlap_for_segment(segment, U_fixed, rank))
        seed_stats = summarize_overlaps(overlaps)
        seed_stats.update(
            {
                "seed": int(seed),
                "n_windows": len(overlaps),
                "effective_window_size": int(effective_window),
            }
        )
        per_seed.append(seed_stats)
        all_overlaps.extend(overlaps)
        effective_window_sizes.append(effective_window)

    result = summarize_overlaps(all_overlaps)
    result.update(
        {
            "window_size": int(window_size),
            "effective_window_size_min": int(min(effective_window_sizes)),
            "effective_window_size_max": int(max(effective_window_sizes)),
            "n_seeds": len(rollout_paths),
            "n_windows": len(all_overlaps),
        }
    )
    return result, per_seed


def evaluate_full_rollout(
    *,
    rollout_paths: list[Path],
    U_fixed: np.ndarray,
    rank: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    overlaps = []
    per_seed = []
    full_lengths = []
    for path in rollout_paths:
        seed = seed_from_path(path)
        deltas = load_imagined_deter_deltas(path)
        if deltas.shape[0] < rank:
            raise ValueError(f"{path.name}: full rollout has fewer than rank deltas.")
        overlap = overlap_for_segment(deltas, U_fixed, rank)
        overlaps.append(overlap)
        full_lengths.append(deltas.shape[0])
        per_seed.append(
            {
                "seed": int(seed),
                "mean_overlap": float(overlap),
                "std_overlap": 0.0,
                "min": float(overlap),
                "max": float(overlap),
                "n_windows": 1,
                "effective_window_size": int(deltas.shape[0]),
            }
        )

    result = summarize_overlaps(overlaps)
    result.update(
        {
            "window_size": "full",
            "effective_window_size_min": int(min(full_lengths)),
            "effective_window_size_max": int(max(full_lengths)),
            "n_seeds": len(rollout_paths),
            "n_windows": len(overlaps),
        }
    )
    return result, per_seed


def load_imagined_deter_deltas(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=True) as data:
        imagined_deter = extract_imagined_deter(data, path)
    if imagined_deter.shape[0] < 2:
        raise ValueError(f"{path.name}: imagined deter trajectory is too short.")
    return np.diff(imagined_deter, axis=0).astype(np.float64, copy=False)


def extract_imagined_deter(data: np.lib.npyio.NpzFile, path: Path) -> np.ndarray:
    if "imagined_deter" in data:
        array = np.asarray(data["imagined_deter"])
    elif "imagined_latent" in data:
        array = np.asarray(data["imagined_latent"])
    elif "imagined_latents" in data:
        array = np.asarray(data["imagined_latents"])
    else:
        raise KeyError(
            f"{path} does not contain imagined_deter, imagined_latent, "
            "or imagined_latents."
        )
    if array.ndim != 2:
        raise ValueError(f"{path.name}: imagined latent must be 2D, got {array.shape}.")
    if array.shape[1] == DETER_DIM:
        return array.astype(np.float64, copy=False)
    if array.shape[1] >= DETER_START + DETER_DIM:
        return array[:, DETER_START : DETER_START + DETER_DIM].astype(
            np.float64,
            copy=False,
        )
    raise ValueError(
        f"{path.name}: cannot extract deter from imagined latent shape {array.shape}."
    )


def overlap_for_segment(segment: np.ndarray, U_fixed: np.ndarray, rank: int) -> float:
    estimator = SelfConsistencyDriftEstimator(
        rank=rank,
        window_size=segment.shape[0],
        min_buffer_size=rank,
        deter_dim=DETER_DIM,
    )
    for delta_z in segment:
        estimator.record(delta_z)
    U_hiad = estimator.estimate_U()
    overlap = float(np.linalg.norm(U_fixed.T @ U_hiad, ord="fro") ** 2 / rank)
    return float(np.clip(overlap, 0.0, 1.0))


def summarize_overlaps(overlaps: list[float]) -> dict[str, Any]:
    if not overlaps:
        raise ValueError("No overlaps were computed.")
    values = np.asarray(overlaps, dtype=np.float64)
    return {
        "mean_overlap": float(np.mean(values)),
        "std_overlap": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def seed_from_path(path: Path) -> int:
    match = ROLLOUT_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Cannot parse seed from {path.name}.")
    return int(match.group(1))


def verdict_for_overlap(overlap: float) -> str:
    if overlap > PROCEED_THRESHOLD:
        return "PROCEED"
    if overlap >= MARGINAL_THRESHOLD:
        return "MARGINAL"
    return "PIVOT"


def print_summary(
    *,
    result: dict[str, Any],
    default_per_seed: list[dict[str, Any]],
    output_path: Path,
) -> None:
    print("HIAD sanity check: Delta z_t PCA vs U_drift")
    print(f"Rollouts: {result['n_seeds']} Cheetah seeds")
    print(f"U_fixed: {result['u_fixed_path']}")
    print(f"Rank: {result['rank']}")
    print()
    print("window   mean      std       min       max       n_windows")
    print("------   --------  --------  --------  --------  ---------")
    for window, values in result["results_by_window"].items():
        print(
            f"{window:<6}   "
            f"{values['mean_overlap']:.4f}    "
            f"{values['std_overlap']:.4f}    "
            f"{values['min']:.4f}    "
            f"{values['max']:.4f}    "
            f"{values['n_windows']}"
        )

    if default_per_seed:
        print()
        print("Per-seed overlaps for window_size=25")
        print("seed   mean      std       n_windows")
        print("----   --------  --------  ---------")
        for item in default_per_seed:
            print(
                f"{item['seed']:<4d}   "
                f"{item['mean_overlap']:.4f}    "
                f"{item['std_overlap']:.4f}    "
                f"{item['n_windows']}"
            )

    verdict = result["verdict"]
    if verdict == "PROCEED":
        print("\nVERDICT: PROCEED - HIAD proxy recovers drift geometry")
    elif verdict == "MARGINAL":
        print("\nVERDICT: MARGINAL - run sensitivity before scaling")
    else:
        print("\nVERDICT: PIVOT - proxy mismatched, consider Path A")
    print(f"Saved: {display_path(output_path)}")


if __name__ == "__main__":
    main()
