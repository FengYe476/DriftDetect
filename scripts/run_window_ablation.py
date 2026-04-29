"""Run the imagination window ablation for DriftDetect Month 2 Week 2.

This script extracts new latent-aware rollouts for imagination starts 50 and
500, reuses the existing start=200 v2 rollouts for seeds 0-4, and prints
per-band error curve comparisons in latent and observation space.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = REPO_ROOT / "results" / "checkpoints" / "cheetah_run_500k.pt"
OUTPUT_DIR = REPO_ROOT / "results" / "rollouts" / "window_ablation"

sys.path.insert(0, str(REPO_ROOT))

from src.diagnostics.error_curves import aggregate_curves  # noqa: E402
from src.models.dreamerv3_adapter import DreamerV3Adapter  # noqa: E402


ABLATION_STARTS = [50, 200, 500]
ABLATION_SEEDS = range(5)
TOTAL_STEPS = 1000
HORIZON = 200
TASK = "dmc_cheetah_run"
TABLE_STEPS = (25, 100, 175)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def save_rollout(path: Path, seed: int, imagination_start: int, rollout: dict[str, np.ndarray]) -> None:
    metadata = {
        "version": "v2_window_ablation",
        "task": TASK,
        "seed": seed,
        "total_steps": TOTAL_STEPS,
        "imagination_start": imagination_start,
        "horizon": HORIZON,
        "agent_type": "dreamerv3_trained_checkpoint",
        "checkpoint_path": display_path(CHECKPOINT),
        "obs_alignment": (
            "true and imagined arrays contain next observations for actions "
            f"at steps {imagination_start}-{imagination_start + HORIZON - 1}"
        ),
        "latent_alignment": (
            "true_latent and imagined_latent contain deterministic RSSM features "
            f"for steps {imagination_start + 1}-{imagination_start + HORIZON}"
        ),
    }
    np.savez(
        path,
        true_obs=rollout["true_obs"].astype(np.float32),
        imagined_obs=rollout["imagined_obs"].astype(np.float32),
        actions=rollout["actions"].astype(np.float32),
        rewards=rollout["rewards"].astype(np.float32),
        true_latent=rollout["true_latent"].astype(np.float32),
        imagined_latent=rollout["imagined_latent"].astype(np.float32),
        metadata=np.array(metadata, dtype=object),
    )


def output_path_for(start: int, seed: int) -> Path:
    return OUTPUT_DIR / f"cheetah_v3_start{start}_seed{seed}_v2.npz"


def existing_start_200_path(seed: int) -> Path:
    return REPO_ROOT / "results" / "rollouts" / f"cheetah_v3_seed{seed}_v2.npz"


def ensure_extractions() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    adapter = DreamerV3Adapter(task=TASK)
    adapter.load_checkpoint(str(CHECKPOINT))

    for start in ABLATION_STARTS:
        if start == 200:
            continue
        for seed in ABLATION_SEEDS:
            output = output_path_for(start, seed)
            if output.exists():
                print(
                    f"Skipping start={start}, seed={seed}: {display_path(output)} already exists.",
                    flush=True,
                )
                continue
            print(f"Extracting start={start}, seed={seed}...", flush=True)
            rollout = adapter.extract_rollout(
                seed=seed,
                total_steps=TOTAL_STEPS,
                imagination_start=start,
                horizon=HORIZON,
                include_latent=True,
            )
            save_rollout(output, seed, start, rollout)
            print(f"Saved {display_path(output)}.", flush=True)


def rollout_paths_for_start(start: int) -> list[Path]:
    if start == 200:
        return [existing_start_200_path(seed) for seed in ABLATION_SEEDS]
    return [output_path_for(start, seed) for seed in ABLATION_SEEDS]


def run_analysis() -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    latent_results: dict[int, dict[str, Any]] = {}
    obs_results: dict[int, dict[str, Any]] = {}
    for start in ABLATION_STARTS:
        paths = rollout_paths_for_start(start)
        latent_results[start] = aggregate_curves(
            rollout_paths=paths,
            signal_key_true="true_latent",
            signal_key_imag="imagined_latent",
            metric="mse",
        )
        obs_results[start] = aggregate_curves(
            rollout_paths=paths,
            signal_key_true="true_obs",
            signal_key_imag="imagined_obs",
            metric="mse",
        )
    return latent_results, obs_results


def print_table(title: str, results_by_start: dict[int, dict[str, Any]]) -> None:
    print(title)
    print()
    print(
        "              start=50                    start=200                   start=500"
    )
    print(
        "Band       step25   step100  step175   step25   step100  step175   step25   step100  step175"
    )
    for band in band_names(results_by_start[50]):
        row = f"{band:10s}"
        for start in ABLATION_STARTS:
            mean = results_by_start[start][band]["mean"]
            row += "".join(f"{mean[step]:9.4f}" for step in TABLE_STEPS)
        print(row)
    print()


def print_consistency(title: str, results_by_start: dict[int, dict[str, Any]]) -> None:
    print(title)
    for band in band_names(results_by_start[50]):
        ratios = {
            start: safe_ratio(
                results_by_start[start][band]["mean"][175],
                results_by_start[start][band]["mean"][25],
            )
            for start in ABLATION_STARTS
        }
        finite_ratios = [value for value in ratios.values() if np.isfinite(value)]
        if not finite_ratios:
            status = "CONSISTENT"
        else:
            status = (
                "CONSISTENT"
                if max(finite_ratios) / max(min(finite_ratios), 1e-8) <= 2.0
                else "INCONSISTENT - investigate"
            )
        ratio_text = "  ".join(f"start={start}:{ratios[start]:.2f}" for start in ABLATION_STARTS)
        print(f"  {band:10s}: {ratio_text}  -> {status}")
    print()


def safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-8 and abs(numerator) < 1e-8:
        return 1.0
    if abs(denominator) < 1e-8:
        return float("inf")
    return float(numerator / denominator)


def band_names(result: dict[str, Any]) -> list[str]:
    return [key for key in result if key != "_info"]


def main() -> None:
    print(f"Using scipy in the active environment for extraction+analysis.", flush=True)
    ensure_extractions()
    latent_results, obs_results = run_analysis()

    print()
    print("=== LATENT-SPACE PER-BAND MSE (5 seeds, horizon=200) ===")
    print()
    print_table("", latent_results)
    print_consistency("Latent-space qualitative consistency:", latent_results)

    print("=== OBS-SPACE PER-BAND MSE (5 seeds, horizon=200) ===")
    print()
    print_table("", obs_results)
    print_consistency("Obs-space qualitative consistency:", obs_results)


if __name__ == "__main__":
    main()
