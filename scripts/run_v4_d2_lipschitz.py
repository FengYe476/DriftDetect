#!/usr/bin/env python3
"""Run the DreamerV4 Decoder Amplification D2 Lipschitz probe.

This is the V4 analogue of scripts/run_decoder_d2.py. It perturbs flattened
packed tokenizer latents along frequency-band directions and measures the image
L2 response of the frozen tokenizer decoder.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DREAMER4_ROOT = REPO_ROOT / "external" / "dreamer4"
DREAMER4_PKG_ROOT = DREAMER4_ROOT / "dreamer4"
DEFAULT_ROLLOUT_DIR = REPO_ROOT / "results" / "rollouts_v4"
DEFAULT_OUTPUT_NPZ = REPO_ROOT / "results" / "tables" / "v4_decoder_d2_results.npz"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "results" / "tables" / "v4_decoder_d2_results.json"
DEFAULT_TASK = "cheetah-run"
DEFAULT_NUM_SEEDS = 20
DEFAULT_HORIZON = 200
DEFAULT_ANCHOR_STEPS = (50, 100, 150)
DEFAULT_EPSILONS = (0.01, 0.05, 0.1, 0.5, 1.0)
DEFAULT_PACKED_SHAPE = (8, 64)
V3_LIPSCHITZ_EPS1 = {
    "dc_trend": 1.221,
    "very_low": 1.666,
    "low": 2.140,
    "mid": 2.344,
    "high": 2.168,
}

for path in (REPO_ROOT, DREAMER4_ROOT, DREAMER4_PKG_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.diagnostics.freq_decompose import DEFAULT_BANDS, bandpass  # noqa: E402
from src.models.dreamerv4_adapter import DreamerV4Adapter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe DreamerV4 tokenizer decoder sensitivity to band-filtered "
            "packed-latent perturbation directions."
        )
    )
    parser.add_argument(
        "--checkpoint_dir",
        required=True,
        help="Directory containing tokenizer.pt and dynamics.pt.",
    )
    parser.add_argument("--task", default=DEFAULT_TASK, help="V4 task name.")
    parser.add_argument(
        "--rollout-dir",
        type=Path,
        default=DEFAULT_ROLLOUT_DIR,
        help="Directory containing cheetah-run_v4_seed{seed}_v1.npz rollouts.",
    )
    parser.add_argument("--num-seeds", type=int, default=DEFAULT_NUM_SEEDS)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument(
        "--anchor-steps",
        type=int,
        nargs="+",
        default=list(DEFAULT_ANCHOR_STEPS),
    )
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=list(DEFAULT_EPSILONS),
    )
    parser.add_argument(
        "--filter-type",
        choices=["butterworth", "chebyshev1", "fir"],
        default="butterworth",
    )
    parser.add_argument("--min-direction-norm", type=float, default=1e-8)
    parser.add_argument("--schedule", default="shortcut", choices=("shortcut", "finest"))
    parser.add_argument("--schedule_d", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_NPZ)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_OUTPUT_JSON)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("V4 D2 is CUDA-only. Run this script on RunPod with CUDA.")
    if args.num_seeds <= 0:
        raise ValueError("--num-seeds must be positive.")
    validate_anchor_steps(args.anchor_steps, args.horizon)
    if any(epsilon <= 0 for epsilon in args.epsilons):
        raise ValueError("All epsilons must be positive.")

    args.checkpoint_dir = resolve_path(args.checkpoint_dir)
    args.rollout_dir = resolve_path(args.rollout_dir)
    args.output = resolve_path(args.output)
    args.json_output = resolve_path(args.json_output)
    return args


def resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def rollout_path_for(rollout_dir: Path, task: str, seed: int) -> Path:
    return rollout_dir / f"{task}_v4_seed{seed}_v1.npz"


def validate_anchor_steps(anchor_steps: list[int], horizon: int) -> None:
    if not anchor_steps:
        raise ValueError("At least one anchor step is required.")
    invalid = [step for step in anchor_steps if step < 0 or step >= horizon]
    if invalid:
        raise ValueError(f"Anchor steps must be in [0, {horizon - 1}], got {invalid}.")


def validate_true_latent(
    path: Path,
    latent: np.ndarray,
    *,
    key: str,
    expected_horizon: int,
    expected_latent_dim: int,
) -> None:
    if latent.ndim != 2:
        raise ValueError(f"{display_path(path)} expected 2D {key}.")
    if latent.shape != (expected_horizon, expected_latent_dim):
        raise ValueError(
            f"{display_path(path)} expected {key} shape "
            f"({expected_horizon}, {expected_latent_dim}), got {latent.shape}."
        )


def band_filter_latent(
    latent: np.ndarray,
    *,
    band_names: list[str],
    filter_type: str,
) -> dict[str, np.ndarray]:
    filtered_by_band = {}
    for band in band_names:
        low, high = DEFAULT_BANDS[band]
        filtered = np.empty_like(latent, dtype=np.float32)
        for dim in range(latent.shape[1]):
            filtered[:, dim] = bandpass(
                latent[:, dim],
                low=low,
                high=high,
                filter_type=filter_type,
            ).astype(np.float32)
        filtered_by_band[band] = filtered
    return filtered_by_band


def direction_for_anchor(
    filtered_latent: np.ndarray,
    anchor_step: int,
    *,
    min_norm: float,
) -> tuple[np.ndarray, float]:
    direction = filtered_latent[anchor_step] - filtered_latent.mean(axis=0)
    norm = float(np.linalg.norm(direction))
    if norm < min_norm:
        return np.full(direction.shape, np.nan, dtype=np.float32), norm
    return (direction / norm).astype(np.float32), norm


def probe_anchor(
    adapter: DreamerV4Adapter,
    z_anchor: np.ndarray,
    directions: dict[str, np.ndarray],
    *,
    band_names: list[str],
    epsilons: np.ndarray,
    packed_shape: tuple[int, int],
) -> np.ndarray:
    candidates = [z_anchor.astype(np.float32, copy=True)]
    candidate_index: list[tuple[int, int]] = []

    for band_index, band in enumerate(band_names):
        direction = directions[band]
        for epsilon_index, epsilon in enumerate(epsilons):
            z_perturbed = z_anchor.astype(np.float32, copy=True)
            if np.all(np.isfinite(direction)):
                z_perturbed += float(epsilon) * direction
            else:
                z_perturbed[:] = np.nan
            candidates.append(z_perturbed)
            candidate_index.append((band_index, epsilon_index))

    candidate_array = np.stack(candidates, axis=0)
    valid_mask = np.isfinite(candidate_array).all(axis=1)
    response = np.full((len(band_names), len(epsilons)), np.nan, dtype=np.float32)

    if not valid_mask[0]:
        return response

    valid_candidates = candidate_array[valid_mask].reshape(
        int(valid_mask.sum()),
        packed_shape[0],
        packed_shape[1],
    )
    with torch.no_grad():
        decoded_valid = adapter.decode(valid_candidates)
    decoded_flat = decoded_valid.reshape(decoded_valid.shape[0], -1)

    decoded_lookup: dict[int, np.ndarray] = {}
    valid_indices = np.flatnonzero(valid_mask)
    for row, original_index in enumerate(valid_indices):
        decoded_lookup[int(original_index)] = decoded_flat[row]

    anchor_decoded = decoded_lookup[0]
    for candidate_offset, (band_index, epsilon_index) in enumerate(
        candidate_index,
        start=1,
    ):
        if candidate_offset not in decoded_lookup:
            continue
        delta = np.linalg.norm(decoded_lookup[candidate_offset] - anchor_decoded)
        response[band_index, epsilon_index] = float(delta)

    return response


def summarize(
    anchor_responses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.sum(np.isfinite(anchor_responses), axis=0).astype(np.int64)
    mean = np.nanmean(anchor_responses, axis=0).astype(np.float32)
    std = np.zeros_like(mean, dtype=np.float32)
    for band_index in range(anchor_responses.shape[1]):
        for epsilon_index in range(anchor_responses.shape[2]):
            values = anchor_responses[:, band_index, epsilon_index]
            values = values[np.isfinite(values)]
            if values.size > 1:
                std[band_index, epsilon_index] = float(np.std(values, ddof=1))
    return mean, std, counts


def print_summary_table(
    obs_delta: np.ndarray,
    lipschitz_at_eps1: np.ndarray,
    *,
    band_names: list[str],
    epsilons: np.ndarray,
) -> None:
    print()
    print("V4 DECODER D2 SUMMARY: MEAN IMAGE L2 RESPONSE")
    header = f"{'band':<12}" + "".join(f"  eps={epsilon:<8g}" for epsilon in epsilons)
    header += "  Lipschitz(eps=1.0)"
    print(header)
    for band_index, band in enumerate(band_names):
        row = f"{band:<12}"
        for epsilon_index in range(len(epsilons)):
            row += f"  {obs_delta[band_index, epsilon_index]:<12.6f}"
        row += f"  {lipschitz_at_eps1[band_index]:<12.6f}"
        print(row)


def write_json_summary(
    path: Path,
    *,
    obs_delta: np.ndarray,
    obs_delta_std: np.ndarray,
    lipschitz_at_eps1: np.ndarray,
    band_names: list[str],
    epsilons: np.ndarray,
    n_anchors: int,
    args: argparse.Namespace,
    adapter: DreamerV4Adapter,
) -> None:
    payload: dict[str, Any] = {
        "analysis": "v4_decoder_d2_lipschitz",
        "checkpoint_dir": display_path(args.checkpoint_dir),
        "rollout_dir": display_path(args.rollout_dir),
        "task": args.task,
        "n_anchors": int(n_anchors),
        "anchor_steps": [int(step) for step in args.anchor_steps],
        "epsilons": epsilons.astype(float).tolist(),
        "band_names": band_names,
        "obs_delta": {
            band: {
                f"eps_{float(epsilon):g}": float(obs_delta[band_index, epsilon_index])
                for epsilon_index, epsilon in enumerate(epsilons)
            }
            for band_index, band in enumerate(band_names)
        },
        "obs_delta_std": {
            band: {
                f"eps_{float(epsilon):g}": float(obs_delta_std[band_index, epsilon_index])
                for epsilon_index, epsilon in enumerate(epsilons)
            }
            for band_index, band in enumerate(band_names)
        },
        "lipschitz_at_eps1": {
            band: float(lipschitz_at_eps1[band_index])
            for band_index, band in enumerate(band_names)
        },
        "v3_lipschitz_at_eps1_reference": V3_LIPSCHITZ_EPS1,
        "packed_shape": list(infer_packed_shape(adapter)),
        "adapter_config": adapter.config_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def infer_packed_shape(adapter: DreamerV4Adapter) -> tuple[int, int]:
    if adapter.n_spatial is not None and adapter.d_spatial is not None:
        return int(adapter.n_spatial), int(adapter.d_spatial)
    return DEFAULT_PACKED_SHAPE


def main() -> None:
    args = parse_args()
    seeds = np.arange(args.num_seeds, dtype=np.int64)
    epsilons = np.asarray(args.epsilons, dtype=np.float32)
    band_names = list(DEFAULT_BANDS.keys())

    print("Loading DreamerV4 adapter")
    print(f"Checkpoint dir: {display_path(args.checkpoint_dir)}")
    print(f"Rollout dir: {display_path(args.rollout_dir)}")
    print(f"Task: {args.task}")
    print(f"Evaluating seeds: {seeds[0]}-{seeds[-1]}")
    print(f"Anchor steps: {args.anchor_steps}")
    print(f"Epsilons: {epsilons.tolist()}")
    print(f"Filter type: {args.filter_type}")

    adapter = DreamerV4Adapter(
        task=args.task,
        schedule=args.schedule,
        schedule_d=args.schedule_d,
        device="cuda",
    )
    adapter.load_checkpoint(str(args.checkpoint_dir))
    assert adapter.decoder is not None
    adapter.decoder.requires_grad_(False)
    adapter.decoder.eval()

    packed_shape = infer_packed_shape(adapter)
    latent_dim = packed_shape[0] * packed_shape[1]
    if packed_shape != DEFAULT_PACKED_SHAPE:
        print(f"Warning: packed shape is {packed_shape}, expected {DEFAULT_PACKED_SHAPE}.")

    anchor_responses: list[np.ndarray] = []
    direction_norms: list[np.ndarray] = []
    rollout_files = []

    for seed in seeds:
        rollout_path = rollout_path_for(args.rollout_dir, args.task, int(seed))
        if not rollout_path.exists():
            raise FileNotFoundError(f"Missing rollout file: {display_path(rollout_path)}")
        rollout_files.append(display_path(rollout_path))
        print(f"  seed {seed}: filtering band directions", flush=True)
        with np.load(rollout_path, allow_pickle=True) as rollout:
            true_latent = rollout["true_latent"].astype(np.float32, copy=False)
            imagined_latent = rollout["imagined_latent"].astype(np.float32, copy=False)

        validate_true_latent(
            rollout_path,
            true_latent,
            key="true_latent",
            expected_horizon=args.horizon,
            expected_latent_dim=latent_dim,
        )
        validate_true_latent(
            rollout_path,
            imagined_latent,
            key="imagined_latent",
            expected_horizon=args.horizon,
            expected_latent_dim=latent_dim,
        )
        filtered_by_band = band_filter_latent(
            true_latent,
            band_names=band_names,
            filter_type=args.filter_type,
        )

        for anchor_step in args.anchor_steps:
            directions = {}
            norms = np.empty(len(band_names), dtype=np.float32)
            for band_index, band in enumerate(band_names):
                direction, norm = direction_for_anchor(
                    filtered_by_band[band],
                    anchor_step,
                    min_norm=args.min_direction_norm,
                )
                directions[band] = direction
                norms[band_index] = norm

            response = probe_anchor(
                adapter,
                true_latent[anchor_step],
                directions,
                band_names=band_names,
                epsilons=epsilons,
                packed_shape=packed_shape,
            )
            anchor_responses.append(response)
            direction_norms.append(norms)

    if not anchor_responses:
        raise RuntimeError("No anchor responses were collected.")

    anchor_response_array = np.stack(anchor_responses, axis=0)
    direction_norm_array = np.stack(direction_norms, axis=0)
    obs_delta, obs_delta_std, obs_delta_counts = summarize(anchor_response_array)
    n_anchors = int(anchor_response_array.shape[0])

    eps1_matches = np.where(np.isclose(epsilons, 1.0))[0]
    if eps1_matches.size == 0:
        raise ValueError("epsilons must include 1.0 to report Lipschitz(eps=1.0).")
    eps1_index = int(eps1_matches[0])
    lipschitz_at_eps1 = obs_delta[:, eps1_index] / float(epsilons[eps1_index])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        obs_delta=obs_delta,
        obs_delta_std=obs_delta_std,
        band_names=np.asarray(band_names),
        epsilons=epsilons,
        lipschitz_at_eps1=lipschitz_at_eps1.astype(np.float32),
        n_anchors=np.asarray(n_anchors, dtype=np.int64),
        anchor_steps=np.asarray(args.anchor_steps, dtype=np.int64),
        obs_delta_counts=obs_delta_counts,
        direction_norms=direction_norm_array,
        rollout_files=np.asarray(rollout_files),
    )
    write_json_summary(
        args.json_output,
        obs_delta=obs_delta,
        obs_delta_std=obs_delta_std,
        lipschitz_at_eps1=lipschitz_at_eps1,
        band_names=band_names,
        epsilons=epsilons,
        n_anchors=n_anchors,
        args=args,
        adapter=adapter,
    )

    print_summary_table(
        obs_delta,
        lipschitz_at_eps1,
        band_names=band_names,
        epsilons=epsilons,
    )
    print()
    print("V3 D2 Lipschitz at eps=1.0 (from Month 3):")
    print(
        "dc_trend=1.221, very_low=1.666, low=2.140, mid=2.344, high=2.168"
    )
    if np.any(obs_delta_counts < n_anchors):
        print()
        print(
            "Warning: some band directions were below the minimum norm; "
            "see obs_delta_counts in the saved file."
        )
    print()
    print(f"Anchors used: {n_anchors}")
    print(f"Saved NPZ: {display_path(args.output)}")
    print(f"Saved JSON: {display_path(args.json_output)}")


if __name__ == "__main__":
    main()
