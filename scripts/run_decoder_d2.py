"""Run the Decoder Amplification D2 Lipschitz probe experiment.

This probes the trained DreamerV3 decoder by perturbing true RSSM latents along
frequency-band directions in the continuous deter state, then measuring the
L2 response in decoded observation space.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = REPO_ROOT / "results" / "checkpoints" / "cheetah_run_500k.pt"
DEFAULT_ROLLOUT_DIR = REPO_ROOT / "results" / "rollouts"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "tables" / "decoder_d2_results.npz"
DEFAULT_TASK = "dmc_cheetah_run"
DEFAULT_NUM_SEEDS = 20
DEFAULT_HORIZON = 200
DEFAULT_ANCHOR_STEPS = (50, 100, 150)
DEFAULT_EPSILONS = (0.01, 0.05, 0.1, 0.5, 1.0)

sys.path.insert(0, str(REPO_ROOT))

from src.diagnostics.freq_decompose import (  # noqa: E402
    DEFAULT_BANDS,
    DETER_SLICE,
    bandpass,
    extract_deter,
)
from src.models.dreamerv3_adapter import DreamerV3Adapter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe trained DreamerV3 decoder sensitivity to band-filtered "
            "deter-state perturbation directions."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to the trained DreamerV3 checkpoint.",
    )
    parser.add_argument(
        "--rollout-dir",
        type=Path,
        default=DEFAULT_ROLLOUT_DIR,
        help="Directory containing cheetah_v3_seed{seed}_v2.npz rollouts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output .npz path for aggregated D2 results.",
    )
    parser.add_argument(
        "--task",
        default=DEFAULT_TASK,
        help="DreamerV3 task name used to build the model architecture.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=DEFAULT_NUM_SEEDS,
        help="Number of rollout seeds to evaluate, starting from seed 0.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help="Expected rollout horizon in each .npz file.",
    )
    parser.add_argument(
        "--anchor-steps",
        type=int,
        nargs="+",
        default=list(DEFAULT_ANCHOR_STEPS),
        help="Reference steps from true_latent to use as anchor points.",
    )
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=list(DEFAULT_EPSILONS),
        help="Perturbation magnitudes to test along each band direction.",
    )
    parser.add_argument(
        "--filter-type",
        choices=["butterworth", "chebyshev1", "fir"],
        default="butterworth",
        help="Temporal filter used for band directions.",
    )
    parser.add_argument(
        "--min-direction-norm",
        type=float,
        default=1e-8,
        help="Minimum norm required before normalizing a band direction.",
    )
    return parser.parse_args()


def select_device() -> str:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def feat_size_from_config(config: object) -> int:
    dyn_discrete = getattr(config, "dyn_discrete")
    dyn_stoch = getattr(config, "dyn_stoch")
    dyn_deter = getattr(config, "dyn_deter")
    if dyn_discrete:
        return int(dyn_stoch * dyn_discrete + dyn_deter)
    return int(dyn_stoch + dyn_deter)


def decode_flat_batch(
    decoder: torch.nn.Module,
    latent_batch: torch.Tensor,
    obs_keys: tuple[str, ...],
) -> torch.Tensor:
    decoded = decoder(latent_batch)
    parts = []
    for key in obs_keys:
        if key not in decoded:
            raise KeyError(f"Decoder output is missing expected key {key!r}.")
        parts.append(decoded[key].mode().reshape(latent_batch.shape[0], -1))
    return torch.cat(parts, dim=-1)


def rollout_path_for(rollout_dir: Path, seed: int) -> Path:
    return rollout_dir / f"cheetah_v3_seed{seed}_v2.npz"


def validate_true_latent(
    path: Path,
    true_latent: np.ndarray,
    *,
    expected_horizon: int,
    expected_feat_size: int,
) -> None:
    if true_latent.ndim != 2:
        raise ValueError(
            f"{display_path(path)} expected 2D true_latent, got {true_latent.ndim}D."
        )
    if true_latent.shape[0] != expected_horizon:
        raise ValueError(
            f"{display_path(path)} expected horizon {expected_horizon}, "
            f"got {true_latent.shape[0]}."
        )
    if true_latent.shape[1] != expected_feat_size:
        raise ValueError(
            f"{display_path(path)} expected latent dim {expected_feat_size}, "
            f"got {true_latent.shape[1]}."
        )


def validate_anchor_steps(anchor_steps: list[int], horizon: int) -> None:
    if not anchor_steps:
        raise ValueError("At least one anchor step is required.")
    invalid = [step for step in anchor_steps if step < 0 or step >= horizon]
    if invalid:
        raise ValueError(
            f"Anchor steps must be in [0, {horizon - 1}], got {invalid}."
        )


def band_filter_deter(
    deter: np.ndarray,
    *,
    band_names: list[str],
    filter_type: str,
) -> dict[str, np.ndarray]:
    filtered_by_band: dict[str, np.ndarray] = {}
    for band in band_names:
        low, high = DEFAULT_BANDS[band]
        filtered = np.empty_like(deter, dtype=np.float32)
        for dim in range(deter.shape[1]):
            filtered[:, dim] = bandpass(
                deter[:, dim],
                low=low,
                high=high,
                filter_type=filter_type,
            ).astype(np.float32)
        filtered_by_band[band] = filtered
    return filtered_by_band


def direction_for_anchor(
    filtered_deter: np.ndarray,
    anchor_step: int,
    *,
    min_norm: float,
) -> tuple[np.ndarray, float]:
    direction = filtered_deter[anchor_step] - filtered_deter.mean(axis=0)
    norm = float(np.linalg.norm(direction))
    if norm < min_norm:
        return np.full(direction.shape, np.nan, dtype=np.float32), norm
    return (direction / norm).astype(np.float32), norm


def probe_anchor(
    decoder: torch.nn.Module,
    obs_keys: tuple[str, ...],
    z_anchor: np.ndarray,
    directions: dict[str, np.ndarray],
    *,
    band_names: list[str],
    epsilons: np.ndarray,
    device: str,
) -> np.ndarray:
    candidates = [z_anchor.astype(np.float32, copy=True)]
    candidate_index: list[tuple[int, int]] = []

    for band_index, band in enumerate(band_names):
        direction = directions[band]
        for epsilon_index, epsilon in enumerate(epsilons):
            z_perturbed = z_anchor.astype(np.float32, copy=True)
            if np.all(np.isfinite(direction)):
                z_perturbed[DETER_SLICE] += float(epsilon) * direction
            else:
                z_perturbed[:] = np.nan
            candidates.append(z_perturbed)
            candidate_index.append((band_index, epsilon_index))

    candidate_array = np.stack(candidates, axis=0)
    valid_mask = np.isfinite(candidate_array).all(axis=1)
    response = np.full((len(band_names), len(epsilons)), np.nan, dtype=np.float32)

    if valid_mask[0]:
        valid_tensor = torch.as_tensor(
            candidate_array[valid_mask],
            device=device,
            dtype=torch.float32,
        )
        decoded_valid = decode_flat_batch(decoder, valid_tensor, obs_keys)
        decoded_lookup: dict[int, torch.Tensor] = {}
        valid_indices = np.flatnonzero(valid_mask)
        for row, original_index in enumerate(valid_indices):
            decoded_lookup[int(original_index)] = decoded_valid[row]

        anchor_decoded = decoded_lookup[0]
        for candidate_offset, (band_index, epsilon_index) in enumerate(
            candidate_index,
            start=1,
        ):
            if candidate_offset not in decoded_lookup:
                continue
            delta = torch.linalg.vector_norm(
                decoded_lookup[candidate_offset] - anchor_decoded,
                ord=2,
            )
            response[band_index, epsilon_index] = float(delta.detach().cpu())

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
    *,
    band_names: list[str],
    epsilons: np.ndarray,
) -> None:
    print()
    print("=== DECODER D2 SUMMARY: MEAN OBS L2 RESPONSE ===")
    print()
    header = f"{'Band':<12}" + "".join(f"  eps={epsilon:<8g}" for epsilon in epsilons)
    print(header)
    for band_index, band in enumerate(band_names):
        row = f"{band:<12}"
        for epsilon_index in range(len(epsilons)):
            row += f"  {obs_delta[band_index, epsilon_index]:<12.6f}"
        print(row)


def main() -> None:
    args = parse_args()
    validate_anchor_steps(args.anchor_steps, args.horizon)
    if args.num_seeds <= 0:
        raise ValueError("--num-seeds must be positive.")
    if any(epsilon <= 0 for epsilon in args.epsilons):
        raise ValueError("All epsilons must be positive.")

    device = select_device()
    checkpoint = args.checkpoint.resolve()
    rollout_dir = args.rollout_dir.resolve()
    output_path = args.output.resolve()
    seeds = np.arange(args.num_seeds, dtype=np.int64)
    epsilons = np.asarray(args.epsilons, dtype=np.float32)
    band_names = list(DEFAULT_BANDS.keys())

    print(f"Device: {device}")
    print(f"Checkpoint: {display_path(checkpoint)}")
    print(f"Rollout dir: {display_path(rollout_dir)}")
    print(f"Evaluating seeds: {seeds[0]}-{seeds[-1]}")
    print(f"Anchor steps: {args.anchor_steps}")
    print(f"Epsilons: {epsilons.tolist()}")
    print(f"Filter type: {args.filter_type}")

    adapter = DreamerV3Adapter(task=args.task, device=device)
    try:
        adapter.load_checkpoint(str(checkpoint))
        assert adapter.agent is not None
        assert adapter.config is not None

        decoder = adapter.agent._wm.heads["decoder"]
        decoder.requires_grad_(False)
        decoder.eval()

        obs_keys = tuple(getattr(decoder, "mlp_shapes", {}).keys())
        if not obs_keys:
            raise RuntimeError(
                "No vector observation decoder keys found. This experiment expects "
                "the proprioceptive DreamerV3 decoder."
            )

        feat_size = feat_size_from_config(adapter.config)
        anchor_responses: list[np.ndarray] = []
        direction_norms: list[np.ndarray] = []

        with torch.no_grad():
            for seed in seeds:
                rollout_path = rollout_path_for(rollout_dir, int(seed))
                if not rollout_path.exists():
                    raise FileNotFoundError(
                        f"Missing rollout file: {display_path(rollout_path)}"
                    )

                print(f"  seed {seed}: filtering band directions", flush=True)
                with np.load(rollout_path) as rollout:
                    true_latent = rollout["true_latent"].astype(np.float32, copy=False)

                validate_true_latent(
                    rollout_path,
                    true_latent,
                    expected_horizon=args.horizon,
                    expected_feat_size=feat_size,
                )

                deter = extract_deter(true_latent).astype(np.float32, copy=False)
                filtered_by_band = band_filter_deter(
                    deter,
                    band_names=band_names,
                    filter_type=args.filter_type,
                )

                for anchor_step in args.anchor_steps:
                    directions: dict[str, np.ndarray] = {}
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
                        decoder,
                        obs_keys,
                        true_latent[anchor_step],
                        directions,
                        band_names=band_names,
                        epsilons=epsilons,
                        device=device,
                    )
                    anchor_responses.append(response)
                    direction_norms.append(norms)

        if not anchor_responses:
            raise RuntimeError("No anchor responses were collected.")

        anchor_response_array = np.stack(anchor_responses, axis=0)
        direction_norm_array = np.stack(direction_norms, axis=0)
        obs_delta, obs_delta_std, obs_delta_counts = summarize(anchor_response_array)
        n_anchors = int(anchor_response_array.shape[0])

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path,
            obs_delta=obs_delta,
            obs_delta_std=obs_delta_std,
            band_names=np.asarray(band_names),
            epsilons=epsilons,
            n_anchors=np.asarray(n_anchors, dtype=np.int64),
            anchor_steps=np.asarray(args.anchor_steps, dtype=np.int64),
            obs_delta_counts=obs_delta_counts,
            direction_norms=direction_norm_array,
        )

        print_summary_table(obs_delta, band_names=band_names, epsilons=epsilons)
        if np.any(obs_delta_counts < n_anchors):
            print()
            print(
                "Warning: some band directions were below the minimum norm; "
                "see obs_delta_counts in the saved file."
            )
        print()
        print(f"Anchors used: {n_anchors}")
        print(f"Saved results: {display_path(output_path)}")
    finally:
        adapter._close_env()


if __name__ == "__main__":
    main()
