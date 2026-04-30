"""Run the Decoder Amplification D3 linear-vs-nonlinear separation experiment.

The probe compares the actual decoded observation difference,

    decoder(imagined_latent) - decoder(true_latent)

against the decoder applied directly to the raw latent difference,

    decoder(imagined_latent - true_latent)

following the Month 3 D3 plan. The raw latent-difference input is intentionally
out of the normal RSSM feature distribution; the resulting gap is used as a
diagnostic for nonlinear/bias-sensitive decoder behavior under this probe.
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
DEFAULT_OUTPUT = REPO_ROOT / "results" / "tables" / "decoder_d3_results.npz"
DEFAULT_TASK = "dmc_cheetah_run"
DEFAULT_NUM_SEEDS = 20
DEFAULT_HORIZON = 200
SUMMARY_STEPS = (25, 100, 175)
SUMMARY_WINDOW = slice(10, 190)

sys.path.insert(0, str(REPO_ROOT))

from src.models.dreamerv3_adapter import DreamerV3Adapter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare decoder(imagined)-decoder(true) with "
            "decoder(imagined-true) for DreamerV3 Cheetah v2 rollouts."
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
        help="Output .npz path for aggregated D3 results.",
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
        "--ratio-eps",
        type=float,
        default=1e-8,
        help="Small denominator stabilizer for nonlinearity ratio.",
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


def validate_rollout_arrays(
    path: Path,
    true_latent: np.ndarray,
    imagined_latent: np.ndarray,
    *,
    expected_horizon: int,
    expected_feat_size: int,
) -> None:
    if true_latent.shape != imagined_latent.shape:
        raise ValueError(
            f"{display_path(path)} has mismatched latent shapes: "
            f"true_latent={true_latent.shape}, imagined_latent={imagined_latent.shape}."
        )
    if true_latent.ndim != 2:
        raise ValueError(
            f"{display_path(path)} expected 2D latent arrays, got {true_latent.ndim}D."
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


def compute_seed_metrics(
    decoder: torch.nn.Module,
    obs_keys: tuple[str, ...],
    true_latent: np.ndarray,
    imagined_latent: np.ndarray,
    *,
    device: str,
    ratio_eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    true_tensor = torch.as_tensor(true_latent, device=device, dtype=torch.float32)
    imagined_tensor = torch.as_tensor(
        imagined_latent,
        device=device,
        dtype=torch.float32,
    )
    delta_tensor = imagined_tensor - true_tensor

    obs_true = decode_flat_batch(decoder, true_tensor, obs_keys)
    obs_imag = decode_flat_batch(decoder, imagined_tensor, obs_keys)
    obs_delta_decoded = decode_flat_batch(decoder, delta_tensor, obs_keys)

    actual_obs_diff = obs_imag - obs_true
    nonlinearity_gap = torch.linalg.vector_norm(
        actual_obs_diff - obs_delta_decoded,
        ord=2,
        dim=-1,
    )
    actual_obs_diff_norm = torch.linalg.vector_norm(actual_obs_diff, ord=2, dim=-1)
    ratio = nonlinearity_gap / (actual_obs_diff_norm + ratio_eps)

    return (
        nonlinearity_gap.detach().cpu().numpy().astype(np.float32),
        actual_obs_diff_norm.detach().cpu().numpy().astype(np.float32),
        ratio.detach().cpu().numpy().astype(np.float32),
    )


def print_summary_table(
    nonlinearity_gap: np.ndarray,
    actual_obs_diff_norm: np.ndarray,
    ratio: np.ndarray,
) -> None:
    gap_mean = nonlinearity_gap.mean(axis=0)
    actual_mean = actual_obs_diff_norm.mean(axis=0)
    ratio_mean = ratio.mean(axis=0)
    overall_ratio = float(np.mean(ratio[:, SUMMARY_WINDOW]))

    print()
    print("=== DECODER D3 SUMMARY: LINEAR VS NONLINEAR SEPARATION ===")
    print()
    print(
        f"{'Step':>6}  {'Mean Gap':>12}  {'Mean Actual L2':>15}  "
        f"{'Mean Ratio':>12}"
    )
    for step in SUMMARY_STEPS:
        print(
            f"{step:>6}  "
            f"{gap_mean[step]:>12.6f}  "
            f"{actual_mean[step]:>15.6f}  "
            f"{ratio_mean[step]:>12.6f}"
        )

    print()
    print(f"Overall mean ratio across steps [10, 189]: {overall_ratio:.6f}")
    if overall_ratio < 0.1:
        hint = "decoder is approximately linear under this probe"
    elif overall_ratio > 0.5:
        hint = "strong nonlinearity under this probe"
    else:
        hint = "moderate nonlinearity under this probe"
    print(f"Interpretation hint: {hint}.")


def main() -> None:
    args = parse_args()
    if args.num_seeds <= 0:
        raise ValueError("--num-seeds must be positive.")
    if args.ratio_eps <= 0:
        raise ValueError("--ratio-eps must be positive.")

    device = select_device()
    checkpoint = args.checkpoint.resolve()
    rollout_dir = args.rollout_dir.resolve()
    output_path = args.output.resolve()
    seeds = np.arange(args.num_seeds, dtype=np.int64)

    print(f"Device: {device}")
    print(f"Checkpoint: {display_path(checkpoint)}")
    print(f"Rollout dir: {display_path(rollout_dir)}")
    print(f"Evaluating seeds: {seeds[0]}-{seeds[-1]}")
    print("Note: decoder(latent_delta) intentionally feeds an OOD latent difference.")

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
        nonlinearity_gap = np.empty((len(seeds), args.horizon), dtype=np.float32)
        actual_obs_diff_norm = np.empty((len(seeds), args.horizon), dtype=np.float32)
        ratio = np.empty((len(seeds), args.horizon), dtype=np.float32)

        with torch.no_grad():
            for row, seed in enumerate(seeds):
                rollout_path = rollout_path_for(rollout_dir, int(seed))
                if not rollout_path.exists():
                    raise FileNotFoundError(
                        f"Missing rollout file: {display_path(rollout_path)}"
                    )

                print(f"  seed {seed}: {display_path(rollout_path)}", flush=True)
                with np.load(rollout_path) as rollout:
                    true_latent = rollout["true_latent"].astype(np.float32, copy=False)
                    imagined_latent = rollout["imagined_latent"].astype(
                        np.float32,
                        copy=False,
                    )

                validate_rollout_arrays(
                    rollout_path,
                    true_latent,
                    imagined_latent,
                    expected_horizon=args.horizon,
                    expected_feat_size=feat_size,
                )

                seed_gap, seed_actual, seed_ratio = compute_seed_metrics(
                    decoder,
                    obs_keys,
                    true_latent,
                    imagined_latent,
                    device=device,
                    ratio_eps=args.ratio_eps,
                )
                nonlinearity_gap[row] = seed_gap
                actual_obs_diff_norm[row] = seed_actual
                ratio[row] = seed_ratio

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path,
            nonlinearity_gap=nonlinearity_gap,
            actual_obs_diff_norm=actual_obs_diff_norm,
            ratio=ratio,
            seeds=seeds,
        )

        print_summary_table(nonlinearity_gap, actual_obs_diff_norm, ratio)
        print()
        print(f"Saved results: {display_path(output_path)}")
    finally:
        adapter._close_env()


if __name__ == "__main__":
    main()
