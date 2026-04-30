"""Run the Decoder Amplification D1 random-decoder control experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = REPO_ROOT / "results" / "checkpoints" / "cheetah_run_500k.pt"
DEFAULT_ROLLOUT_DIR = REPO_ROOT / "results" / "rollouts"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "tables" / "decoder_d1_results.npz"
DEFAULT_TASK = "dmc_cheetah_run"
DEFAULT_NUM_SEEDS = 20
DEFAULT_HORIZON = 200
SUMMARY_STEPS = (25, 100, 175)

sys.path.insert(0, str(REPO_ROOT))

from src.models.dreamerv3_adapter import DreamerV3Adapter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decode imagined and true DreamerV3 RSSM features with both the "
            "trained decoder and a fresh random decoder, then compare drift."
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
        help="Output .npz path for aggregated decoder drift curves.",
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
        "--random-decoder-seed",
        type=int,
        default=0,
        help="Torch seed used for the fresh random decoder initialization.",
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


def build_random_decoder(
    adapter: DreamerV3Adapter,
    *,
    seed: int,
) -> torch.nn.Module:
    assert adapter.agent is not None
    assert adapter.env is not None
    assert adapter.config is not None

    trained_decoder = adapter.agent._wm.heads["decoder"]
    obs_space = adapter.env.observation_space
    shapes = {key: tuple(space.shape) for key, space in obs_space.spaces.items()}
    feat_size = feat_size_from_config(adapter.config)

    torch.manual_seed(seed)
    decoder = trained_decoder.__class__(
        feat_size,
        shapes,
        **adapter.config.decoder,
    ).to(adapter.config.device)
    decoder.requires_grad_(False)
    decoder.eval()
    return decoder


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


def l2_curve(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    return torch.linalg.vector_norm(a - b, ord=2, dim=-1).detach().cpu().numpy().astype(
        np.float32
    )


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


def safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return float("inf")
    return numerator / denominator


def print_summary_table(
    trained_decoder_drift: np.ndarray,
    random_decoder_drift: np.ndarray,
) -> None:
    trained_mean = trained_decoder_drift.mean(axis=0)
    random_mean = random_decoder_drift.mean(axis=0)

    print()
    print("=== DECODER D1 SUMMARY: MEAN L2 DRIFT ACROSS 20 CHEETAH V2 ROLLOUTS ===")
    print()
    print(
        f"{'Step':>6}  {'Trained Mean L2':>17}  {'Random Mean L2':>16}  "
        f"{'Ratio (trained/random)':>23}"
    )
    for step in SUMMARY_STEPS:
        ratio = safe_ratio(float(trained_mean[step]), float(random_mean[step]))
        print(
            f"{step:>6}  "
            f"{trained_mean[step]:>17.6f}  "
            f"{random_mean[step]:>16.6f}  "
            f"{ratio:>23.6f}"
        )


def main() -> None:
    args = parse_args()
    device = select_device()
    checkpoint = args.checkpoint.resolve()
    rollout_dir = args.rollout_dir.resolve()
    output_path = args.output.resolve()
    seeds = np.arange(args.num_seeds, dtype=np.int64)

    print(f"Device: {device}")
    print(f"Checkpoint: {display_path(checkpoint)}")
    print(f"Rollout dir: {display_path(rollout_dir)}")
    print(f"Evaluating seeds: {seeds[0]}-{seeds[-1]}")

    adapter = DreamerV3Adapter(task=args.task, device=device)
    try:
        adapter.load_checkpoint(str(checkpoint))
        assert adapter.agent is not None
        assert adapter.config is not None

        trained_decoder = adapter.agent._wm.heads["decoder"]
        trained_decoder.requires_grad_(False)
        trained_decoder.eval()

        obs_keys = tuple(getattr(trained_decoder, "mlp_shapes", {}).keys())
        if not obs_keys:
            raise RuntimeError(
                "No vector observation decoder keys found. This experiment expects "
                "the proprioceptive DreamerV3 decoder."
            )

        random_decoder = build_random_decoder(
            adapter,
            seed=args.random_decoder_seed,
        )
        feat_size = feat_size_from_config(adapter.config)

        trained_decoder_drift = np.empty(
            (len(seeds), args.horizon), dtype=np.float32
        )
        random_decoder_drift = np.empty((len(seeds), args.horizon), dtype=np.float32)

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
                        np.float32, copy=False
                    )

                validate_rollout_arrays(
                    rollout_path,
                    true_latent,
                    imagined_latent,
                    expected_horizon=args.horizon,
                    expected_feat_size=feat_size,
                )

                true_tensor = torch.as_tensor(
                    true_latent,
                    device=device,
                    dtype=torch.float32,
                )
                imagined_tensor = torch.as_tensor(
                    imagined_latent,
                    device=device,
                    dtype=torch.float32,
                )

                trained_decoded_true = decode_flat_batch(
                    trained_decoder, true_tensor, obs_keys
                )
                trained_decoded_imag = decode_flat_batch(
                    trained_decoder, imagined_tensor, obs_keys
                )
                random_decoded_true = decode_flat_batch(
                    random_decoder, true_tensor, obs_keys
                )
                random_decoded_imag = decode_flat_batch(
                    random_decoder, imagined_tensor, obs_keys
                )

                trained_decoder_drift[row] = l2_curve(
                    trained_decoded_imag,
                    trained_decoded_true,
                )
                random_decoder_drift[row] = l2_curve(
                    random_decoded_imag,
                    random_decoded_true,
                )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path,
            trained_decoder_drift=trained_decoder_drift,
            random_decoder_drift=random_decoder_drift,
            seeds=seeds,
        )

        print_summary_table(trained_decoder_drift, random_decoder_drift)
        print()
        print(f"Saved results: {display_path(output_path)}")
    finally:
        adapter._close_env()


if __name__ == "__main__":
    main()
