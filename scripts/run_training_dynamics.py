"""Run frequency-domain diagnostics across DreamerV3 training checkpoints.

For each archived checkpoint, this script extracts latent-aware Cheetah
rollouts, decomposes true/imagined latent trajectories into frequency bands,
and stores per-band MSE summaries for training-dynamics analysis.
"""

from __future__ import annotations

import argparse
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_DIR = (
    REPO_ROOT / "results" / "training_dynamics" / "cheetah" / "checkpoints_archive"
)
DEFAULT_OUTPUT = REPO_ROOT / "results" / "tables" / "v3_cheetah_training_dynamics.npz"
DEFAULT_TASK = "dmc_cheetah_run"
DEFAULT_N_SEEDS = 5
DEFAULT_HORIZON = 200
DEFAULT_TOTAL_STEPS = 1000
DEFAULT_IMAGINATION_START = 200
DEFAULT_CHECKPOINT_STEP_SIZE = 5000
DEFAULT_TRIM_START = 25
DEFAULT_TRIM_END = 175
SUMMARY_STEPS = (25, 100, 175)

sys.path.insert(0, str(REPO_ROOT))

from src.diagnostics.freq_decompose import (  # noqa: E402
    DEFAULT_BANDS,
    decompose_pair,
    extract_deter,
    fit_pca,
)
from src.models.dreamerv3_adapter import DreamerV3Adapter  # noqa: E402


@dataclass(frozen=True)
class CheckpointInfo:
    path: Path
    name: str
    number: int
    training_step: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract rollouts and compute per-band latent MSE summaries across "
            "archived DreamerV3 Cheetah training checkpoints."
        )
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory containing checkpoint_NNNNN.pt files.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output .npz path for the training dynamics dataset.",
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=DEFAULT_N_SEEDS,
        help="Number of rollout seeds to run per checkpoint.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help="Open-loop imagination horizon.",
    )
    parser.add_argument(
        "--task",
        default=DEFAULT_TASK,
        help="DreamerV3 DMC task name.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="If output exists, skip checkpoints already present in it.",
    )
    parser.add_argument(
        "--max_checkpoints",
        type=int,
        default=None,
        help="Optional limit on number of discovered checkpoints to process.",
    )
    parser.add_argument(
        "--checkpoint_step_size",
        type=int,
        default=DEFAULT_CHECKPOINT_STEP_SIZE,
        help=(
            "Training-step spacing used to map checkpoint_00001.pt -> step 5000, "
            "checkpoint_00002.pt -> step 10000, etc."
        ),
    )
    parser.add_argument(
        "--total_steps",
        type=int,
        default=DEFAULT_TOTAL_STEPS,
        help="Environment steps collected before/through the imagination window.",
    )
    parser.add_argument(
        "--imagination_start",
        type=int,
        default=DEFAULT_IMAGINATION_START,
        help="Step at which open-loop imagination begins.",
    )
    parser.add_argument(
        "--trim_start",
        type=int,
        default=DEFAULT_TRIM_START,
        help="First horizon step included in checkpoint-level mean MSE.",
    )
    parser.add_argument(
        "--trim_end",
        type=int,
        default=DEFAULT_TRIM_END,
        help="Last horizon step included in checkpoint-level mean MSE, inclusive.",
    )
    parser.add_argument(
        "--filter_type",
        choices=["butterworth", "chebyshev1", "fir"],
        default="butterworth",
        help="Temporal filter used for frequency decomposition.",
    )
    parser.add_argument(
        "--var_threshold",
        type=float,
        default=0.95,
        help="PCA variance threshold for latent/deter decomposition.",
    )
    parser.add_argument(
        "--n_pcs",
        type=int,
        default=None,
        help="Optional fixed PCA component count. Defaults to var_threshold.",
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


def checkpoint_number(path: Path) -> int:
    match = re.fullmatch(r"checkpoint_(\d+)\.pt", path.name)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)(?=\.pt$)", path.name)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not infer checkpoint number from {path.name!r}.")


def discover_checkpoints(
    checkpoint_dir: Path,
    *,
    checkpoint_step_size: int,
) -> list[CheckpointInfo]:
    checkpoint_dir = checkpoint_dir.resolve()
    paths = sorted(checkpoint_dir.glob("checkpoint_*.pt"), key=checkpoint_number)
    checkpoints = []
    for path in paths:
        number = checkpoint_number(path)
        checkpoints.append(
            CheckpointInfo(
                path=path,
                name=path.stem,
                number=number,
                training_step=number * checkpoint_step_size,
            )
        )
    return checkpoints


def validate_args(args: argparse.Namespace) -> None:
    if args.n_seeds <= 0:
        raise ValueError("--n_seeds must be positive.")
    if args.horizon <= 0:
        raise ValueError("--horizon must be positive.")
    if args.total_steps < args.imagination_start + args.horizon:
        raise ValueError("--total_steps must cover imagination_start + horizon.")
    if args.checkpoint_step_size <= 0:
        raise ValueError("--checkpoint_step_size must be positive.")
    if args.max_checkpoints is not None and args.max_checkpoints < 0:
        raise ValueError("--max_checkpoints must be non-negative.")
    if not 0 <= args.trim_start <= args.trim_end < args.horizon:
        raise ValueError(
            "--trim_start and --trim_end must define an inclusive window inside "
            f"[0, {args.horizon - 1}]."
        )
    for step in SUMMARY_STEPS:
        if step >= args.horizon:
            raise ValueError(f"SUMMARY_STEPS contains {step}, outside horizon.")


def load_existing_results(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def initial_results(
    existing: dict[str, Any] | None,
    *,
    band_count: int,
    summary_step_count: int,
) -> dict[str, list[Any]]:
    if existing is None:
        return {
            "checkpoint_names": [],
            "training_steps": [],
            "band_mse": [],
            "band_mse_steps": [],
            "n_pcs": [],
            "var_explained": [],
            "failed_checkpoint_names": [],
            "failed_errors": [],
        }

    return {
        "checkpoint_names": [str(value) for value in existing["checkpoint_names"]],
        "training_steps": [int(value) for value in existing["training_steps"]],
        "band_mse": [
            np.asarray(row, dtype=np.float32)
            for row in existing.get("band_mse", np.empty((0, band_count)))
        ],
        "band_mse_steps": [
            np.asarray(row, dtype=np.float32)
            for row in existing.get(
                "band_mse_steps",
                np.empty((0, band_count, summary_step_count)),
            )
        ],
        "n_pcs": [int(value) for value in existing.get("n_pcs", [])],
        "var_explained": [float(value) for value in existing.get("var_explained", [])],
        "failed_checkpoint_names": [
            str(value) for value in existing.get("failed_checkpoint_names", [])
        ],
        "failed_errors": [str(value) for value in existing.get("failed_errors", [])],
    }


def save_results(
    output_path: Path,
    results: dict[str, list[Any]],
    *,
    band_names: list[str],
    metadata: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if results["band_mse"]:
        band_mse = np.stack(results["band_mse"], axis=0).astype(np.float32)
    else:
        band_mse = np.empty((0, len(band_names)), dtype=np.float32)

    if results["band_mse_steps"]:
        band_mse_steps = np.stack(results["band_mse_steps"], axis=0).astype(np.float32)
    else:
        band_mse_steps = np.empty(
            (0, len(band_names), len(SUMMARY_STEPS)),
            dtype=np.float32,
        )

    np.savez(
        output_path,
        checkpoint_names=np.asarray(results["checkpoint_names"]),
        training_steps=np.asarray(results["training_steps"], dtype=np.int64),
        band_mse=band_mse,
        band_mse_steps=band_mse_steps,
        band_names=np.asarray(band_names),
        summary_steps=np.asarray(SUMMARY_STEPS, dtype=np.int64),
        n_pcs=np.asarray(results["n_pcs"], dtype=np.int64),
        var_explained=np.asarray(results["var_explained"], dtype=np.float32),
        failed_checkpoint_names=np.asarray(results["failed_checkpoint_names"]),
        failed_errors=np.asarray(results["failed_errors"]),
        metadata=np.asarray(metadata, dtype=object),
    )


def validate_rollout(rollout: dict[str, np.ndarray], *, horizon: int) -> None:
    required = ("true_latent", "imagined_latent")
    for key in required:
        if key not in rollout:
            raise KeyError(f"Rollout is missing {key!r}.")
        if rollout[key].shape != (horizon, 1536):
            raise ValueError(
                f"Expected {key} shape {(horizon, 1536)}, got {rollout[key].shape}."
            )
    if rollout["true_latent"].shape != rollout["imagined_latent"].shape:
        raise ValueError("true_latent and imagined_latent shapes do not match.")


def extract_checkpoint_rollouts(
    adapter: DreamerV3Adapter,
    *,
    n_seeds: int,
    total_steps: int,
    imagination_start: int,
    horizon: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    true_latents: list[np.ndarray] = []
    imagined_latents: list[np.ndarray] = []
    for seed in range(n_seeds):
        print(f"    extracting seed {seed}", flush=True)
        rollout = adapter.extract_rollout(
            seed=seed,
            total_steps=total_steps,
            imagination_start=imagination_start,
            horizon=horizon,
            include_latent=True,
        )
        validate_rollout(rollout, horizon=horizon)
        true_latents.append(rollout["true_latent"].astype(np.float32, copy=False))
        imagined_latents.append(
            rollout["imagined_latent"].astype(np.float32, copy=False)
        )
    return true_latents, imagined_latents


def checkpoint_band_metrics(
    true_latents: list[np.ndarray],
    imagined_latents: list[np.ndarray],
    *,
    band_names: list[str],
    trim_slice: slice,
    filter_type: str,
    var_threshold: float,
    n_pcs: int | None,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    pooled_deter = np.concatenate(
        [extract_deter(true_latent) for true_latent in true_latents],
        axis=0,
    )
    pca, _ = fit_pca(
        pooled_deter,
        n_components=n_pcs,
        var_threshold=var_threshold,
    )

    per_seed_curves = []
    for true_latent, imagined_latent in zip(true_latents, imagined_latents):
        true_dec, imagined_dec = decompose_pair(
            true_latent,
            imagined_latent,
            pca=pca,
            filter_type=filter_type,
            var_threshold=var_threshold,
            n_pcs=n_pcs,
        )
        seed_curves = []
        for band in band_names:
            diff = true_dec[band] - imagined_dec[band]
            seed_curves.append(np.mean(diff**2, axis=1))
        per_seed_curves.append(np.stack(seed_curves, axis=0))

    curves = np.stack(per_seed_curves, axis=0).astype(np.float32)
    band_mse = np.mean(curves[:, :, trim_slice], axis=(0, 2)).astype(np.float32)
    band_mse_steps = np.mean(curves[:, :, SUMMARY_STEPS], axis=0).astype(np.float32)
    var_explained = float(np.sum(pca.explained_variance_ratio_))
    return band_mse, band_mse_steps, int(pca.n_components_), var_explained


def process_checkpoint(
    checkpoint: CheckpointInfo,
    *,
    args: argparse.Namespace,
    device: str,
    band_names: list[str],
    trim_slice: slice,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    adapter = DreamerV3Adapter(task=args.task, device=device)
    try:
        adapter.load_checkpoint(str(checkpoint.path))
        true_latents, imagined_latents = extract_checkpoint_rollouts(
            adapter,
            n_seeds=args.n_seeds,
            total_steps=args.total_steps,
            imagination_start=args.imagination_start,
            horizon=args.horizon,
        )
        return checkpoint_band_metrics(
            true_latents,
            imagined_latents,
            band_names=band_names,
            trim_slice=trim_slice,
            filter_type=args.filter_type,
            var_threshold=args.var_threshold,
            n_pcs=args.n_pcs,
        )
    finally:
        adapter._close_env()


def ranking_text(values: np.ndarray, band_names: list[str]) -> str:
    order = np.argsort(values)[::-1]
    return " > ".join(band_names[index] for index in order)


def print_checkpoint_line(
    checkpoint: CheckpointInfo,
    band_mse: np.ndarray,
    *,
    band_names: list[str],
) -> None:
    values = "  ".join(
        f"{band}={value:.6f}" for band, value in zip(band_names, band_mse)
    )
    print(
        f"{checkpoint.name} step={checkpoint.training_step}: {values}",
        flush=True,
    )


def print_final_summary(results: dict[str, list[Any]], band_names: list[str]) -> None:
    if not results["band_mse"]:
        print("No checkpoints completed successfully.")
        return

    band_mse = np.stack(results["band_mse"], axis=0)
    first = band_mse[0]
    last = band_mse[-1]
    first_dominant = band_names[int(np.argmax(first))]
    last_dominant = band_names[int(np.argmax(last))]
    first_ranking = ranking_text(first, band_names)
    last_ranking = ranking_text(last, band_names)

    print()
    print("=== TRAINING DYNAMICS SUMMARY ===")
    print(f"Completed checkpoints: {len(results['band_mse'])}")
    print(f"First checkpoint dominant band: {first_dominant}")
    print(f"Last checkpoint dominant band: {last_dominant}")
    print(f"First ranking: {first_ranking}")
    print(f"Last ranking:  {last_ranking}")
    print(f"Ranking changed: {'YES' if first_ranking != last_ranking else 'NO'}")
    if results["failed_checkpoint_names"]:
        print(f"Failed checkpoints: {len(results['failed_checkpoint_names'])}")


def main() -> None:
    args = parse_args()
    validate_args(args)

    checkpoint_dir = args.checkpoint_dir.resolve()
    output_path = args.output_path.resolve()
    device = select_device()
    band_names = list(DEFAULT_BANDS.keys())
    trim_slice = slice(args.trim_start, args.trim_end + 1)

    checkpoints = discover_checkpoints(
        checkpoint_dir,
        checkpoint_step_size=args.checkpoint_step_size,
    )
    if args.max_checkpoints is not None:
        checkpoints = checkpoints[: args.max_checkpoints]
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoint_*.pt files found in {display_path(checkpoint_dir)}."
        )

    existing = load_existing_results(output_path) if args.skip_existing else None
    results = initial_results(
        existing,
        band_count=len(band_names),
        summary_step_count=len(SUMMARY_STEPS),
    )
    processed_names = set(results["checkpoint_names"])

    metadata = {
        "task": args.task,
        "n_seeds": args.n_seeds,
        "horizon": args.horizon,
        "total_steps": args.total_steps,
        "imagination_start": args.imagination_start,
        "trim_start": args.trim_start,
        "trim_end": args.trim_end,
        "checkpoint_step_size": args.checkpoint_step_size,
        "filter_type": args.filter_type,
        "var_threshold": args.var_threshold,
        "n_pcs": args.n_pcs,
        "device": device,
        "checkpoint_dir": display_path(checkpoint_dir),
    }

    print(f"Device: {device}")
    print(f"Checkpoint dir: {display_path(checkpoint_dir)}")
    print(f"Output path: {display_path(output_path)}")
    print(f"Discovered checkpoints: {len(checkpoints)}")
    print(f"Seeds per checkpoint: {args.n_seeds}")
    print(f"Trim window: [{args.trim_start}, {args.trim_end}]")
    if args.skip_existing and processed_names:
        print(f"Already processed checkpoints in output: {len(processed_names)}")

    for index, checkpoint in enumerate(checkpoints, start=1):
        if args.skip_existing and checkpoint.name in processed_names:
            print(
                f"[{index}/{len(checkpoints)}] Skipping existing {checkpoint.name}",
                flush=True,
            )
            continue

        print(
            f"[{index}/{len(checkpoints)}] Processing {checkpoint.name} "
            f"(step={checkpoint.training_step})",
            flush=True,
        )
        try:
            band_mse, band_mse_steps, n_pcs, var_explained = process_checkpoint(
                checkpoint,
                args=args,
                device=device,
                band_names=band_names,
                trim_slice=trim_slice,
            )
        except Exception as exc:  # noqa: BLE001 - keep long batch runs alive.
            error = f"{type(exc).__name__}: {exc}"
            print(f"  ERROR: {checkpoint.name} failed: {error}", flush=True)
            traceback.print_exc()
            results["failed_checkpoint_names"].append(checkpoint.name)
            results["failed_errors"].append(error)
            save_results(
                output_path,
                results,
                band_names=band_names,
                metadata=metadata,
            )
            continue

        results["checkpoint_names"].append(checkpoint.name)
        results["training_steps"].append(checkpoint.training_step)
        results["band_mse"].append(band_mse)
        results["band_mse_steps"].append(band_mse_steps)
        results["n_pcs"].append(n_pcs)
        results["var_explained"].append(var_explained)
        processed_names.add(checkpoint.name)

        print_checkpoint_line(checkpoint, band_mse, band_names=band_names)
        print(f"  PCA: n_pcs={n_pcs}, var_explained={var_explained:.4f}", flush=True)
        save_results(
            output_path,
            results,
            band_names=band_names,
            metadata=metadata,
        )

    print_final_summary(results, band_names)
    print(f"Saved results: {display_path(output_path)}")


if __name__ == "__main__":
    main()
