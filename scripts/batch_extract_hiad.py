#!/usr/bin/env python3
"""Batch-extract HIAD-damped Cheetah rollouts for 20 seeds.

Usage:
    python scripts/batch_extract_hiad.py \
        --checkpoint results/checkpoints/cheetah_run_500k.pt \
        --output_dir results/rollouts_hiad/ \
        --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
        --hiad_eta 0.20 \
        --hiad_r 10 \
        --hiad_re_est_freq 25 \
        --hiad_basis_mode initial_basis \
        --hiad_initial_basis results/smad/U_drift_cheetah_r10.npy

Self-bootstrap mode:
    python scripts/batch_extract_hiad.py \
        --checkpoint results/checkpoints/cheetah_run_500k.pt \
        --output_dir results/rollouts_hiad_bootstrap/ \
        --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
        --hiad_eta 0.20 \
        --hiad_r 10 \
        --hiad_re_est_freq 25 \
        --hiad_basis_mode self_bootstrap
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXTRACT_SCRIPT = REPO_ROOT / "src" / "diagnostics" / "extract_rollout.py"
DEFAULT_CHECKPOINT = REPO_ROOT / "results" / "checkpoints" / "cheetah_run_500k.pt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "rollouts_hiad"
DEFAULT_INITIAL_BASIS = REPO_ROOT / "results" / "smad" / "U_drift_cheetah_r10.npy"
DEFAULT_SEEDS = list(range(20))
TASK = "dmc_cheetah_run"
TOTAL_STEPS = 1000
IMAGINATION_START = 200
HORIZON = 200


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()

    for index, seed in enumerate(args.seeds, start=1):
        output = args.output_dir / f"cheetah_hiad_seed{seed}.npz"
        label = f"{index}/{len(args.seeds)}"
        if output.exists() and not args.overwrite:
            print(
                f"[seed {label}] skipping existing {display_path(output)}",
                flush=True,
            )
            continue

        print(f"[seed {label}] extracting seed={seed}...", flush=True)
        subprocess.run(build_command(args, seed, output), cwd=REPO_ROOT, check=True)
        print(f"[seed {label}] saved {display_path(output)}", flush=True)

    elapsed = time.perf_counter() - start_time
    print(f"HIAD batch extraction complete in {elapsed / 60.0:.1f} minutes.", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-extract HIAD-damped Cheetah rollouts."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--hiad_eta", type=float, default=0.20)
    parser.add_argument("--hiad_r", type=int, default=10)
    parser.add_argument("--hiad_re_est_freq", type=int, default=25)
    parser.add_argument("--hiad_window_size", type=int, default=25)
    parser.add_argument(
        "--hiad_basis_mode",
        choices=["initial_basis", "self_bootstrap"],
        default="initial_basis",
    )
    parser.add_argument("--hiad_initial_basis", type=Path, default=DEFAULT_INITIAL_BASIS)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    args.checkpoint = resolve_path(args.checkpoint)
    args.output_dir = resolve_path(args.output_dir)
    if args.hiad_initial_basis is not None:
        args.hiad_initial_basis = resolve_path(args.hiad_initial_basis)
    if args.hiad_r <= 0:
        raise ValueError("--hiad_r must be positive.")
    if args.hiad_re_est_freq <= 0:
        raise ValueError("--hiad_re_est_freq must be positive.")
    if args.hiad_window_size <= 0:
        raise ValueError("--hiad_window_size must be positive.")
    if args.hiad_eta < 0:
        raise ValueError("--hiad_eta must be non-negative.")
    if not args.seeds:
        raise ValueError("--seeds must contain at least one seed.")
    return args


def build_command(args: argparse.Namespace, seed: int, output: Path) -> list[str]:
    command = [
        sys.executable,
        str(EXTRACT_SCRIPT),
        "--task",
        TASK,
        "--seed",
        str(seed),
        "--imagination_start",
        str(IMAGINATION_START),
        "--horizon",
        str(HORIZON),
        "--total_steps",
        str(TOTAL_STEPS),
        "--checkpoint",
        str(args.checkpoint),
        "--output",
        str(output),
        "--use_hiad",
        "--hiad_eta",
        str(args.hiad_eta),
        "--hiad_r",
        str(args.hiad_r),
        "--hiad_re_est_freq",
        str(args.hiad_re_est_freq),
        "--hiad_window_size",
        str(args.hiad_window_size),
        "--hiad_basis_mode",
        args.hiad_basis_mode,
    ]
    if args.hiad_basis_mode == "initial_basis" and args.hiad_initial_basis is not None:
        command.extend(["--hiad_initial_basis", str(args.hiad_initial_basis)])
    if args.device is not None:
        command.extend(["--device", args.device])
    if args.verbose:
        command.append("--verbose")
    return command


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
