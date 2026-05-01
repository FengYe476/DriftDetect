"""Batch rollout extraction for DreamerV4 latent-aware rollouts on RunPod."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "rollouts_v4"
DEFAULT_SEEDS = ",".join(str(seed) for seed in range(20))

sys.path.insert(0, str(REPO_ROOT))


def load_adapter_class():
    path = REPO_ROOT / "src" / "models" / "dreamerv4_adapter.py"
    spec = importlib.util.spec_from_file_location("_driftdetect_dreamerv4_adapter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load DreamerV4Adapter from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.DreamerV4Adapter


def parse_seeds(value: str) -> list[int]:
    seeds = [item.strip() for item in value.split(",") if item.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required.")
    return [int(seed) for seed in seeds]


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rollout_summary(path: Path, seed: int, elapsed_s: float) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    keys = (
        "true_obs",
        "imagined_obs",
        "true_latent",
        "imagined_latent",
        "actions",
        "action_mask",
        "rewards",
    )
    return {
        "file": display_path(path),
        "seed": int(seed),
        "sha256": sha256_file(path),
        "elapsed_s": float(elapsed_s),
        "shapes": {key: list(data[key].shape) for key in keys if key in data},
    }


def write_manifest(
    path: Path,
    *,
    task: str,
    seeds: list[int],
    context_length: int,
    horizon: int,
    adapter: DreamerV4Adapter,
    schedule: str,
    schedule_d: float,
    files: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> None:
    manifest = {
        "version": "v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "seeds": seeds,
        "context_length": int(context_length),
        "horizon": int(horizon),
        "checkpoint_revision": "local",
        "checkpoint": adapter.checkpoint_info,
        "schedule": schedule,
        "schedule_d": float(schedule_d),
        "files": files,
        "errors": errors,
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint_dir",
        required=True,
        help="Directory containing tokenizer.pt and dynamics.pt.",
    )
    parser.add_argument(
        "--dataset_dir",
        required=True,
        help="Directory containing V4 raw demos and preprocessed frame shards.",
    )
    parser.add_argument("--task", default="cheetah-run", help="V4 task name.")
    parser.add_argument(
        "--seeds",
        default=DEFAULT_SEEDS,
        help=f"Comma-separated seed list. Default: {DEFAULT_SEEDS}",
    )
    parser.add_argument("--context_length", type=int, default=8, help="Context frame count.")
    parser.add_argument("--horizon", type=int, default=200, help="Imagination step count.")
    parser.add_argument(
        "--output_dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for V4 rollouts.",
    )
    parser.add_argument(
        "--schedule",
        default="shortcut",
        choices=("shortcut", "finest"),
        help="Denoising schedule.",
    )
    parser.add_argument(
        "--schedule_d",
        type=float,
        default=0.25,
        help="Shortcut schedule step size.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    seeds = parse_seeds(args.seeds)
    checkpoint_dir = resolve_path(args.checkpoint_dir)
    dataset_dir = resolve_path(args.dataset_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest_v4.json"

    print(f"Loading DreamerV4 adapter from {display_path(checkpoint_dir)}", flush=True)
    DreamerV4Adapter = load_adapter_class()
    adapter = DreamerV4Adapter(
        task=args.task,
        schedule=args.schedule,
        schedule_d=args.schedule_d,
    )
    adapter.load_checkpoint(str(checkpoint_dir))

    files: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    total = len(seeds)
    for index, seed in enumerate(seeds, start=1):
        output = output_dir / f"{args.task}_v4_seed{seed}_v1.npz"
        label = f"{index}/{total}"
        print(f"[{label}] Extracting seed={seed} -> {display_path(output)}", flush=True)
        start = time.perf_counter()
        try:
            adapter.extract_rollout(
                dataset_path=dataset_dir,
                task_name=args.task,
                seed=seed,
                context_length=args.context_length,
                horizon=args.horizon,
                include_latent=True,
                output_path=output,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - start
            errors.append(
                {
                    "seed": int(seed),
                    "elapsed_s": float(elapsed),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"[{label}] ERROR seed={seed}: {exc!r}", flush=True)
            continue

        elapsed = time.perf_counter() - start
        files.append(rollout_summary(output, seed, elapsed))
        print(f"[{label}] Saved in {elapsed:.2f}s", flush=True)

    write_manifest(
        manifest_path,
        task=args.task,
        seeds=seeds,
        context_length=args.context_length,
        horizon=args.horizon,
        adapter=adapter,
        schedule=args.schedule,
        schedule_d=args.schedule_d,
        files=files,
        errors=errors,
    )
    print(
        f"Wrote {display_path(manifest_path)} with {len(files)} files "
        f"and {len(errors)} errors.",
        flush=True,
    )


if __name__ == "__main__":
    main()
