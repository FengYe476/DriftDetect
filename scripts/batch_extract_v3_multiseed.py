#!/usr/bin/env python3
"""Parameterized DreamerV3 latent-aware rollout extraction.

This is the multi-seed wrapper-friendly version of
scripts/batch_extract_moment_match.py. It keeps the Month 8 extraction format:
DreamerV3Adapter, include_latent=True, and *_v3_seed*_v2.npz outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.dreamerv3_adapter import DreamerV3Adapter  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def latent_spec(adapter: DreamerV3Adapter) -> dict[str, int | bool]:
    assert adapter.agent is not None
    rssm = adapter.agent._wm.dynamics
    if rssm._discrete:
        stoch_dim = int(rssm._stoch * rssm._discrete)
    else:
        stoch_dim = int(rssm._stoch)
    return {
        "latent_dim": int(rssm._deter + stoch_dim),
        "deter_dim": int(rssm._deter),
        "stoch_discrete": bool(rssm._discrete),
        "num_categoricals": int(rssm._stoch),
        "num_classes": int(rssm._discrete) if rssm._discrete else 0,
    }


def task_short(task: str) -> str:
    if task == "dmc_cheetah_run":
        return "cheetah"
    if task == "dmc_cartpole_swingup":
        return "cartpole"
    cleaned = task.removeprefix("dmc_")
    return cleaned.split("_", 1)[0]


def save_rollout(
    path: Path,
    *,
    rollout_seed: int,
    rollout: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> None:
    metadata = {
        "version": "v2",
        "task": args.task,
        "seed": int(rollout_seed),
        "total_steps": int(args.total_steps),
        "imagination_start": int(args.imagination_start),
        "horizon": int(args.horizon),
        "agent_type": "dreamerv3_trained_checkpoint",
        "checkpoint_path": display_path(args.checkpoint),
        "obs_alignment": (
            "true and imagined arrays contain next observations for actions "
            f"at steps {args.imagination_start}-{args.imagination_start + args.horizon - 1}"
        ),
        "latent_alignment": (
            "true_latent and imagined_latent contain deterministic RSSM features "
            f"for steps {args.imagination_start + 1}-{args.imagination_start + args.horizon}"
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


def rollout_summary(path: Path) -> dict[str, object]:
    data = np.load(path, allow_pickle=True)
    return {
        "file": display_path(path),
        "seed": int(data["metadata"].item()["seed"]),
        "sha256": sha256_file(path),
        "shapes": {
            key: list(data[key].shape)
            for key in (
                "true_obs",
                "imagined_obs",
                "actions",
                "rewards",
                "true_latent",
                "imagined_latent",
            )
        },
    }


def update_manifest(adapter: DreamerV3Adapter, args: argparse.Namespace) -> dict[str, object]:
    files = [
        args.output_dir / f"{args.task_short}_v3_seed{seed}_v2.npz"
        for seed in args.seeds
    ]
    existing = [path for path in files if path.exists()]
    manifest = {
        "version": "v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "checkpoint": display_path(args.checkpoint),
        "total_steps": int(args.total_steps),
        "imagination_start": int(args.imagination_start),
        "horizon": int(args.horizon),
        "latent_spec": latent_spec(adapter),
        "files": [rollout_summary(path) for path in existing],
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract DreamerV3 latent-aware rollouts.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--task", default="dmc_cheetah_run")
    parser.add_argument("--num_seeds", type=int, default=20)
    parser.add_argument("--total_steps", type=int, default=1000)
    parser.add_argument("--imagination_start", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--device", default=None)
    parser.add_argument("--manifest_name", default="manifest_v3_multiseed.json")
    args = parser.parse_args()

    args.checkpoint = resolve_path(args.checkpoint)
    args.output_dir = resolve_path(args.output_dir)
    args.manifest = args.output_dir / args.manifest_name
    args.seeds = range(int(args.num_seeds))
    args.task_short = task_short(args.task)
    if args.num_seeds <= 0:
        raise ValueError("--num_seeds must be positive.")
    if args.total_steps < args.imagination_start + args.horizon:
        raise ValueError("--total_steps must cover imagination_start + horizon.")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {display_path(args.checkpoint)}")
    return args


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    adapter = DreamerV3Adapter(task=args.task, device=args.device)
    adapter.load_checkpoint(str(args.checkpoint))

    for rollout_seed in args.seeds:
        output = args.output_dir / f"{args.task_short}_v3_seed{rollout_seed}_v2.npz"
        label = f"{rollout_seed + 1}/{args.num_seeds}"
        if output.exists():
            print(
                f"Skipping seed {label}: {display_path(output)} already exists.",
                flush=True,
            )
            continue
        print(f"Extracting seed {label} from {display_path(args.checkpoint)}...", flush=True)
        rollout = adapter.extract_rollout(
            seed=rollout_seed,
            total_steps=args.total_steps,
            imagination_start=args.imagination_start,
            horizon=args.horizon,
            include_latent=True,
        )
        save_rollout(output, rollout_seed=rollout_seed, rollout=rollout, args=args)
        print(f"Saved {display_path(output)}.", flush=True)

    manifest = update_manifest(adapter, args)
    print(
        f"Updated {display_path(args.manifest)} with {len(manifest['files'])} files.",
        flush=True,
    )


if __name__ == "__main__":
    main()
