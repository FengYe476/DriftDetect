"""Batch rollout extraction for DreamerV3 on Cheetah Run.

Extracts 20 rollouts (seeds 0-19) and saves to results/rollouts/.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = REPO_ROOT / "results" / "checkpoints" / "cheetah_run_500k.pt"
OUTPUT_DIR = REPO_ROOT / "results" / "rollouts"
MANIFEST = OUTPUT_DIR / "manifest.json"

sys.path.insert(0, str(REPO_ROOT))

from src.models.dreamerv3_adapter import DreamerV3Adapter  # noqa: E402

TASK = "dmc_cheetah_run"
SEEDS = range(20)
TOTAL_STEPS = 1000
IMAGINATION_START = 200
HORIZON = 200


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def save_rollout(path: Path, seed: int, rollout: dict[str, np.ndarray]) -> None:
    metadata = {
        "task": TASK,
        "seed": seed,
        "total_steps": TOTAL_STEPS,
        "imagination_start": IMAGINATION_START,
        "horizon": HORIZON,
        "agent_type": "dreamerv3_trained_checkpoint",
        "checkpoint_path": display_path(CHECKPOINT),
        "obs_alignment": (
            "true and imagined arrays contain next observations for actions "
            f"at steps {IMAGINATION_START}-{IMAGINATION_START + HORIZON - 1}"
        ),
    }
    np.savez(
        path,
        true_obs=rollout["true_obs"].astype(np.float32),
        imagined_obs=rollout["imagined_obs"].astype(np.float32),
        actions=rollout["actions"].astype(np.float32),
        rewards=rollout["rewards"].astype(np.float32),
        metadata=np.array(metadata, dtype=object),
    )


def rollout_summary(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {
        "file": display_path(path),
        "sha256": sha256_file(path),
        "shapes": {
            key: list(data[key].shape)
            for key in ("true_obs", "imagined_obs", "actions", "rewards")
        },
    }


def update_manifest() -> dict:
    files = [OUTPUT_DIR / f"cheetah_v3_seed{seed}.npz" for seed in SEEDS]
    existing = [path for path in files if path.exists()]
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task": TASK,
        "checkpoint": display_path(CHECKPOINT),
        "total_steps": TOTAL_STEPS,
        "imagination_start": IMAGINATION_START,
        "horizon": HORIZON,
        "files": [rollout_summary(path) for path in existing],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    adapter = DreamerV3Adapter(task=TASK)
    adapter.load_checkpoint(str(CHECKPOINT))

    for seed in SEEDS:
        output = OUTPUT_DIR / f"cheetah_v3_seed{seed}.npz"
        label = f"{seed + 1}/20"
        if output.exists():
            print(
                f"Skipping seed {label}: {display_path(output)} already exists.",
                flush=True,
            )
            continue
        print(f"Extracting seed {label}...", flush=True)
        rollout = adapter.extract_rollout(
            seed=seed,
            total_steps=TOTAL_STEPS,
            imagination_start=IMAGINATION_START,
            horizon=HORIZON,
        )
        save_rollout(output, seed, rollout)
        print(f"Saved {display_path(output)}.", flush=True)

    manifest = update_manifest()
    print(
        f"Updated {display_path(MANIFEST)} with {len(manifest['files'])} files.",
        flush=True,
    )


if __name__ == "__main__":
    main()
