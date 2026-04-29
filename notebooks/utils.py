"""Shared utilities for DriftDetect notebooks."""

from pathlib import Path


def find_repo_root() -> Path:
    """Find DriftDetect repo root from any subdirectory.

    Walks up from current working directory until it finds the
    project marker (src/models/adapter.py).
    """
    current = Path.cwd().resolve()
    while current != current.parent:
        if (current / "src" / "models" / "adapter.py").exists():
            return current
        current = current.parent
    raise RuntimeError(
        "Cannot find DriftDetect repo root. "
        "Make sure you're inside the project directory."
    )


REPO_ROOT = find_repo_root()
ROLLOUT_DIR = REPO_ROOT / "results" / "rollouts"
CHECKPOINT_DIR = REPO_ROOT / "results" / "checkpoints"
