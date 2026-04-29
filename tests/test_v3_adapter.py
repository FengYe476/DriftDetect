"""Integration tests for the DreamerV3 world model adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.models.dreamerv3_adapter import DreamerV3Adapter


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = REPO_ROOT / "results" / "checkpoints" / "cheetah_run_500k.pt"
REFERENCE = REPO_ROOT / "results" / "rollouts" / "cheetah_v3_seed0.npz"


@pytest.fixture(scope="session")
def adapter() -> DreamerV3Adapter:
    model = DreamerV3Adapter(task="dmc_cheetah_run")
    model.load_checkpoint(str(CHECKPOINT))
    return model


@pytest.fixture(scope="session")
def adapter_rollout(adapter: DreamerV3Adapter) -> dict:
    return adapter.extract_rollout(
        seed=0,
        total_steps=1000,
        imagination_start=200,
        horizon=200,
    )


def test_adapter_loads_checkpoint(adapter: DreamerV3Adapter) -> None:
    """Adapter can load the real checkpoint without errors."""
    assert adapter.agent is not None
    assert adapter.checkpoint_path == CHECKPOINT


def test_adapter_output_shapes(adapter_rollout: dict) -> None:
    """extract_rollout() returns correct shapes."""
    assert adapter_rollout["true_obs"].shape == (200, 17)
    assert adapter_rollout["imagined_obs"].shape == (200, 17)
    assert adapter_rollout["actions"].shape == (200, 6)
    assert adapter_rollout["rewards"].shape == (200,)


def test_adapter_numerical_consistency(adapter_rollout: dict) -> None:
    """
    Adapter output is numerically close to extract_rollout.py output.
    Use tighter thresholds for deterministic env/policy outputs and a looser
    threshold for long-horizon imagined observations.
    """
    reference = np.load(REFERENCE, allow_pickle=True)
    thresholds = {
        "true_obs": 1e-5,
        "imagined_obs": 1e-3,
        "actions": 1e-5,
        "rewards": 1e-5,
    }
    for key in ("true_obs", "imagined_obs", "actions", "rewards"):
        diff = np.linalg.norm(adapter_rollout[key] - reference[key])
        threshold = thresholds[key]
        print(f"{key:<12} L2 diff = {diff:.2e} (threshold {threshold:.0e})")
        assert diff < threshold, f"{key} L2 diff too large: {diff}"


def test_adapter_drift_grows(adapter_rollout: dict) -> None:
    """L2 distance between true and imagined obs grows over horizon."""
    distances = np.linalg.norm(
        adapter_rollout["true_obs"] - adapter_rollout["imagined_obs"],
        axis=1,
    )
    assert distances[100:].mean() > distances[:100].mean()
