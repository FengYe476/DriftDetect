from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.smad.U_estimation import DETER_DIM, DETER_START  # noqa: E402
from src.smad.adaptive_smad.re_estimation import AdaptiveReEstimator  # noqa: E402
from src.smad.adaptive_smad.scheduler import (  # noqa: E402
    AdaptiveSMADScheduler,
    subspace_overlap,
)


RANK = 10
LATENT_DIM = DETER_START + DETER_DIM


def basis(start: int = 0, rank: int = RANK) -> np.ndarray:
    U = np.zeros((DETER_DIM, rank), dtype=np.float64)
    for col in range(rank):
        U[start + col, col] = 1.0
    return U


class RaisingReEstimator:
    def estimate(self) -> np.ndarray:
        raise AssertionError("estimate() should not have been called")


class SequenceReEstimator:
    def __init__(self, values: list[np.ndarray]) -> None:
        self.values = [value.copy() for value in values]
        self.calls = 0

    def estimate(self) -> np.ndarray:
        value = self.values[self.calls]
        self.calls += 1
        return value.copy()


class RecordingPatch:
    def __init__(self) -> None:
        self.updated_U: list[np.ndarray] = []

    def update_U(self, U: np.ndarray) -> None:
        self.updated_U.append(np.asarray(U).copy())


def test_noop_identity_mode_never_reestimates_or_modifies_U() -> None:
    initial_U = basis()
    patch = RecordingPatch()
    scheduler = AdaptiveSMADScheduler(
        RaisingReEstimator(),
        patch,
        re_est_freq=999999,
        activation_s0=50000,
        activation_s1=100000,
        initial_U=initial_U,
    )

    for step in (0, 49999, 50000, 100000):
        assert scheduler.maybe_update(step) is None

    assert patch.updated_U == []
    np.testing.assert_array_equal(scheduler.current_U, initial_U)
    np.testing.assert_array_equal(scheduler.U_history[0]["U"], initial_U)


def test_trigger_cadence_and_overlap_formula() -> None:
    initial_U = basis()
    orthogonal_U = basis(start=RANK)
    patch = RecordingPatch()
    re_estimator = SequenceReEstimator([initial_U, orthogonal_U, initial_U])
    scheduler = AdaptiveSMADScheduler(
        re_estimator,
        patch,
        re_est_freq=50000,
        activation_s0=50000,
        activation_s1=100000,
        initial_U=initial_U,
    )

    assert scheduler.maybe_update(0) is None
    assert scheduler.maybe_update(49999) is None
    first = scheduler.maybe_update(50000)
    assert scheduler.maybe_update(50001) is None
    second = scheduler.maybe_update(100000)
    third = scheduler.maybe_update(150000)

    assert first is not None
    assert second is not None
    assert third is not None
    assert first["step"] == 50000
    assert first["overlap_prev"] == pytest.approx(1.0)
    assert first["overlap_baseline"] == pytest.approx(1.0)
    assert second["overlap_prev"] == pytest.approx(0.0)
    assert second["overlap_baseline"] == pytest.approx(0.0)
    assert third["overlap_prev"] == pytest.approx(0.0)
    assert third["overlap_baseline"] == pytest.approx(1.0)
    assert re_estimator.calls == 3
    assert len(patch.updated_U) == 3
    assert len(scheduler.U_history) == 4

    assert subspace_overlap(initial_U, initial_U) == pytest.approx(1.0)
    assert subspace_overlap(initial_U, orthogonal_U) == pytest.approx(0.0)


def test_state_restore_and_history_copy_semantics() -> None:
    initial_U = basis()
    updated_U = basis(start=RANK)
    scheduler = AdaptiveSMADScheduler(
        SequenceReEstimator([updated_U]),
        RecordingPatch(),
        re_est_freq=50000,
        activation_s0=50000,
        activation_s1=100000,
        initial_U=initial_U,
    )
    assert scheduler.maybe_update(50000) is not None
    history_before_mutation = scheduler.U_history[-1]["U"].copy()

    scheduler.current_U[:] = 0.0
    np.testing.assert_array_equal(scheduler.U_history[-1]["U"], history_before_mutation)

    state = scheduler.state_dict()
    restored_patch = RecordingPatch()
    restored = AdaptiveSMADScheduler(
        RaisingReEstimator(),
        restored_patch,
        re_est_freq=50000,
        activation_s0=50000,
        activation_s1=100000,
        initial_U=initial_U,
    )
    restored.load_state_dict(state)

    np.testing.assert_array_equal(restored.current_U, state["current_U"])
    np.testing.assert_array_equal(restored.U_baseline, state["U_baseline"])
    assert len(restored.U_history) == len(state["U_history"])
    np.testing.assert_array_equal(restored_patch.updated_U[-1], state["current_U"])

    state["U_history"][-1]["U"][:] = 1.0
    np.testing.assert_array_equal(restored.U_history[-1]["U"], history_before_mutation)


class FakeRolloutModel:
    def __init__(self, rank: int, horizon: int) -> None:
        self.rank = rank
        self.horizon = horizon
        self.training = True
        self.calls: list[dict[str, object]] = []

    def eval(self) -> None:
        self.training = False

    def train(self, mode: bool = True) -> None:
        self.training = bool(mode)

    def extract_rollout(
        self,
        *,
        seed: int,
        total_steps: int,
        imagination_start: int,
        horizon: int,
        include_latent: bool,
    ) -> dict[str, np.ndarray]:
        self.calls.append(
            {
                "seed": seed,
                "total_steps": total_steps,
                "imagination_start": imagination_start,
                "horizon": horizon,
                "include_latent": include_latent,
                "grad_enabled": torch.is_grad_enabled(),
                "training": self.training,
            }
        )
        rng = np.random.default_rng(seed)
        true_latent = np.zeros((horizon, LATENT_DIM), dtype=np.float32)
        imagined_latent = np.zeros((horizon, LATENT_DIM), dtype=np.float32)
        coefficients = rng.normal(size=(horizon, self.rank)).astype(np.float32)
        drift = coefficients @ basis(rank=self.rank).T.astype(np.float32)
        imagined_latent[:, DETER_START:] = drift
        return {
            "true_latent": true_latent,
            "imagined_latent": imagined_latent,
        }


def test_re_estimator_uses_mockable_rollout_protocol_and_restores_train_mode() -> None:
    rank = 3
    horizon = 8
    model = FakeRolloutModel(rank=rank, horizon=horizon)
    re_estimator = AdaptiveReEstimator(
        model,
        env_name="dmc_cheetah_run",
        rank=rank,
        n_rollouts=2,
        horizon=horizon,
    )

    U = re_estimator.estimate()

    assert U.shape == (DETER_DIM, rank)
    np.testing.assert_allclose(U.T @ U, np.eye(rank), atol=1e-10)
    assert model.training is True
    assert len(model.calls) == 2
    for seed, call in enumerate(model.calls):
        assert call["seed"] == seed
        assert call["total_steps"] == 50 + horizon
        assert call["imagination_start"] == 50
        assert call["horizon"] == horizon
        assert call["include_latent"] is True
        assert call["grad_enabled"] is False
        assert call["training"] is False
