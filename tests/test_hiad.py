from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.smad.hiad import (  # noqa: E402
    HIADDampingStep,
    HIADRolloutDamper,
    SelfConsistencyDriftEstimator,
)


DETER_DIM = 512
RANK = 10


def make_basis(
    rng: np.random.RandomState,
    deter_dim: int = DETER_DIM,
    rank: int = RANK,
) -> np.ndarray:
    matrix = rng.normal(size=(deter_dim, rank))
    q, _ = np.linalg.qr(matrix, mode="reduced")
    return q


def random_step_pair(
    rng: np.random.RandomState,
    deter_dim: int = DETER_DIM,
) -> tuple[np.ndarray, np.ndarray]:
    return rng.normal(size=deter_dim), rng.normal(size=deter_dim)


def reestimated_entries(history: list[dict[str, object]]) -> list[dict[str, object]]:
    return [entry for entry in history if entry["overlap_with_prev"] is not None]


def test_identity_eta_zero() -> None:
    rng = np.random.RandomState(42)
    damper = HIADRolloutDamper(
        eta=0.0,
        re_est_freq=10,
        window_size=10,
        initial_basis=make_basis(rng),
        basis_mode="initial_basis",
    )

    for _step in range(50):
        z_current, z_raw_next = random_step_pair(rng)
        z_damped = damper.step(z_current, z_raw_next)

        np.testing.assert_allclose(z_damped, z_raw_next, atol=1e-12)


def test_fixed_basis_equivalence() -> None:
    rng = np.random.RandomState(42)
    U = make_basis(rng)
    P = U @ U.T
    damper = HIADRolloutDamper(
        eta=0.20,
        re_est_freq=9999,
        window_size=50,
        initial_basis=U,
        basis_mode="initial_basis",
    )

    for _step in range(50):
        z_current, z_raw_next = random_step_pair(rng)
        delta_z = z_raw_next - z_current
        expected = z_current + (delta_z - 0.20 * (P @ delta_z))

        z_damped = damper.step(z_current, z_raw_next)

        np.testing.assert_allclose(z_damped, expected, atol=1e-10)

    np.testing.assert_allclose(damper.get_current_U(), U, atol=1e-12)
    history = damper.get_basis_history()
    assert len(history) == 1
    assert history[0]["step"] == 0
    assert history[0]["overlap_with_prev"] is None
    np.testing.assert_allclose(history[0]["U"], U, atol=1e-12)


def test_pca_output_shape_and_orthonormality() -> None:
    rng = np.random.RandomState(42)
    estimator = SelfConsistencyDriftEstimator(
        rank=RANK,
        window_size=30,
        deter_dim=DETER_DIM,
    )
    for _step in range(30):
        estimator.record(rng.normal(size=DETER_DIM))

    assert estimator.can_estimate()
    U = estimator.estimate_U()

    assert U.shape == (DETER_DIM, RANK)
    np.testing.assert_allclose(U.T @ U, np.eye(RANK), atol=1e-10)

    for _step in range(5):
        estimator.record(rng.normal(size=DETER_DIM))
    U_refreshed = estimator.estimate_U()

    assert U_refreshed.shape == (DETER_DIM, RANK)
    np.testing.assert_allclose(U_refreshed.T @ U_refreshed, np.eye(RANK), atol=1e-10)


def test_re_estimation_trigger() -> None:
    rng = np.random.RandomState(42)
    initial_U = make_basis(rng, deter_dim=64, rank=5)
    damper = HIADRolloutDamper(
        rank=5,
        eta=0.20,
        re_est_freq=10,
        window_size=10,
        initial_basis=initial_U,
        basis_mode="initial_basis",
        deter_dim=64,
    )

    for step in range(25):
        z_current, z_raw_next = random_step_pair(rng, deter_dim=64)
        damper.step(z_current, z_raw_next)

        if step == 9:
            entries = reestimated_entries(damper.get_basis_history())
            assert len(entries) == 1
            assert entries[0]["step"] == 10
            assert not np.allclose(entries[0]["U"], initial_U)

        if step == 19:
            entries = reestimated_entries(damper.get_basis_history())
            assert len(entries) == 2
            assert entries[1]["step"] == 20

    entries = reestimated_entries(damper.get_basis_history())
    assert len(entries) == 2
    for entry in entries:
        overlap = entry["overlap_with_prev"]
        assert overlap is not None
        assert 0.0 <= overlap <= 1.0


def test_self_bootstrap_undamped_phase() -> None:
    rng = np.random.RandomState(42)
    damper = HIADRolloutDamper(
        rank=3,
        eta=0.30,
        re_est_freq=10,
        window_size=10,
        basis_mode="self_bootstrap",
        deter_dim=64,
    )

    for step in range(15):
        z_current, z_raw_next = random_step_pair(rng, deter_dim=64)
        if step < 10:
            assert damper.get_current_U() is None

        z_damped = damper.step(z_current, z_raw_next)

        if step < 10:
            np.testing.assert_allclose(z_damped, z_raw_next, atol=1e-12)
        else:
            assert not np.allclose(z_damped, z_raw_next)

    assert damper.get_current_U() is not None


def test_damping_only_affects_projected_direction() -> None:
    rng = np.random.RandomState(42)
    U = make_basis(rng, rank=1)
    u = U[:, 0]
    delta_z = rng.normal(size=DETER_DIM)
    component_along_U = (delta_z @ u) * u
    component_orthogonal = delta_z - component_along_U
    damping_step = HIADDampingStep(eta=0.50)
    damping_step.update_projector(U)

    damped = damping_step.damp(delta_z)
    damped_orthogonal = damped - (damped @ u) * u

    np.testing.assert_allclose(damped_orthogonal, component_orthogonal, atol=1e-10)
    np.testing.assert_allclose(damped @ u, 0.50 * (delta_z @ u), atol=1e-10)
