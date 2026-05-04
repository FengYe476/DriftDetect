"""Rollout-level orchestration for HIAD inference damping."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.smad.U_estimation import DETER_DIM
from src.smad.hiad.damping_inference import HIADDampingStep
from src.smad.hiad.self_consistency_drift import SelfConsistencyDriftEstimator


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INITIAL_BASIS = REPO_ROOT / "results" / "smad" / "U_drift_cheetah_r10.npy"


class HIADRolloutDamper:
    """Orchestrate a single HIAD-damped imagination rollout."""

    def __init__(
        self,
        rank: int = 10,
        eta: float = 0.20,
        re_est_freq: int = 25,
        window_size: int = 25,
        min_buffer_size: int | None = None,
        initial_basis: str | np.ndarray | None = None,
        basis_mode: str = "initial_basis",
        deter_dim: int = DETER_DIM,
    ) -> None:
        if basis_mode not in {"initial_basis", "self_bootstrap"}:
            raise ValueError(
                "basis_mode must be 'initial_basis' or 'self_bootstrap', "
                f"got {basis_mode!r}."
            )
        self.basis_mode = basis_mode
        self.re_est_freq = _positive_int(re_est_freq, "re_est_freq")
        self.deter_dim = _positive_int(deter_dim, "deter_dim")

        self.estimator = SelfConsistencyDriftEstimator(
            rank=rank,
            window_size=window_size,
            min_buffer_size=min_buffer_size,
            deter_dim=self.deter_dim,
        )
        self.damping_step = HIADDampingStep(eta=eta)
        self.step_count = 0
        self.current_U: np.ndarray | None = None
        self.basis_history: list[dict[str, Any]] = []

        if self.basis_mode == "initial_basis":
            U = _load_basis(initial_basis)
            self._set_basis(U, step=0, overlap_with_prev=None)

    def step(
        self,
        z_deter_current: np.ndarray,
        z_deter_raw_next: np.ndarray,
    ) -> np.ndarray:
        """Process one imagination step and return the damped next deter state."""

        current = _as_vector(z_deter_current, self.deter_dim, "z_deter_current")
        raw_next = _as_vector(z_deter_raw_next, self.deter_dim, "z_deter_raw_next")
        delta_z = raw_next - current

        self.estimator.record(delta_z)
        damped_next = current + self.damping_step.damp(delta_z)

        self.step_count += 1
        if (
            self.step_count % self.re_est_freq == 0
            and self.estimator.can_estimate()
        ):
            U_new = self.estimator.estimate_U()
            overlap = (
                None
                if self.current_U is None
                else subspace_overlap(self.current_U, U_new)
            )
            self._set_basis(U_new, step=self.step_count, overlap_with_prev=overlap)

        return damped_next

    def get_basis_history(self) -> list[dict[str, Any]]:
        """Return a defensive copy of the basis history."""

        return [
            {
                "step": int(entry["step"]),
                "U": np.asarray(entry["U"], dtype=np.float64).copy(),
                "overlap_with_prev": entry["overlap_with_prev"],
            }
            for entry in self.basis_history
        ]

    def get_current_U(self) -> np.ndarray | None:
        """Return the current basis, or ``None`` before bootstrap estimation."""

        if self.current_U is None:
            return None
        return self.current_U.copy()

    def _set_basis(
        self,
        U: np.ndarray,
        *,
        step: int,
        overlap_with_prev: float | None,
    ) -> None:
        U = _as_basis(U, self.deter_dim, self.estimator.rank)
        self.damping_step.update_projector(U)
        self.current_U = U.copy()
        self.basis_history.append(
            {
                "step": int(step),
                "U": U.copy(),
                "overlap_with_prev": overlap_with_prev,
            }
        )


def subspace_overlap(U_a: np.ndarray, U_b: np.ndarray) -> float:
    """Return squared Frobenius subspace overlap in the range ``[0, 1]``."""

    U_a = np.asarray(U_a, dtype=np.float64)
    U_b = np.asarray(U_b, dtype=np.float64)
    if U_a.ndim != 2 or U_b.ndim != 2:
        raise ValueError("U_a and U_b must both be 2D basis matrices.")
    if U_a.shape != U_b.shape:
        raise ValueError(f"basis shapes must match, got {U_a.shape} and {U_b.shape}.")
    rank = U_a.shape[1]
    if rank <= 0:
        raise ValueError("basis rank must be positive.")
    return float(np.linalg.norm(U_a.T @ U_b, ord="fro") ** 2 / rank)


def _load_basis(initial_basis: str | np.ndarray | None) -> np.ndarray:
    if initial_basis is None:
        return np.load(DEFAULT_INITIAL_BASIS)
    if isinstance(initial_basis, np.ndarray):
        return np.asarray(initial_basis, dtype=np.float64).copy()
    path = Path(initial_basis)
    return np.load(path)


def _positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def _as_vector(array: np.ndarray, deter_dim: int, name: str) -> np.ndarray:
    vector = np.asarray(array, dtype=np.float64)
    if vector.ndim != 1 or vector.shape[0] != deter_dim:
        raise ValueError(f"{name} must have shape ({deter_dim},), got {vector.shape}.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")
    return vector.copy()


def _as_basis(U: np.ndarray, deter_dim: int, rank: int) -> np.ndarray:
    basis = np.asarray(U, dtype=np.float64)
    if basis.ndim != 2 or basis.shape != (deter_dim, rank):
        raise ValueError(
            f"U must have shape ({deter_dim}, {rank}), got {basis.shape}."
        )
    if not np.all(np.isfinite(basis)):
        raise ValueError("U must contain only finite values.")
    return basis.copy()
