"""Scheduler for Adaptive-SMAD basis refreshes."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.smad.U_estimation import make_projector
from src.smad.anchor_loss import activation_schedule


class AdaptiveSMADScheduler:
    """Coordinate periodic ``U_drift`` re-estimation and patch updates."""

    def __init__(
        self,
        re_estimator: Any,
        img_step_patch: Any,
        re_est_freq: int = 50000,
        activation_s0: int = 50000,
        activation_s1: int = 100000,
        initial_U: np.ndarray | None = None,
    ) -> None:
        if initial_U is None:
            raise ValueError("initial_U is required.")
        if re_est_freq <= 0:
            raise ValueError(f"re_est_freq must be positive, got {re_est_freq}.")

        self.re_estimator = re_estimator
        self.img_step_patch = img_step_patch
        self.re_est_freq = int(re_est_freq)
        self.activation_s0 = int(activation_s0)
        self.activation_s1 = int(activation_s1)
        self.U_baseline = np.asarray(initial_U, dtype=np.float64).copy()
        self.current_U = self.U_baseline.copy()
        self.U_history = [{"step": 0, "U": self.current_U.copy()}]

    def maybe_update(self, step: int) -> dict[str, float | int] | None:
        """Refresh ``U`` at scheduled re-estimation steps."""

        step = int(step)
        if step < self.activation_s0 or step % self.re_est_freq != 0:
            return None

        start_time = time.perf_counter()
        U_new = np.asarray(self.re_estimator.estimate(), dtype=np.float64)
        time_seconds = time.perf_counter() - start_time

        overlap_prev = subspace_overlap(U_new, self.current_U)
        overlap_baseline = subspace_overlap(U_new, self.U_baseline)
        self._update_patch(U_new)
        self.current_U = U_new.copy()
        self.U_history.append({"step": step, "U": U_new.copy()})
        return {
            "step": step,
            "overlap_prev": float(overlap_prev),
            "overlap_baseline": float(overlap_baseline),
            "time_seconds": float(time_seconds),
        }

    def gamma(self, step: int | float) -> float:
        """Return the configured SMAD activation value for ``step``."""

        return activation_schedule(step, self.activation_s0, self.activation_s1)

    def state_dict(self) -> dict[str, Any]:
        """Return checkpointable scheduler state."""

        return {
            "current_U": self.current_U.copy(),
            "U_baseline": self.U_baseline.copy(),
            "U_history": [
                {"step": int(entry["step"]), "U": np.asarray(entry["U"]).copy()}
                for entry in self.U_history
            ],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore scheduler state and update the active patch."""

        self.current_U = np.asarray(state["current_U"], dtype=np.float64).copy()
        self.U_baseline = np.asarray(state["U_baseline"], dtype=np.float64).copy()
        self.U_history = [
            {"step": int(entry["step"]), "U": np.asarray(entry["U"]).copy()}
            for entry in state["U_history"]
        ]
        self._update_patch(self.current_U)

    def _update_patch(self, U: np.ndarray) -> None:
        if hasattr(self.img_step_patch, "update_U"):
            self.img_step_patch.update_U(U)
            return
        if hasattr(self.img_step_patch, "update_projector"):
            self.img_step_patch.update_projector(make_projector(U))
            return
        raise TypeError(
            "img_step_patch must expose update_U(U) or update_projector(P)."
        )


def subspace_overlap(U_a: np.ndarray, U_b: np.ndarray) -> float:
    """Return squared Frobenius subspace overlap in the range [0, 1]."""

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
