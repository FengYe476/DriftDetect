"""Self-consistency drift proxy for HIAD.

The estimator records raw RSSM deterministic update vectors from an imagination
rollout and estimates the active update subspace with PCA.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from src.smad.U_estimation import DETER_DIM


class SelfConsistencyDriftEstimator:
    """Estimate drift basis from a sliding buffer of raw RSSM update vectors."""

    def __init__(
        self,
        rank: int,
        window_size: int,
        min_buffer_size: int | None = None,
        deter_dim: int = DETER_DIM,
    ) -> None:
        self.rank = _positive_int(rank, "rank")
        self.window_size = _positive_int(window_size, "window_size")
        self.deter_dim = _positive_int(deter_dim, "deter_dim")
        if self.rank > self.deter_dim:
            raise ValueError(
                f"rank must be <= deter_dim={self.deter_dim}, got {self.rank}."
            )

        self.min_buffer_size = (
            self.rank
            if min_buffer_size is None
            else _positive_int(min_buffer_size, "min_buffer_size")
        )
        if self.min_buffer_size < self.rank:
            raise ValueError(
                "min_buffer_size must be >= rank, got "
                f"{self.min_buffer_size} < {self.rank}."
            )
        if self.min_buffer_size > self.window_size:
            raise ValueError(
                "min_buffer_size must be <= window_size, got "
                f"{self.min_buffer_size} > {self.window_size}."
            )

        self._buffer: deque[np.ndarray] = deque(maxlen=self.window_size)

    def record(self, delta_z: np.ndarray) -> None:
        """Append one ``Delta z_t`` vector to the sliding buffer."""

        self._buffer.append(_as_vector(delta_z, self.deter_dim).copy())

    def can_estimate(self) -> bool:
        """Return true if the buffer has enough vectors for PCA."""

        return len(self._buffer) >= self.min_buffer_size

    def estimate_U(self) -> np.ndarray:
        """Run PCA on the current buffer and return an orthonormal basis."""

        if not self.can_estimate():
            raise ValueError(
                "buffer must contain at least "
                f"{self.min_buffer_size} vectors before PCA, got {len(self._buffer)}."
            )

        update_pool = np.stack(tuple(self._buffer), axis=0).astype(
            np.float64,
            copy=False,
        )
        update_pool = update_pool - update_pool.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(update_pool, full_matrices=False)
        U = vt[: self.rank].T
        _assert_orthonormal(U)
        return U

    def reset(self) -> None:
        """Clear the sliding buffer."""

        self._buffer.clear()


def _positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def _as_vector(array: np.ndarray, deter_dim: int) -> np.ndarray:
    vector = np.asarray(array, dtype=np.float64)
    if vector.ndim != 1 or vector.shape[0] != deter_dim:
        raise ValueError(
            f"delta_z must have shape ({deter_dim},), got {vector.shape}."
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError("delta_z must contain only finite values.")
    return vector


def _assert_orthonormal(U: np.ndarray, atol: float = 1e-10) -> None:
    gram = U.T @ U
    error = float(np.max(np.abs(gram - np.eye(U.shape[1]))))
    if error > atol:
        raise RuntimeError(f"U columns are not orthonormal; max error={error:.3e}.")
