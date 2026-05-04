"""NumPy-level HIAD damping for deterministic RSSM updates."""

from __future__ import annotations

import numpy as np


class HIADDampingStep:
    """Apply projected damping to a single RSSM deterministic update."""

    def __init__(self, eta: float) -> None:
        self._eta = float(eta)
        if self._eta < 0.0:
            raise ValueError(f"eta must be non-negative, got {eta}.")
        self._P: np.ndarray | None = None

    def update_projector(self, U: np.ndarray) -> None:
        """Recompute ``P = U @ U.T`` from a new basis."""

        U = _as_basis(U)
        self._P = U @ U.T

    def damp(self, delta_z: np.ndarray) -> np.ndarray:
        """Return ``delta_z - eta * P @ delta_z``.

        If no projector has been set, this is the bootstrap identity path.
        """

        vector = np.asarray(delta_z, dtype=np.float64)
        if vector.ndim != 1:
            raise ValueError(f"delta_z must be a 1D vector, got {vector.shape}.")
        if not np.all(np.isfinite(vector)):
            raise ValueError("delta_z must contain only finite values.")
        if self._P is None or self._eta == 0.0:
            return vector.copy()
        if vector.shape[0] != self._P.shape[0]:
            raise ValueError(
                f"delta_z length must match projector dim {self._P.shape[0]}, "
                f"got {vector.shape[0]}."
            )
        return vector - self._eta * (self._P @ vector)

    @property
    def active(self) -> bool:
        """Return true if a projector has been set."""

        return self._P is not None


def _as_basis(U: np.ndarray) -> np.ndarray:
    basis = np.asarray(U, dtype=np.float64)
    if basis.ndim != 2:
        raise ValueError(f"U must have shape (D, r), got {basis.shape}.")
    if basis.shape[0] <= 0 or basis.shape[1] <= 0:
        raise ValueError(f"U must have non-empty dimensions, got {basis.shape}.")
    if not np.all(np.isfinite(basis)):
        raise ValueError("U must contain only finite values.")
    return basis.copy()
