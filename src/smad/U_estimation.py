"""U_drift basis estimation for SMAD.

Estimates the top-r PCA directions of imagination drift vectors from v2 rollout
files. The resulting basis is saved as a .npy file and loaded at training time
as a fixed, frozen projection matrix.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


DETER_START = 1024
DETER_DIM = 512


def estimate_U_drift(
    rollout_dir: str | Path,
    pattern: str,
    r: int = 10,
    trim: tuple[int, int] = (25, 175),
) -> np.ndarray:
    """Estimate a rank-r drift basis from matching rollout files.

    Args:
        rollout_dir: Directory containing rollout .npz files.
        pattern: Glob pattern, for example ``"cheetah_v3_seed*_v2.npz"``.
        r: Number of drift PCA directions to keep.
        trim: Inclusive time window used for drift PCA.

    Returns:
        ``U_drift`` with shape ``(512, r)`` and orthonormal columns.
    """

    if not 1 <= int(r) <= DETER_DIM:
        raise ValueError(f"r must be in [1, {DETER_DIM}], got {r}.")
    trim_start, trim_end = trim
    if not 0 <= trim_start <= trim_end:
        raise ValueError(f"trim must be an inclusive non-empty window, got {trim}.")

    rollout_paths = sorted(Path(rollout_dir).glob(pattern))
    if not rollout_paths:
        raise FileNotFoundError(
            f"No rollout files matched {pattern!r} in {Path(rollout_dir)}."
        )

    drift_chunks = []
    for path in rollout_paths:
        with np.load(path, allow_pickle=True) as data:
            true_latent = _require_array(data, "true_latent", path)
            imagined_latent = _require_array(data, "imagined_latent", path)
        _validate_latent(true_latent, "true_latent", path)
        _validate_latent(imagined_latent, "imagined_latent", path)
        if trim_end >= true_latent.shape[0]:
            raise ValueError(
                f"trim end {trim_end} is outside horizon {true_latent.shape[0]} "
                f"for {path}."
            )
        true_deter = true_latent[:, DETER_START:].astype(np.float64, copy=False)
        imagined_deter = imagined_latent[:, DETER_START:].astype(
            np.float64,
            copy=False,
        )
        drift = imagined_deter - true_deter
        drift_chunks.append(drift[trim_start : trim_end + 1])

    drift_pool = np.concatenate(drift_chunks, axis=0)
    return estimate_U_from_drift(drift_pool, r=r)


def estimate_U_from_drift(drift_pool: np.ndarray, r: int = 10) -> np.ndarray:
    """Estimate a rank-r drift basis from pooled deterministic drift vectors.

    Args:
        drift_pool: Array with shape ``(N, 512)`` containing deterministic
            drift vectors.
        r: Number of drift PCA directions to keep.

    Returns:
        ``U_drift`` with shape ``(512, r)`` and orthonormal columns.
    """

    if not 1 <= int(r) <= DETER_DIM:
        raise ValueError(f"r must be in [1, {DETER_DIM}], got {r}.")
    drift_pool = np.asarray(drift_pool, dtype=np.float64)
    if drift_pool.ndim != 2 or drift_pool.shape[1] != DETER_DIM:
        raise ValueError(
            f"drift_pool must have shape (N, {DETER_DIM}), got {drift_pool.shape}."
        )
    if drift_pool.shape[0] < int(r):
        raise ValueError(
            f"drift_pool must contain at least r={r} vectors, got {drift_pool.shape[0]}."
        )
    drift_pool = drift_pool - drift_pool.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(drift_pool, full_matrices=False)
    U = vt[:r].T
    _assert_orthonormal(U)
    return U


def save_U(U: np.ndarray, path: str | Path) -> None:
    """Save a drift basis to a ``.npy`` file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(U))


def load_U(path: str | Path) -> np.ndarray:
    """Load a drift basis from a ``.npy`` file."""

    U = np.load(Path(path))
    _validate_U(U)
    return U


def make_projector(U: np.ndarray, device: str | "torch.device" = "cpu") -> "torch.Tensor":
    """Return the fixed projection matrix ``P = U @ U.T`` as a torch tensor."""

    import torch

    _validate_U(U)
    U_tensor = torch.as_tensor(U, dtype=torch.float32, device=device)
    return U_tensor @ U_tensor.T


def _require_array(data: np.lib.npyio.NpzFile, key: str, path: Path) -> np.ndarray:
    if key not in data:
        raise KeyError(f"{path} does not contain {key!r}.")
    return np.asarray(data[key])


def _validate_latent(array: np.ndarray, key: str, path: Path) -> None:
    if array.ndim != 2 or array.shape[1] < DETER_START + DETER_DIM:
        raise ValueError(
            f"{path}:{key} expected shape (T, >=1536), got {array.shape}."
        )


def _validate_U(U: np.ndarray) -> None:
    if U.ndim != 2 or U.shape[0] != DETER_DIM:
        raise ValueError(f"U must have shape ({DETER_DIM}, r), got {U.shape}.")
    if not 1 <= U.shape[1] <= DETER_DIM:
        raise ValueError(f"U rank must be in [1, {DETER_DIM}], got {U.shape[1]}.")


def _assert_orthonormal(U: np.ndarray, atol: float = 1e-10) -> None:
    _validate_U(U)
    gram = U.T @ U
    error = float(np.max(np.abs(gram - np.eye(U.shape[1]))))
    if error > atol:
        raise RuntimeError(f"U columns are not orthonormal; max error={error:.3e}.")
