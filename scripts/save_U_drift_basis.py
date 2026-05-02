#!/usr/bin/env python3
"""Estimate and save the fixed U_drift basis used by SMAD Phase 2."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.smad.U_estimation import estimate_U_drift, save_U  # noqa: E402


ROLLOUT_DIR = REPO_ROOT / "results" / "rollouts"
PATTERN = "cheetah_v3_seed*_v2.npz"
RANK = 10
TRIM = (25, 175)
OUTPUT_PATH = REPO_ROOT / "results" / "smad" / "U_drift_cheetah_r10.npy"
DETER_START = 1024


def main() -> None:
    U = estimate_U_drift(
        rollout_dir=ROLLOUT_DIR,
        pattern=PATTERN,
        r=RANK,
        trim=TRIM,
    )
    save_U(U, OUTPUT_PATH)

    explained_variance_ratio = drift_pca_explained_variance_ratio(
        rollout_dir=ROLLOUT_DIR,
        pattern=PATTERN,
        trim=TRIM,
    )
    orth_error = np.linalg.norm(U.T @ U - np.eye(U.shape[1]), ord="fro")

    print("Saved SMAD U_drift basis")
    print(f"Path: {OUTPUT_PATH.relative_to(REPO_ROOT)}")
    print(f"Shape: {U.shape}")
    print(f"Orthonormality ||U.T @ U - I||_F: {orth_error:.6e}")
    print(
        "Top-3 drift PCA explained variance ratios: "
        + ", ".join(f"{value:.6f}" for value in explained_variance_ratio[:3])
    )


def drift_pca_explained_variance_ratio(
    *,
    rollout_dir: Path,
    pattern: str,
    trim: tuple[int, int],
) -> np.ndarray:
    trim_start, trim_end = trim
    paths = sorted(rollout_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No rollout files matched {pattern!r} in {rollout_dir}.")

    drift_chunks = []
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            true_latent = require_array(data, "true_latent", path)
            imagined_latent = require_array(data, "imagined_latent", path)
        true_deter = true_latent[:, DETER_START:].astype(np.float64, copy=False)
        imagined_deter = imagined_latent[:, DETER_START:].astype(
            np.float64,
            copy=False,
        )
        drift_chunks.append((imagined_deter - true_deter)[trim_start : trim_end + 1])

    drift_pool = np.concatenate(drift_chunks, axis=0)
    drift_pool = drift_pool - drift_pool.mean(axis=0, keepdims=True)
    _, singular_values, _ = np.linalg.svd(drift_pool, full_matrices=False)
    explained_variance = singular_values**2 / max(drift_pool.shape[0] - 1, 1)
    total = explained_variance.sum()
    if total <= 0:
        raise ValueError("Drift PCA total variance is zero.")
    return explained_variance / total


def require_array(data: np.lib.npyio.NpzFile, key: str, path: Path) -> np.ndarray:
    if key not in data:
        raise KeyError(f"{path} does not contain {key!r}.")
    return np.asarray(data[key])


if __name__ == "__main__":
    main()
