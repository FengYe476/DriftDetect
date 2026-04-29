"""Frequency decomposition utilities for DriftDetect latent diagnostics.

The main path operates on DreamerV3 RSSM features by extracting the continuous
deterministic state, projecting it with PCA, and filtering each component into
temporal frequency bands. Low-dimensional observation-space inputs are also
supported as a direct filtering path for sanity checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt


DEFAULT_BANDS = {
    "dc_trend": (0.0, 0.02),
    "very_low": (0.02, 0.05),
    "low": (0.05, 0.10),
    "mid": (0.10, 0.20),
    "high": (0.20, 0.50),
}
DETER_SLICE = slice(1024, 1536)  # get_feat() = concat([stoch(1024), deter(512)])


@dataclass(frozen=True)
class PCAResult:
    """Minimal PCA object backed by NumPy SVD."""

    mean_: np.ndarray
    components_: np.ndarray
    explained_variance_ratio_: np.ndarray
    all_explained_variance_ratio_: np.ndarray
    n_components_: int


def extract_deter(latent: np.ndarray) -> np.ndarray:
    """Extract the continuous DreamerV3 deter state from full RSSM features."""

    latent = np.asarray(latent)
    if latent.ndim != 2 or latent.shape[1] != 1536:
        raise ValueError(f"Expected latent shape (T, 1536), got {latent.shape}.")
    return latent[:, DETER_SLICE]


def fit_pca(
    deter: np.ndarray,
    n_components: int | None = None,
    var_threshold: float = 0.95,
) -> tuple[PCAResult, int]:
    """Fit PCA to deter trajectories using NumPy SVD."""

    deter = _as_2d_float(deter, name="deter")
    if deter.shape[0] < 2:
        raise ValueError("PCA requires at least two time points.")
    if not 0 < var_threshold <= 1:
        raise ValueError("var_threshold must be in the interval (0, 1].")

    centered = deter - deter.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    explained_variance = singular_values**2 / max(deter.shape[0] - 1, 1)
    total_variance = explained_variance.sum()
    if total_variance <= 0:
        all_ratios = np.zeros_like(explained_variance)
        all_ratios[0] = 1.0
    else:
        all_ratios = explained_variance / total_variance

    max_components = vt.shape[0]
    if n_components is None:
        cumulative = np.cumsum(all_ratios)
        k = int(np.searchsorted(cumulative, var_threshold, side="left") + 1)
    else:
        k = int(n_components)
    if not 1 <= k <= max_components:
        raise ValueError(f"n_components must be between 1 and {max_components}, got {k}.")

    pca = PCAResult(
        mean_=deter.mean(axis=0),
        components_=vt[:k],
        explained_variance_ratio_=all_ratios[:k],
        all_explained_variance_ratio_=all_ratios,
        n_components_=k,
    )
    return pca, k


def apply_pca(deter: np.ndarray, pca: PCAResult) -> np.ndarray:
    """Project deter trajectories into a fitted PCA basis."""

    deter = _as_2d_float(deter, name="deter")
    if deter.shape[1] != pca.mean_.shape[0]:
        raise ValueError(
            f"Expected input dimension {pca.mean_.shape[0]}, got {deter.shape[1]}."
        )
    return (deter - pca.mean_) @ pca.components_.T


def bandpass(
    signal_1d: np.ndarray,
    low: float,
    high: float,
    fs: float = 1.0,
    order: int = 4,
) -> np.ndarray:
    """Filter a 1D signal into a frequency band with zero phase shift."""

    signal_1d = np.asarray(signal_1d, dtype=np.float64)
    if signal_1d.ndim != 1:
        raise ValueError(f"bandpass expects a 1D signal, got shape {signal_1d.shape}.")
    if signal_1d.size < 2:
        return signal_1d.copy()
    if fs <= 0:
        raise ValueError("fs must be positive.")
    if order <= 0:
        raise ValueError("order must be positive.")
    if low < 0 or high < 0:
        raise ValueError("Band frequencies must be non-negative.")
    if high < low:
        raise ValueError("high must be greater than or equal to low.")

    nyquist = fs / 2.0
    if low == 0 and high >= nyquist:
        return signal_1d.copy()
    if high <= 0:
        return np.full_like(signal_1d, signal_1d.mean())

    high = min(high, nyquist)
    if low == 0:
        sos = butter(order, high, btype="lowpass", output="sos", fs=fs)
    elif high >= nyquist:
        sos = butter(order, low, btype="highpass", output="sos", fs=fs)
    else:
        sos = butter(order, (low, high), btype="bandpass", output="sos", fs=fs)

    padlen = min(signal_1d.size - 1, 3 * (2 * sos.shape[0] + 1))
    return sosfiltfilt(sos, signal_1d, padlen=padlen)


def decompose(
    signal: np.ndarray,
    fs: float = 1.0,
    bands: dict[str, tuple[float, float]] | None = None,
    n_pcs: int | None = None,
    var_threshold: float = 0.95,
    pca: PCAResult | None = None,
) -> dict[str, Any]:
    """Decompose latent, deter-only, PCA, or obs-space signals into bands."""

    prepared, info, _ = _prepare_signal(
        signal=signal,
        n_pcs=n_pcs,
        var_threshold=var_threshold,
        pca=pca,
    )
    return _filter_prepared(prepared, info, fs=fs, bands=bands)


def decompose_pair(
    true_signal: np.ndarray,
    imagined_signal: np.ndarray,
    **kwargs: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Decompose true and imagined signals with a shared PCA basis."""

    fs = kwargs.pop("fs", 1.0)
    bands = kwargs.pop("bands", None)
    n_pcs = kwargs.pop("n_pcs", None)
    var_threshold = kwargs.pop("var_threshold", 0.95)
    pca = kwargs.pop("pca", None)
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected keyword arguments: {unknown}")

    true_prepared, true_info, fitted_pca = _prepare_signal(
        signal=true_signal,
        n_pcs=n_pcs,
        var_threshold=var_threshold,
        pca=pca,
    )
    shared_pca = fitted_pca if true_info["pca_applied"] else None
    imagined_prepared, imagined_info, _ = _prepare_signal(
        signal=imagined_signal,
        n_pcs=n_pcs,
        var_threshold=var_threshold,
        pca=shared_pca,
    )

    true_decomposed = _filter_prepared(true_prepared, true_info, fs=fs, bands=bands)
    imagined_decomposed = _filter_prepared(
        imagined_prepared,
        imagined_info,
        fs=fs,
        bands=bands,
    )
    return true_decomposed, imagined_decomposed


def _prepare_signal(
    signal: np.ndarray,
    n_pcs: int | None,
    var_threshold: float,
    pca: PCAResult | None,
) -> tuple[np.ndarray, dict[str, Any], PCAResult | None]:
    signal = _as_2d_float(signal, name="signal")
    input_dim = signal.shape[1]

    if input_dim == 1536:
        pca_input = extract_deter(signal)
        return _prepare_with_pca(
            pca_input,
            input_dim=input_dim,
            n_pcs=n_pcs,
            var_threshold=var_threshold,
            pca=pca,
        )
    if input_dim == 512:
        return _prepare_with_pca(
            signal,
            input_dim=input_dim,
            n_pcs=n_pcs,
            var_threshold=var_threshold,
            pca=pca,
        )
    if input_dim <= 50:
        if pca is not None:
            raise ValueError("PCA objects are only valid for latent or deter inputs.")
        info = {
            "n_pcs": input_dim,
            "var_explained": 1.0,
            "input_dim": input_dim,
            "output_dim": input_dim,
            "pca_applied": False,
        }
        return signal, info, None
    raise ValueError(
        "Unsupported signal dimension. Expected 1536 full latent, 512 deter, "
        f"or <=50 obs/PCA dimensions; got {input_dim}."
    )


def _prepare_with_pca(
    signal: np.ndarray,
    input_dim: int,
    n_pcs: int | None,
    var_threshold: float,
    pca: PCAResult | None,
) -> tuple[np.ndarray, dict[str, Any], PCAResult]:
    if pca is None:
        pca, _ = fit_pca(signal, n_components=n_pcs, var_threshold=var_threshold)
    projected = apply_pca(signal, pca)
    var_explained = float(np.sum(pca.explained_variance_ratio_))
    info = {
        "n_pcs": pca.n_components_,
        "var_explained": var_explained,
        "input_dim": input_dim,
        "output_dim": pca.n_components_,
        "pca_applied": True,
    }
    return projected, info, pca


def _filter_prepared(
    prepared: np.ndarray,
    info: dict[str, Any],
    fs: float,
    bands: dict[str, tuple[float, float]] | None,
) -> dict[str, Any]:
    active_bands = DEFAULT_BANDS if bands is None else bands
    result: dict[str, Any] = {}
    for name, (low, high) in active_bands.items():
        filtered = np.empty_like(prepared, dtype=np.float64)
        for dim in range(prepared.shape[1]):
            filtered[:, dim] = bandpass(prepared[:, dim], low=low, high=high, fs=fs)
        result[name] = filtered.astype(np.float32)

    result["_info"] = {
        **info,
        "bands": {name: tuple(bounds) for name, bounds in active_bands.items()},
        "fs": fs,
    }
    return result


def _as_2d_float(array: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {array.shape}.")
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"{name} must be non-empty, got shape {array.shape}.")
    return array
