"""Per-band error curve utilities for DriftDetect diagnostics.

This module turns frequency-decomposed true and imagined trajectories into
per-step error curves. It also aggregates curves across rollout seeds with a
shared PCA basis for latent-space comparisons.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.diagnostics.freq_decompose import (
    PCAResult,
    decompose_pair,
    extract_deter,
    fit_pca,
)


def per_step_band_mse(
    true_signal: np.ndarray,
    imagined_signal: np.ndarray,
    **decompose_kwargs: Any,
) -> dict[str, Any]:
    """Compute per-step mean squared error for each frequency band."""

    true_dec, imagined_dec = decompose_pair(true_signal, imagined_signal, **decompose_kwargs)
    result: dict[str, Any] = {}
    for band in _band_keys(true_dec):
        diff = true_dec[band] - imagined_dec[band]
        result[band] = np.mean(diff**2, axis=1)
    result["_info"] = {
        **true_dec["_info"],
        "metric": "mse",
        "n_steps": _infer_n_steps(result),
    }
    return result


def per_step_band_l2(
    true_signal: np.ndarray,
    imagined_signal: np.ndarray,
    **decompose_kwargs: Any,
) -> dict[str, Any]:
    """Compute per-step L2 error for each frequency band."""

    true_dec, imagined_dec = decompose_pair(true_signal, imagined_signal, **decompose_kwargs)
    result: dict[str, Any] = {}
    for band in _band_keys(true_dec):
        diff = true_dec[band] - imagined_dec[band]
        result[band] = np.linalg.norm(diff, axis=1)
    result["_info"] = {
        **true_dec["_info"],
        "metric": "l2",
        "n_steps": _infer_n_steps(result),
    }
    return result


def aggregate_curves(
    rollout_paths: list[str | Path],
    signal_key_true: str,
    signal_key_imag: str,
    metric: str = "mse",
    **decompose_kwargs: Any,
) -> dict[str, Any]:
    """Aggregate per-band error curves across multiple rollout files."""

    if not rollout_paths:
        raise ValueError("rollout_paths must contain at least one file.")
    if metric not in {"mse", "l2"}:
        raise ValueError(f"metric must be 'mse' or 'l2', got {metric!r}.")
    if "pca" in decompose_kwargs:
        raise ValueError("aggregate_curves fits shared PCA internally; do not pass pca.")

    paths = [Path(path) for path in rollout_paths]
    true_signals, imagined_signals = _load_rollout_signals(
        paths,
        signal_key_true=signal_key_true,
        signal_key_imag=signal_key_imag,
    )
    shared_pca = _fit_shared_pca(true_signals, decompose_kwargs)

    metric_fn = per_step_band_mse if metric == "mse" else per_step_band_l2
    curve_kwargs = dict(decompose_kwargs)
    if shared_pca is not None:
        curve_kwargs["pca"] = shared_pca

    per_seed_curves = [
        metric_fn(true_signal, imagined_signal, **curve_kwargs)
        for true_signal, imagined_signal in zip(true_signals, imagined_signals)
    ]
    band_names = _band_keys(per_seed_curves[0])
    result: dict[str, Any] = {}
    for band in band_names:
        all_curves = np.stack([curves[band] for curves in per_seed_curves], axis=0)
        result[band] = {
            "mean": np.mean(all_curves, axis=0),
            "std": np.std(all_curves, axis=0),
            "all": all_curves,
            "ci_low": np.percentile(all_curves, 2.5, axis=0),
            "ci_high": np.percentile(all_curves, 97.5, axis=0),
        }

    result["_info"] = {
        "n_seeds": len(paths),
        "metric": metric,
        "signal_keys": {
            "true": signal_key_true,
            "imagined": signal_key_imag,
        },
        "rollout_paths": [str(path) for path in paths],
        "decompose_info": per_seed_curves[0]["_info"],
        "per_seed_decompose_info": [curves["_info"] for curves in per_seed_curves],
        "shared_pca": shared_pca is not None,
    }
    return result


def total_error_curve(true_signal: np.ndarray, imagined_signal: np.ndarray) -> np.ndarray:
    """Compute the raw per-step L2 error without frequency decomposition."""

    true_signal = _as_2d_array(true_signal, name="true_signal")
    imagined_signal = _as_2d_array(imagined_signal, name="imagined_signal")
    _validate_pair_shapes(true_signal, imagined_signal)
    return np.linalg.norm(true_signal - imagined_signal, axis=1)


def _fit_shared_pca(
    true_signals: list[np.ndarray],
    decompose_kwargs: dict[str, Any],
) -> PCAResult | None:
    input_dim = true_signals[0].shape[1]
    n_pcs = decompose_kwargs.get("n_pcs")
    var_threshold = decompose_kwargs.get("var_threshold", 0.95)

    if input_dim == 1536:
        pooled = np.concatenate([extract_deter(signal) for signal in true_signals], axis=0)
    elif input_dim == 512:
        pooled = np.concatenate(true_signals, axis=0)
    elif input_dim <= 50:
        return None
    else:
        raise ValueError(
            "Unsupported signal dimension. Expected 1536 full latent, 512 deter, "
            f"or <=50 obs/PCA dimensions; got {input_dim}."
        )

    pca, _ = fit_pca(
        pooled,
        n_components=n_pcs,
        var_threshold=var_threshold,
    )
    return pca


def _load_rollout_signals(
    paths: list[Path],
    signal_key_true: str,
    signal_key_imag: str,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    true_signals = []
    imagined_signals = []
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            if signal_key_true not in data:
                raise KeyError(f"{path} does not contain {signal_key_true!r}.")
            if signal_key_imag not in data:
                raise KeyError(f"{path} does not contain {signal_key_imag!r}.")
            true_signal = _as_2d_array(data[signal_key_true], name=signal_key_true)
            imagined_signal = _as_2d_array(data[signal_key_imag], name=signal_key_imag)
            _validate_pair_shapes(true_signal, imagined_signal)
            true_signals.append(true_signal)
            imagined_signals.append(imagined_signal)

    reference_shape = true_signals[0].shape
    for path, true_signal in zip(paths, true_signals):
        if true_signal.shape != reference_shape:
            raise ValueError(
                f"All true signals must have shape {reference_shape}; "
                f"{path} has {true_signal.shape}."
            )
    return true_signals, imagined_signals


def _validate_pair_shapes(true_signal: np.ndarray, imagined_signal: np.ndarray) -> None:
    if true_signal.shape != imagined_signal.shape:
        raise ValueError(
            "true_signal and imagined_signal must have the same shape, got "
            f"{true_signal.shape} and {imagined_signal.shape}."
        )


def _as_2d_array(array: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {array.shape}.")
    return array


def _band_keys(decomposed: dict[str, Any]) -> list[str]:
    return [key for key in decomposed if key != "_info"]


def _infer_n_steps(curves: dict[str, Any]) -> int:
    bands = _band_keys(curves)
    if not bands:
        return 0
    return int(curves[bands[0]].shape[0])
