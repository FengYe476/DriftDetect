"""Tests for per-band error curve diagnostics."""

from __future__ import annotations

import numpy as np

from src.diagnostics.error_curves import (
    aggregate_curves,
    per_step_band_l2,
    per_step_band_mse,
    total_error_curve,
)
from src.diagnostics.freq_decompose import DEFAULT_BANDS


def test_per_step_band_mse_shape() -> None:
    rng = np.random.default_rng(0)
    true_signal = rng.normal(size=(100, 17))
    imagined_signal = true_signal + 0.1 * rng.normal(size=(100, 17))

    curves = per_step_band_mse(true_signal, imagined_signal)

    for band in DEFAULT_BANDS:
        assert curves[band].shape == (100,)
        assert np.all(curves[band] >= 0)


def test_per_step_band_mse_identical_signals() -> None:
    rng = np.random.default_rng(1)
    true_signal = rng.normal(size=(100, 17))

    curves = per_step_band_mse(true_signal, true_signal.copy())

    for band in DEFAULT_BANDS:
        assert np.allclose(curves[band], 0.0, atol=1e-6)


def test_per_step_band_mse_known_drift() -> None:
    drift = np.linspace(0.0, 1.0, 100)[:, None]
    true_signal = np.zeros((100, 17))
    imagined_signal = np.repeat(drift, 17, axis=1)

    curves = per_step_band_mse(true_signal, imagined_signal)
    total_errors = {band: float(np.sum(curves[band])) for band in DEFAULT_BANDS}
    dc_curve = curves["dc_trend"]

    assert max(total_errors, key=total_errors.get) == "dc_trend"
    assert np.all(np.diff(dc_curve[10:-10]) >= -1e-4)


def test_per_step_band_l2_vs_mse() -> None:
    rng = np.random.default_rng(2)
    true_signal = rng.normal(size=(100, 17))
    imagined_signal = true_signal + 0.2 * rng.normal(size=(100, 17))

    mse_curves = per_step_band_mse(true_signal, imagined_signal)
    l2_curves = per_step_band_l2(true_signal, imagined_signal)

    for band in DEFAULT_BANDS:
        assert np.all(mse_curves[band] >= 0)
        assert np.all(l2_curves[band] >= 0)
        np.testing.assert_allclose(
            l2_curves[band],
            np.sqrt(mse_curves[band] * true_signal.shape[1]),
            rtol=1e-5,
            atol=1e-6,
        )


def test_aggregate_curves_shape(tmp_path) -> None:
    rng = np.random.default_rng(3)
    paths = []
    for seed in range(3):
        path = tmp_path / f"rollout_{seed}.npz"
        true_obs = rng.normal(size=(50, 17))
        imagined_obs = true_obs + 0.1 * rng.normal(size=(50, 17))
        np.savez(path, true_obs=true_obs, imagined_obs=imagined_obs)
        paths.append(path)

    result = aggregate_curves(
        rollout_paths=paths,
        signal_key_true="true_obs",
        signal_key_imag="imagined_obs",
    )

    for band in DEFAULT_BANDS:
        assert result[band]["mean"].shape == (50,)
        assert result[band]["std"].shape == (50,)
        assert result[band]["all"].shape == (3, 50)
        assert result[band]["ci_low"].shape == (50,)
        assert result[band]["ci_high"].shape == (50,)


def test_aggregate_curves_pooled_pca(tmp_path) -> None:
    rng = np.random.default_rng(4)
    weights = rng.normal(size=(4, 512))
    paths = []
    for seed in range(3):
        path = tmp_path / f"latent_rollout_{seed}.npz"
        base = rng.normal(size=(50, 4))
        true_latent = np.zeros((50, 1536), dtype=np.float32)
        imagined_latent = np.zeros((50, 1536), dtype=np.float32)
        true_deter = base @ weights
        imagined_deter = true_deter + 0.05 * rng.normal(size=(50, 512))
        true_latent[:, 1024:] = true_deter
        imagined_latent[:, 1024:] = imagined_deter
        np.savez(path, true_latent=true_latent, imagined_latent=imagined_latent)
        paths.append(path)

    result = aggregate_curves(
        rollout_paths=paths,
        signal_key_true="true_latent",
        signal_key_imag="imagined_latent",
        var_threshold=0.95,
    )

    info = result["_info"]["decompose_info"]
    assert info["pca_applied"] is True
    per_seed_n_pcs = {
        seed_info["n_pcs"] for seed_info in result["_info"]["per_seed_decompose_info"]
    }
    assert per_seed_n_pcs == {info["n_pcs"]}
    assert result["_info"]["shared_pca"] is True


def test_total_error_curve() -> None:
    drift = np.linspace(0.0, 1.0, 100)[:, None]
    true_signal = np.zeros((100, 17))
    imagined_signal = np.repeat(drift, 17, axis=1)

    curve = total_error_curve(true_signal, imagined_signal)

    assert curve.shape == (100,)
    assert np.all(np.diff(curve) >= -1e-12)
