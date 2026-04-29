"""Tests for frequency decomposition diagnostics."""

from __future__ import annotations

import numpy as np

from src.diagnostics.freq_decompose import (
    DEFAULT_BANDS,
    decompose,
    decompose_pair,
    extract_deter,
    fit_pca,
)


def band_energy(decomposed: dict, band: str) -> float:
    return float(np.sum(decomposed[band] ** 2))


def test_extract_deter() -> None:
    latent = np.arange(50 * 1536, dtype=np.float32).reshape(50, 1536)
    deter = extract_deter(latent)

    assert deter.shape == (50, 512)
    np.testing.assert_array_equal(deter, latent[:, 1024:])


def test_pca_variance_threshold() -> None:
    rng = np.random.default_rng(0)
    dominant = rng.normal(scale=10.0, size=(200, 5))
    noise = rng.normal(scale=0.01, size=(200, 507))
    deter = np.concatenate([dominant, noise], axis=1)

    _pca, k = fit_pca(deter, var_threshold=0.95)

    assert 4 <= k <= 6


def test_bandpass_pure_sine() -> None:
    t = np.arange(200)
    signal = np.sin(2 * np.pi * 0.07 * t)[:, None]

    decomposed = decompose(signal, bands=DEFAULT_BANDS)
    energies = {band: band_energy(decomposed, band) for band in DEFAULT_BANDS}
    total_energy = sum(energies.values())

    assert energies["low"] / total_energy > 0.90
    for band in ("dc_trend", "very_low", "mid", "high"):
        assert energies[band] / total_energy < 0.05


def test_bandpass_multi_frequency() -> None:
    t = np.arange(200)
    very_low_component = np.sin(2 * np.pi * 0.03 * t)
    mid_component = np.sin(2 * np.pi * 0.15 * t)
    signal = (very_low_component + mid_component)[:, None]

    decomposed = decompose(signal, bands=DEFAULT_BANDS)
    very_low_corr = np.corrcoef(
        decomposed["very_low"][:, 0],
        very_low_component,
    )[0, 1]
    mid_corr = np.corrcoef(decomposed["mid"][:, 0], mid_component)[0, 1]
    energies = {band: band_energy(decomposed, band) for band in DEFAULT_BANDS}
    total_energy = sum(energies.values())

    assert very_low_corr > 0.85
    assert mid_corr > 0.85
    assert energies["very_low"] / total_energy > 0.30
    assert energies["mid"] / total_energy > 0.30
    for band in ("dc_trend", "low", "high"):
        assert energies[band] / total_energy < 0.10


def test_decompose_pair_shared_pca() -> None:
    rng = np.random.default_rng(1)
    base = rng.normal(size=(200, 4))
    weights = rng.normal(size=(4, 512))
    true_deter = base @ weights
    imagined_deter = true_deter + 0.05 * rng.normal(size=(200, 512))
    true_signal = np.zeros((200, 1536), dtype=np.float32)
    imagined_signal = np.zeros((200, 1536), dtype=np.float32)
    true_signal[:, 1024:] = true_deter
    imagined_signal[:, 1024:] = imagined_deter

    true_dec, imagined_dec = decompose_pair(
        true_signal,
        imagined_signal,
        var_threshold=0.95,
    )

    assert true_dec["_info"]["pca_applied"] is True
    assert imagined_dec["_info"]["pca_applied"] is True
    assert true_dec["_info"]["n_pcs"] == imagined_dec["_info"]["n_pcs"]
    assert true_dec["_info"]["bands"] == imagined_dec["_info"]["bands"]
    for band in DEFAULT_BANDS:
        assert true_dec[band].shape == imagined_dec[band].shape


def test_obs_space_no_pca() -> None:
    rng = np.random.default_rng(2)
    obs = rng.normal(size=(200, 17))

    decomposed = decompose(obs)

    assert decomposed["_info"]["pca_applied"] is False
    for band in DEFAULT_BANDS:
        assert decomposed[band].shape == (200, 17)


def test_band_reconstruction() -> None:
    t = np.arange(200)
    signal = np.sin(2 * np.pi * 0.07 * t)
    signal = signal[:, None]

    decomposed = decompose(signal, bands=DEFAULT_BANDS)
    reconstructed = sum(decomposed[band] for band in DEFAULT_BANDS)

    assert np.allclose(reconstructed[20:-20], signal[20:-20], atol=0.15)
