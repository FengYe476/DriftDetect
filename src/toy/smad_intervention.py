"""SMAD analysis and intervention on the toy RSSM."""

from __future__ import annotations

import types
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from src.diagnostics.freq_decompose import DEFAULT_BANDS, bandpass


RANKS = (3, 5, 10)
VAR_THRESHOLD = 0.95
MIN_NORM = 1e-8


def collect_rollouts(
    rssm,
    env,
    n_episodes: int = 20,
    horizon: int = 200,
) -> dict[str, np.ndarray]:
    """Collect posterior and imagined rollouts from the toy RSSM."""

    true_deter = []
    imagined_deter = []
    true_obs = []
    imagined_obs = []
    actions = []
    rewards = []

    device = next(rssm.parameters()).device
    rssm.eval()
    with torch.no_grad():
        for _ in range(n_episodes):
            obs, action, reward = sample_episode(env, horizon)
            obs_tensor = torch.as_tensor(obs[None], device=device, dtype=torch.float32)
            action_tensor = torch.as_tensor(action[None], device=device, dtype=torch.float32)
            post, _ = rssm.observe(obs_tensor, action_tensor, sample=False)
            start = {key: value[:, 0] for key, value in post.items()}
            imag = rssm.imagine(start, action_tensor, sample=False)
            decoded_imag = rssm.decode_state(imag)

            true_deter.append(post["deter"][0].detach().cpu().numpy())
            imagined_deter.append(imag["deter"][0].detach().cpu().numpy())
            true_obs.append(obs)
            imagined_obs.append(decoded_imag[0].detach().cpu().numpy())
            actions.append(action)
            rewards.append(reward)

    return {
        "true_deter": np.asarray(true_deter, dtype=np.float32),
        "imagined_deter": np.asarray(imagined_deter, dtype=np.float32),
        "true_obs": np.asarray(true_obs, dtype=np.float32),
        "imagined_obs": np.asarray(imagined_obs, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
    }


def analyze_drift(
    true_deter: np.ndarray,
    imagined_deter: np.ndarray,
    trim: tuple[int, int] = (25, 175),
) -> dict[str, Any]:
    """Analyze drift subspaces and frequency-band MSE for toy rollouts."""

    true_deter = as_rollout_array(true_deter, "true_deter")
    imagined_deter = as_rollout_array(imagined_deter, "imagined_deter")
    if true_deter.shape != imagined_deter.shape:
        raise ValueError(
            f"true_deter and imagined_deter shapes differ: "
            f"{true_deter.shape} vs {imagined_deter.shape}."
        )
    validate_trim(trim, true_deter.shape[1])

    max_rank = min(max(RANKS), true_deter.shape[-1])
    drift_fit = fit_drift_basis(true_deter, imagined_deter, max_rank=max_rank, trim=trim)
    posterior_fit = fit_posterior_basis(true_deter, max_rank=max_rank)
    sfa_fit = fit_sfa_basis(true_deter, max_rank=max_rank)
    frequency = frequency_metrics(true_deter, imagined_deter, trim=trim)

    overlaps = {}
    for rank in RANKS:
        if rank > max_rank:
            continue
        U_drift = drift_fit["basis"][:, :rank]
        U_posterior = posterior_fit["basis"][:, :rank]
        U_sfa = sfa_fit["basis"][:, :rank]
        overlaps[str(rank)] = {
            "sfa_posterior": subspace_overlap(U_sfa, U_posterior),
            "sfa_drift": subspace_overlap(U_sfa, U_drift),
            "posterior_drift": subspace_overlap(U_posterior, U_drift),
        }

    return {
        "trim_window_inclusive": [int(trim[0]), int(trim[1])],
        "ranks": [rank for rank in RANKS if rank <= max_rank],
        "overlaps": overlaps,
        "frequency": frequency,
        "drift_pca": {
            "explained_variance_ratio_top10": drift_fit["explained_variance_ratio"][
                :max_rank
            ].tolist(),
            "cumulative_top10": float(
                np.sum(drift_fit["explained_variance_ratio"][:max_rank])
            ),
        },
        "posterior": {
            "pca_components_95pct": int(posterior_fit["pca_components"]),
            "autocorrelation_top10": posterior_fit["autocorr"][:max_rank].tolist(),
        },
        "sfa": {
            "whitened_dims": int(sfa_fit["whitened_dims"]),
            "slow_eigenvalues_top10": sfa_fit["slow_eigenvalues"][:max_rank].tolist(),
        },
        "bases": {
            "U_drift": drift_fit["basis"],
            "U_posterior": posterior_fit["basis"],
            "U_sfa": sfa_fit["basis"],
        },
    }


def apply_damping(
    rssm,
    U_drift: np.ndarray,
    eta: float,
    env,
    n_episodes: int = 20,
    horizon: int = 200,
    baseline_rollouts: Mapping[str, np.ndarray] | None = None,
    trim: tuple[int, int] = (25, 175),
) -> dict[str, Any]:
    """Apply SMAD damping to the toy RSSM and evaluate damped rollouts."""

    if baseline_rollouts is None:
        baseline_rollouts = collect_rollouts(rssm, env, n_episodes=n_episodes, horizon=horizon)
    baseline_freq = frequency_metrics(
        baseline_rollouts["true_deter"],
        baseline_rollouts["imagined_deter"],
        trim=trim,
    )

    restore = patch_toy_img_step(rssm, U_drift, eta)
    try:
        damped_rollouts = collect_rollouts(
            rssm,
            env,
            n_episodes=n_episodes,
            horizon=horizon,
        )
    finally:
        restore()

    damped_freq = frequency_metrics(
        damped_rollouts["true_deter"],
        damped_rollouts["imagined_deter"],
        trim=trim,
    )
    return {
        "eta": float(eta),
        "baseline_frequency": baseline_freq,
        "damped_frequency": damped_freq,
        "J_slow_reduction_pct": percent_reduction(
            baseline_freq["J_slow"],
            damped_freq["J_slow"],
        ),
        "J_total_reduction_pct": percent_reduction(
            baseline_freq["J_total"],
            damped_freq["J_total"],
        ),
        "per_band_reduction_pct": {
            band: percent_reduction(
                baseline_freq["band_mse"][band],
                damped_freq["band_mse"][band],
            )
            for band in DEFAULT_BANDS
        },
    }


def sample_episode(env, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs_list = []
    action_list = []
    reward_list = []
    obs = env.reset()
    for _ in range(horizon):
        action = env.rng.uniform(-1.0, 1.0, size=(1,)).astype(np.float32)
        obs_list.append(obs.astype(np.float32))
        action_list.append(action)
        obs, reward, done, _ = env.step(action)
        reward_list.append(float(reward))
        if done:
            break
    return (
        np.asarray(obs_list, dtype=np.float32),
        np.asarray(action_list, dtype=np.float32),
        np.asarray(reward_list, dtype=np.float32),
    )


def patch_toy_img_step(rssm, U_drift: np.ndarray, eta: float):
    """Monkey-patch a MinimalRSSM instance's img_step with SMAD damping."""

    eta = float(eta)
    original_img_step = rssm.img_step
    device = next(rssm.parameters()).device
    U = torch.as_tensor(U_drift, dtype=torch.float32, device=device)
    P = U @ U.T

    def img_step_damped(self, prev_state, action, sample=True):
        if eta == 0.0:
            return original_img_step(prev_state, action, sample=sample)
        prior = original_img_step(prev_state, action, sample=False)
        delta = prior["deter"] - prev_state["deter"]
        P_local = P.to(device=delta.device, dtype=delta.dtype)
        deter = prior["deter"] - eta * (delta @ P_local.T)
        mean, logvar = self.prior_stats(deter)
        stoch = mean if not sample else mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        return {"stoch": stoch, "deter": deter, "mean": mean, "logvar": logvar}

    rssm.img_step = types.MethodType(img_step_damped, rssm)

    def restore() -> None:
        rssm.img_step = original_img_step

    return restore


def fit_drift_basis(
    true_deter: np.ndarray,
    imagined_deter: np.ndarray,
    *,
    max_rank: int,
    trim: tuple[int, int],
) -> dict[str, np.ndarray]:
    drift = imagined_deter[:, trim[0] : trim[1] + 1] - true_deter[:, trim[0] : trim[1] + 1]
    matrix = drift.reshape(-1, drift.shape[-1])
    return pca_basis(matrix, max_rank=max_rank)


def fit_posterior_basis(true_deter: np.ndarray, *, max_rank: int) -> dict[str, Any]:
    flat = true_deter.reshape(-1, true_deter.shape[-1])
    pca = pca_basis(flat, max_rank=true_deter.shape[-1])
    retained = max(
        max_rank,
        components_for_variance(pca["explained_variance_ratio"], VAR_THRESHOLD),
    )
    retained = min(retained, pca["components"].shape[0])
    components = pca["components"][:retained]
    projected = (true_deter - pca["mean"].reshape(1, 1, -1)) @ components.T
    autocorr = lag1_autocorr(projected)
    order = np.argsort(autocorr)[::-1]
    selected = order[:max_rank]
    basis = components[selected].T
    return {
        "basis": orthonormalize(basis),
        "pca_components": retained,
        "autocorr": autocorr[selected],
        "selected_indices": selected,
    }


def fit_sfa_basis(true_deter: np.ndarray, *, max_rank: int) -> dict[str, Any]:
    flat = true_deter.reshape(-1, true_deter.shape[-1]).astype(np.float64)
    mean = flat.mean(axis=0, keepdims=True)
    std = flat.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    z_rollouts = (true_deter - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
    z_flat = z_rollouts.reshape(-1, true_deter.shape[-1])
    pca = pca_basis(z_flat, max_rank=true_deter.shape[-1])
    retained = max(
        max_rank,
        components_for_variance(pca["explained_variance_ratio"], VAR_THRESHOLD),
    )
    retained = min(retained, pca["components"].shape[0])
    components = pca["components"][:retained]
    eigvals = pca["eigenvalues"][:retained]
    whiten = components.T / np.sqrt(np.maximum(eigvals, 1e-8))

    delta = np.diff(z_rollouts, axis=1).reshape(-1, true_deter.shape[-1])
    delta_white = delta @ whiten
    cov_delta = np.cov(delta_white, rowvar=False)
    slow_vals, slow_vecs = np.linalg.eigh(cov_delta)
    order = np.argsort(slow_vals)
    slow_vals = slow_vals[order]
    slow_vecs = slow_vecs[:, order]
    original_basis = whiten @ slow_vecs[:, :max_rank]
    original_basis = original_basis / std.reshape(-1, 1)
    return {
        "basis": orthonormalize(original_basis),
        "whitened_dims": retained,
        "slow_eigenvalues": slow_vals,
    }


def frequency_metrics(
    true_deter: np.ndarray,
    imagined_deter: np.ndarray,
    trim: tuple[int, int],
) -> dict[str, Any]:
    true_deter = as_rollout_array(true_deter, "true_deter")
    imagined_deter = as_rollout_array(imagined_deter, "imagined_deter")
    pca = pca_basis(true_deter.reshape(-1, true_deter.shape[-1]), max_rank=true_deter.shape[-1])
    retained = components_for_variance(pca["explained_variance_ratio"], VAR_THRESHOLD)
    components = pca["components"][:retained]
    true_proj = (true_deter - pca["mean"].reshape(1, 1, -1)) @ components.T
    imag_proj = (imagined_deter - pca["mean"].reshape(1, 1, -1)) @ components.T
    true_bands = filter_bands(true_proj)
    imag_bands = filter_bands(imag_proj)
    band_mse = {}
    window = slice(trim[0], trim[1] + 1)
    for band in DEFAULT_BANDS:
        band_mse[band] = float(np.mean((imag_bands[band][:, window] - true_bands[band][:, window]) ** 2))
    total = float(sum(band_mse.values()))
    return {
        "band_mse": band_mse,
        "band_share": {
            band: float(band_mse[band] / total) if total > 0 else 0.0
            for band in DEFAULT_BANDS
        },
        "J_slow": float(band_mse["dc_trend"]),
        "J_total": total,
        "pca_components": int(retained),
        "pca_var_explained": float(np.sum(pca["explained_variance_ratio"][:retained])),
    }


def filter_bands(projected: np.ndarray) -> dict[str, np.ndarray]:
    result = {}
    for band, (low, high) in DEFAULT_BANDS.items():
        out = np.empty_like(projected, dtype=np.float64)
        for episode in range(projected.shape[0]):
            for dim in range(projected.shape[2]):
                out[episode, :, dim] = bandpass(projected[episode, :, dim], low, high)
        result[band] = out
    return result


def pca_basis(matrix: np.ndarray, *, max_rank: int) -> dict[str, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float64)
    mean = matrix.mean(axis=0)
    centered = matrix - mean.reshape(1, -1)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = singular_values**2 / max(matrix.shape[0] - 1, 1)
    total = float(np.sum(eigenvalues))
    ratios = eigenvalues / total if total > 0 else np.zeros_like(eigenvalues)
    return {
        "basis": vt[:max_rank].T,
        "components": vt,
        "mean": mean,
        "eigenvalues": eigenvalues,
        "explained_variance_ratio": ratios,
    }


def lag1_autocorr(projected: np.ndarray) -> np.ndarray:
    values = []
    for dim in range(projected.shape[-1]):
        per_episode = []
        for episode in range(projected.shape[0]):
            x = projected[episode, :, dim]
            x0 = x[:-1] - x[:-1].mean()
            x1 = x[1:] - x[1:].mean()
            denom = float(np.linalg.norm(x0) * np.linalg.norm(x1))
            per_episode.append(0.0 if denom < MIN_NORM else float(np.dot(x0, x1) / denom))
        values.append(float(np.mean(per_episode)))
    return np.asarray(values, dtype=np.float64)


def components_for_variance(ratios: np.ndarray, threshold: float) -> int:
    if ratios.size == 0:
        raise ValueError("ratios must be non-empty.")
    cumulative = np.cumsum(ratios)
    if cumulative[-1] <= 0:
        return 1
    return min(int(np.searchsorted(cumulative, threshold, side="left") + 1), ratios.size)


def subspace_overlap(U_a: np.ndarray, U_b: np.ndarray) -> float:
    rank = U_a.shape[1]
    return float(np.linalg.norm(U_a.T @ U_b, ord="fro") ** 2 / rank)


def orthonormalize(matrix: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(matrix)
    return q[:, : matrix.shape[1]]


def percent_reduction(original: float, new: float) -> float:
    if original == 0:
        return 0.0 if new == 0 else float("-inf")
    return float((original - new) / original * 100.0)


def as_rollout_array(array: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape (N, T, D), got {array.shape}.")
    return array


def validate_trim(trim: tuple[int, int], horizon: int) -> None:
    if not 0 <= trim[0] <= trim[1] < horizon:
        raise ValueError(f"Invalid trim window {trim} for horizon {horizon}.")
