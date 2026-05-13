#!/usr/bin/env python3
"""Run SHARP fidelity diagnostics on existing V3 rollout archives.

The diagnostics are intentionally offline-only: this script reads rollout
``.npz`` files and never loads Dreamer checkpoints or runs the model. Current
Month 9 rollouts store realized environment rewards under ``rewards``. If a
future extractor stores separate true and imagined reward arrays, the reward
prediction-error diagnostics are enabled automatically.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.diagnostics.freq_decompose import DETER_SLICE  # noqa: E402


DEFAULT_ROOT = REPO_ROOT / "results" / "v3_multiseed"
DEFAULT_SEEDS = [43, 44, 45]
DEFAULT_NUM_ROLLOUT_SEEDS = 5
DEFAULT_TRIM = (25, 175)
HORIZONS = [15, 30, 50, 100, 200]
SEGMENTS = {
    "0-5": (0, 5),
    "5-15": (5, 15),
    "15-30": (15, 30),
    "30-50": (30, 50),
    "50-100": (50, 100),
    "100-200": (100, 200),
}
PROBE_STEPS = [0, 1, 2, 5, 10, 20, 50, 100, 200]
REWARD_PAIR_KEYS = [
    ("true_rewards", "imagined_rewards"),
    ("true_reward", "imagined_reward"),
    ("rewards_true", "rewards_imagined"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose SHARP rollout fidelity.")
    parser.add_argument("--task", required=True, choices=["cheetah", "cartpole"])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--num_rollout_seeds", type=int, default=DEFAULT_NUM_ROLLOUT_SEEDS)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    args.root = resolve_path(args.root)
    if args.num_rollout_seeds <= 0:
        raise ValueError("--num_rollout_seeds must be positive.")
    if any(seed <= 0 for seed in args.seeds):
        raise ValueError("--seeds must be positive integers.")
    if args.output is None:
        args.output = (
            REPO_ROOT
            / "results"
            / "tables"
            / f"sharp_fidelity_diagnostics_{args.task}.json"
        )
    else:
        args.output = resolve_path(args.output)
    return args


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def relative_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def rollout_template(task: str) -> str:
    return f"{task}_v3_seed{{seed}}_v2.npz"


def rollout_dirs(root: Path, task: str, train_seed: int) -> tuple[Path, Path]:
    return (
        root / f"{task}_baseline_seed{train_seed}" / "rollouts",
        root / f"{task}_sharp_seed{train_seed}" / "rollouts",
    )


def matched_rollout_paths(
    baseline_dir: Path,
    sharp_dir: Path,
    *,
    template: str,
    num_rollout_seeds: int,
) -> tuple[list[int], list[Path], list[Path], list[int], list[int]]:
    matched: list[int] = []
    baseline_paths: list[Path] = []
    sharp_paths: list[Path] = []
    missing_baseline: list[int] = []
    missing_sharp: list[int] = []
    for rollout_seed in range(num_rollout_seeds):
        baseline_path = baseline_dir / template.format(seed=rollout_seed)
        sharp_path = sharp_dir / template.format(seed=rollout_seed)
        baseline_exists = baseline_path.exists()
        sharp_exists = sharp_path.exists()
        if baseline_exists and sharp_exists:
            matched.append(rollout_seed)
            baseline_paths.append(baseline_path)
            sharp_paths.append(sharp_path)
        if not baseline_exists:
            missing_baseline.append(rollout_seed)
        if not sharp_exists:
            missing_sharp.append(rollout_seed)
    return matched, baseline_paths, sharp_paths, missing_baseline, missing_sharp


def evaluate_seed(
    *,
    root: Path,
    task: str,
    train_seed: int,
    template: str,
    num_rollout_seeds: int,
) -> dict[str, Any]:
    baseline_dir, sharp_dir = rollout_dirs(root, task, train_seed)
    matched, baseline_paths, sharp_paths, missing_baseline, missing_sharp = (
        matched_rollout_paths(
            baseline_dir,
            sharp_dir,
            template=template,
            num_rollout_seeds=num_rollout_seeds,
        )
    )
    status = "complete" if len(matched) == num_rollout_seeds else "partial"
    if not matched:
        status = "missing"

    result: dict[str, Any] = {
        "seed": int(train_seed),
        "status": status,
        "included_in_summary": False,
        "matched_rollout_seeds": [int(seed) for seed in matched],
        "n_matched_rollouts": int(len(matched)),
        "baseline_rollout_dir": relative_path(baseline_dir),
        "sharp_rollout_dir": relative_path(sharp_dir),
        "missing_baseline_rollout_seeds": [int(seed) for seed in missing_baseline],
        "missing_sharp_rollout_seeds": [int(seed) for seed in missing_sharp],
        "errors": [],
        "diagnostics": None,
    }
    if not matched:
        return result

    try:
        baseline = [load_rollout(path) for path in baseline_paths]
        sharp = [load_rollout(path) for path in sharp_paths]
        diagnostics = {
            "scale_normalized_drift": compare_scale_drift(baseline, sharp),
            "reward_error_decomposition": compare_rewards(baseline, sharp),
            "horizon_decomposition": compare_horizon_segments(baseline, sharp),
            "dynamics_vs_head_proxy": compare_obs_probe_steps(baseline, sharp),
        }
        result["diagnostics"] = diagnostics
        result["included_in_summary"] = True
    except Exception as exc:
        result["status"] = "error"
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    return result


def load_rollout(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        required = ["true_latent", "imagined_latent", "true_obs", "imagined_obs", "rewards"]
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"{relative_path(path)} missing keys: {missing}")

        true_latent = latent_features(data["true_latent"])
        imagined_latent = latent_features(data["imagined_latent"])
        true_obs = as_time_features(data["true_obs"], "true_obs")
        imagined_obs = as_time_features(data["imagined_obs"], "imagined_obs")
        rewards = as_time_vector(data["rewards"], "rewards")
        reward_pair = load_reward_pair(data)

    validate_pair(path, true_latent, imagined_latent, "latent")
    validate_pair(path, true_obs, imagined_obs, "obs")
    if true_latent.shape[0] != true_obs.shape[0]:
        raise ValueError(
            f"{relative_path(path)} latent/obs time mismatch: "
            f"{true_latent.shape[0]} vs {true_obs.shape[0]}"
        )
    if rewards.shape[0] != true_obs.shape[0]:
        raise ValueError(
            f"{relative_path(path)} reward/obs time mismatch: "
            f"{rewards.shape[0]} vs {true_obs.shape[0]}"
        )

    if reward_pair is not None:
        true_rewards, imagined_rewards, keys = reward_pair
        if true_rewards.shape != imagined_rewards.shape:
            raise ValueError(
                f"{relative_path(path)} reward prediction shape mismatch: "
                f"{true_rewards.shape} vs {imagined_rewards.shape}"
            )
        if true_rewards.shape[0] != true_obs.shape[0]:
            raise ValueError(
                f"{relative_path(path)} reward prediction time mismatch: "
                f"{true_rewards.shape[0]} vs {true_obs.shape[0]}"
            )
    else:
        true_rewards = None
        imagined_rewards = None
        keys = None

    return {
        "path": path,
        "true_deter": true_latent,
        "imagined_deter": imagined_latent,
        "true_obs": true_obs,
        "imagined_obs": imagined_obs,
        "rewards": rewards,
        "true_rewards": true_rewards,
        "imagined_rewards": imagined_rewards,
        "reward_pair_keys": keys,
    }


def load_reward_pair(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    for true_key, imagined_key in REWARD_PAIR_KEYS:
        if true_key in data and imagined_key in data:
            return (
                as_time_vector(data[true_key], true_key),
                as_time_vector(data[imagined_key], imagined_key),
                [true_key, imagined_key],
            )
    return None


def validate_pair(path: Path, left: np.ndarray, right: np.ndarray, label: str) -> None:
    if left.shape != right.shape:
        raise ValueError(
            f"{relative_path(path)} {label} shape mismatch: {left.shape} vs {right.shape}"
        )


def latent_features(latent: np.ndarray) -> np.ndarray:
    values = as_time_features(latent, "latent")
    if values.shape[1] == 1536:
        return values[:, DETER_SLICE]
    if values.shape[1] == 512:
        return values
    raise ValueError(f"Expected latent dimension 1536 or 512, got {values.shape[1]}.")


def as_time_features(array: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    if values.ndim == 0:
        raise ValueError(f"{name} must have a time dimension.")
    if values.ndim == 1:
        values = values[:, None]
    elif values.ndim > 2:
        values = values.reshape(values.shape[0], -1)
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty, got shape {values.shape}.")
    return values


def as_time_vector(array: np.ndarray, name: str) -> np.ndarray:
    values = as_time_features(array, name)
    return np.mean(values, axis=1)


def compare_scale_drift(
    baseline: list[dict[str, Any]],
    sharp: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_summary = scale_condition_summary(baseline)
    sharp_summary = scale_condition_summary(sharp)
    return {
        "baseline": baseline_summary,
        "sharp": sharp_summary,
        "comparison": {
            "raw_J_total_change_pct": percent_change(
                baseline_summary["raw_J_total"]["mean"],
                sharp_summary["raw_J_total"]["mean"],
            ),
            "normalized_J_total_change_pct": percent_change(
                baseline_summary["normalized_J_total"]["mean"],
                sharp_summary["normalized_J_total"]["mean"],
            ),
            "true_deter_trace_change_pct": percent_change(
                baseline_summary["true_deter_spectrum"]["trace"]["mean"],
                sharp_summary["true_deter_spectrum"]["trace"]["mean"],
            ),
            "imagined_deter_trace_change_pct": percent_change(
                baseline_summary["imagined_deter_spectrum"]["trace"]["mean"],
                sharp_summary["imagined_deter_spectrum"]["trace"]["mean"],
            ),
            "true_effective_rank_delta": subtract(
                sharp_summary["true_deter_spectrum"]["effective_rank"]["mean"],
                baseline_summary["true_deter_spectrum"]["effective_rank"]["mean"],
            ),
            "imagined_effective_rank_delta": subtract(
                sharp_summary["imagined_deter_spectrum"]["effective_rank"]["mean"],
                baseline_summary["imagined_deter_spectrum"]["effective_rank"]["mean"],
            ),
            "true_effective_rank_change_pct": percent_change(
                baseline_summary["true_deter_spectrum"]["effective_rank"]["mean"],
                sharp_summary["true_deter_spectrum"]["effective_rank"]["mean"],
            ),
            "imagined_effective_rank_change_pct": percent_change(
                baseline_summary["imagined_deter_spectrum"]["effective_rank"]["mean"],
                sharp_summary["imagined_deter_spectrum"]["effective_rank"]["mean"],
            ),
        },
    }


def scale_condition_summary(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    raw_values = []
    normalized_values = []
    normalizers = []
    for rollout in rollouts:
        true_deter = rollout["true_deter"]
        imagined_deter = rollout["imagined_deter"]
        window = clipped_slice(*DEFAULT_TRIM, length=true_deter.shape[0])
        raw_j = float(np.mean((imagined_deter[window] - true_deter[window]) ** 2))
        normalizer = float(np.mean(np.var(true_deter, axis=0)))
        raw_values.append(raw_j)
        normalizers.append(normalizer)
        normalized_values.append(safe_ratio(raw_j, normalizer))

    return {
        "raw_J_total": summarize_values(raw_values),
        "normalizer_mean_var_t_true_deter": summarize_values(normalizers),
        "normalized_J_total": summarize_values(normalized_values),
        "true_deter_spectrum": covariance_spectrum_summary(
            [rollout["true_deter"] for rollout in rollouts]
        ),
        "imagined_deter_spectrum": covariance_spectrum_summary(
            [rollout["imagined_deter"] for rollout in rollouts]
        ),
    }


def covariance_spectrum_summary(trajectories: list[np.ndarray]) -> dict[str, Any]:
    spectra = [covariance_spectrum(trajectory) for trajectory in trajectories]
    top20 = np.stack([spec["top20_eigenvalues"] for spec in spectra], axis=0)
    return {
        "top20_eigenvalues_mean": np.mean(top20, axis=0),
        "top20_eigenvalues_std": np.std(top20, axis=0, ddof=0),
        "effective_rank": summarize_values([spec["effective_rank"] for spec in spectra]),
        "trace": summarize_values([spec["trace"] for spec in spectra]),
    }


def covariance_spectrum(trajectory: np.ndarray) -> dict[str, Any]:
    trajectory = as_time_features(trajectory, "trajectory")
    if trajectory.shape[0] < 2:
        eigenvalues = np.zeros(1, dtype=np.float64)
    else:
        centered = trajectory - np.mean(trajectory, axis=0, keepdims=True)
        singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
        eigenvalues = (singular_values**2) / max(trajectory.shape[0] - 1, 1)
    eigenvalues = np.sort(np.asarray(eigenvalues, dtype=np.float64))[::-1]
    max_eig = float(eigenvalues[0]) if eigenvalues.size else 0.0
    threshold = 0.01 * max_eig
    effective_rank = int(np.sum(eigenvalues > threshold)) if max_eig > 0 else 0
    top20 = np.zeros(20, dtype=np.float64)
    top20[: min(20, eigenvalues.size)] = eigenvalues[:20]
    return {
        "top20_eigenvalues": top20,
        "effective_rank": effective_rank,
        "trace": float(np.sum(eigenvalues)),
    }


def compare_rewards(
    baseline: list[dict[str, Any]],
    sharp: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_summary = reward_condition_summary(baseline)
    sharp_summary = reward_condition_summary(sharp)
    prediction_available = (
        baseline_summary["prediction_error_available"]
        and sharp_summary["prediction_error_available"]
    )
    return {
        "baseline": baseline_summary,
        "sharp": sharp_summary,
        "comparison": {
            "available_reward_distribution_change_pct": {
                "mean_reward": percent_change(
                    baseline_summary["available_rewards"]["mean_reward"]["mean"],
                    sharp_summary["available_rewards"]["mean_reward"]["mean"],
                ),
                "reward_variance": percent_change(
                    baseline_summary["available_rewards"]["reward_variance"]["mean"],
                    sharp_summary["available_rewards"]["reward_variance"]["mean"],
                ),
                "cumulative_reward_h200": percent_change(
                    baseline_summary["available_rewards"]["cumulative_by_horizon"]["200"]["mean"],
                    sharp_summary["available_rewards"]["cumulative_by_horizon"]["200"]["mean"],
                ),
            },
            "baseline_vs_sharp_realized_rank_spearman": rank_correlation_by_horizon(
                baseline,
                sharp,
                reward_key="rewards",
            ),
            "prediction_error_available": bool(prediction_available),
            "prediction_error_note": (
                None
                if prediction_available
                else "Separate true/imagined reward arrays are unavailable; "
                "reward prediction-error metrics are not computed."
            ),
        },
    }


def reward_condition_summary(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = [rollout["rewards"] for rollout in rollouts]
    prediction_rollouts = [
        rollout
        for rollout in rollouts
        if rollout["true_rewards"] is not None and rollout["imagined_rewards"] is not None
    ]
    prediction_available = len(prediction_rollouts) == len(rollouts)
    summary: dict[str, Any] = {
        "available_rewards": available_reward_summary(rewards),
        "prediction_error_available": bool(prediction_available),
        "prediction_error": None,
        "reward_pair_keys": (
            rollouts[0]["reward_pair_keys"] if prediction_available else None
        ),
    }
    if prediction_available:
        summary["prediction_error"] = reward_prediction_error_summary(prediction_rollouts)
    return summary


def available_reward_summary(rewards: list[np.ndarray]) -> dict[str, Any]:
    return {
        "mean_reward": summarize_values([float(np.mean(values)) for values in rewards]),
        "reward_variance": summarize_values([float(np.var(values)) for values in rewards]),
        "cumulative_by_horizon": {
            str(horizon): summarize_values(
                [float(np.sum(values[: min(horizon, values.shape[0])])) for values in rewards]
            )
            for horizon in HORIZONS
        },
    }


def reward_prediction_error_summary(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [
        rollout["imagined_rewards"] - rollout["true_rewards"]
        for rollout in rollouts
    ]
    return {
        "per_step_reward_mse": summarize_values(
            [float(np.mean(error**2)) for error in errors]
        ),
        "reward_bias": summarize_values([float(np.mean(error)) for error in errors]),
        "reward_error_variance": summarize_values(
            [float(np.var(error)) for error in errors]
        ),
        "reward_correlation_pearson": summarize_values(
            [
                safe_corr(rollout["imagined_rewards"], rollout["true_rewards"])
                for rollout in rollouts
            ]
        ),
        "cumulative_by_horizon": {
            str(horizon): {
                "true": summarize_values(
                    [
                        float(np.sum(rollout["true_rewards"][: min(horizon, rollout["true_rewards"].shape[0])]))
                        for rollout in rollouts
                    ]
                ),
                "imagined": summarize_values(
                    [
                        float(
                            np.sum(
                                rollout["imagined_rewards"][
                                    : min(horizon, rollout["imagined_rewards"].shape[0])
                                ]
                            )
                        )
                        for rollout in rollouts
                    ]
                ),
                "error": summarize_values(
                    [
                        float(np.sum(error[: min(horizon, error.shape[0])]))
                        for error in errors
                    ]
                ),
            }
            for horizon in HORIZONS
        },
    }


def rank_correlation_by_horizon(
    baseline: list[dict[str, Any]],
    sharp: list[dict[str, Any]],
    *,
    reward_key: str,
) -> dict[str, Any]:
    result = {}
    for horizon in HORIZONS:
        baseline_values = [
            float(np.sum(rollout[reward_key][: min(horizon, rollout[reward_key].shape[0])]))
            for rollout in baseline
        ]
        sharp_values = [
            float(np.sum(rollout[reward_key][: min(horizon, rollout[reward_key].shape[0])]))
            for rollout in sharp
        ]
        result[str(horizon)] = safe_spearman(baseline_values, sharp_values)
    return result


def compare_horizon_segments(
    baseline: list[dict[str, Any]],
    sharp: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_summary = horizon_condition_summary(baseline)
    sharp_summary = horizon_condition_summary(sharp)
    comparison = {}
    for segment in SEGMENTS:
        comparison[segment] = {
            "latent_mse_change_pct": percent_change(
                baseline_summary[segment]["latent_mse"]["mean"],
                sharp_summary[segment]["latent_mse"]["mean"],
            ),
            "obs_mse_change_pct": percent_change(
                baseline_summary[segment]["obs_mse"]["mean"],
                sharp_summary[segment]["obs_mse"]["mean"],
            ),
            "reward_mse_change_pct": percent_change(
                get_nested(baseline_summary, [segment, "reward_mse", "mean"]),
                get_nested(sharp_summary, [segment, "reward_mse", "mean"]),
            ),
        }
    return {
        "segments": {name: [int(start), int(end)] for name, (start, end) in SEGMENTS.items()},
        "baseline": baseline_summary,
        "sharp": sharp_summary,
        "comparison": comparison,
    }


def horizon_condition_summary(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for segment, (start, end) in SEGMENTS.items():
        latent_values = []
        obs_values = []
        reward_mse_values = []
        cumulative_reward_error_values = []
        for rollout in rollouts:
            latent_curve = mse_curve(rollout["true_deter"], rollout["imagined_deter"])
            obs_curve = mse_curve(rollout["true_obs"], rollout["imagined_obs"])
            segment_slice = clipped_slice(start, end, length=latent_curve.shape[0])
            obs_slice = clipped_slice(start, end, length=obs_curve.shape[0])
            if segment_slice.start < segment_slice.stop:
                latent_values.append(float(np.mean(latent_curve[segment_slice])))
            if obs_slice.start < obs_slice.stop:
                obs_values.append(float(np.mean(obs_curve[obs_slice])))
            if rollout["true_rewards"] is not None and rollout["imagined_rewards"] is not None:
                reward_error = rollout["imagined_rewards"] - rollout["true_rewards"]
                reward_slice = clipped_slice(start, end, length=reward_error.shape[0])
                if reward_slice.start < reward_slice.stop:
                    segment_error = reward_error[reward_slice]
                    reward_mse_values.append(float(np.mean(segment_error**2)))
                    cumulative_reward_error_values.append(float(np.sum(segment_error)))
        result[segment] = {
            "latent_mse": summarize_values(latent_values),
            "obs_mse": summarize_values(obs_values),
            "reward_mse": summarize_values(reward_mse_values),
            "cumulative_reward_error": summarize_values(cumulative_reward_error_values),
        }
    return result


def compare_obs_probe_steps(
    baseline: list[dict[str, Any]],
    sharp: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_summary = obs_probe_summary(baseline)
    sharp_summary = obs_probe_summary(sharp)
    ratios = {}
    for step in PROBE_STEPS:
        key = str(step)
        ratios[key] = safe_ratio(
            sharp_summary[key]["obs_mse"]["mean"],
            baseline_summary[key]["obs_mse"]["mean"],
        )
    return {
        "requested_steps": [int(step) for step in PROBE_STEPS],
        "baseline": baseline_summary,
        "sharp": sharp_summary,
        "sharp_over_baseline_ratio": ratios,
        "interpretation": (
            "Step-1 degradation suggests decoder or representation issues; "
            "late-only degradation suggests dynamics entering unfamiliar regions."
        ),
    }


def obs_probe_summary(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for requested_step in PROBE_STEPS:
        values = []
        used_indices = []
        for rollout in rollouts:
            true_obs = rollout["true_obs"]
            imagined_obs = rollout["imagined_obs"]
            used_index = min(requested_step, true_obs.shape[0] - 1)
            used_indices.append(int(used_index))
            values.append(float(np.mean((imagined_obs[used_index] - true_obs[used_index]) ** 2)))
        result[str(requested_step)] = {
            "requested_step": int(requested_step),
            "used_indices": used_indices,
            "obs_mse": summarize_values(values),
        }
    return result


def mse_curve(true_signal: np.ndarray, imagined_signal: np.ndarray) -> np.ndarray:
    return np.mean((imagined_signal - true_signal) ** 2, axis=1)


def clipped_slice(start: int, end: int, *, length: int) -> slice:
    return slice(min(start, length), min(end, length))


def build_cross_seed_summary(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    included = [row for row in seed_results if row.get("included_in_summary")]
    summary: dict[str, Any] = {
        "n_included": int(len(included)),
        "seeds": [int(row["seed"]) for row in included],
        "scale_normalized_drift": {
            "raw_J_total_change_pct": summarize_nested(
                included,
                ["diagnostics", "scale_normalized_drift", "comparison", "raw_J_total_change_pct"],
            ),
            "normalized_J_total_change_pct": summarize_nested(
                included,
                [
                    "diagnostics",
                    "scale_normalized_drift",
                    "comparison",
                    "normalized_J_total_change_pct",
                ],
            ),
            "imagined_deter_trace_change_pct": summarize_nested(
                included,
                [
                    "diagnostics",
                    "scale_normalized_drift",
                    "comparison",
                    "imagined_deter_trace_change_pct",
                ],
            ),
        },
        "reward_error_decomposition": {
            "reward_prediction_error_available_all_seeds": bool(
                included
                and all(
                    get_nested(
                        row,
                        [
                            "diagnostics",
                            "reward_error_decomposition",
                            "comparison",
                            "prediction_error_available",
                        ],
                    )
                    for row in included
                )
            ),
            "cumulative_reward_h200_change_pct": summarize_nested(
                included,
                [
                    "diagnostics",
                    "reward_error_decomposition",
                    "comparison",
                    "available_reward_distribution_change_pct",
                    "cumulative_reward_h200",
                ],
            ),
            "reward_variance_change_pct": summarize_nested(
                included,
                [
                    "diagnostics",
                    "reward_error_decomposition",
                    "comparison",
                    "available_reward_distribution_change_pct",
                    "reward_variance",
                ],
            ),
            "spearman_h200": summarize_nested(
                included,
                [
                    "diagnostics",
                    "reward_error_decomposition",
                    "comparison",
                    "baseline_vs_sharp_realized_rank_spearman",
                    "200",
                ],
            ),
        },
        "horizon_decomposition": {
            segment: {
                "latent_mse_change_pct": summarize_nested(
                    included,
                    [
                        "diagnostics",
                        "horizon_decomposition",
                        "comparison",
                        segment,
                        "latent_mse_change_pct",
                    ],
                ),
                "obs_mse_change_pct": summarize_nested(
                    included,
                    [
                        "diagnostics",
                        "horizon_decomposition",
                        "comparison",
                        segment,
                        "obs_mse_change_pct",
                    ],
                ),
            }
            for segment in SEGMENTS
        },
        "dynamics_vs_head_proxy": {
            str(step): {
                "sharp_over_baseline_ratio": summarize_nested(
                    included,
                    [
                        "diagnostics",
                        "dynamics_vs_head_proxy",
                        "sharp_over_baseline_ratio",
                        str(step),
                    ],
                )
            }
            for step in PROBE_STEPS
        },
    }
    return summary


def build_verdict(cross_summary: dict[str, Any]) -> dict[str, Any]:
    scale = cross_summary["scale_normalized_drift"]
    normalized_change = scale["normalized_J_total_change_pct"]["mean"]
    imagined_trace_change = scale["imagined_deter_trace_change_pct"]["mean"]
    reward = cross_summary["reward_error_decomposition"]
    horizon = cross_summary["horizon_decomposition"]
    proxy = cross_summary["dynamics_vs_head_proxy"]

    scale_findings = []
    if normalized_change is None:
        scale_findings.append("No completed seed pairs for normalized drift verdict.")
    elif normalized_change <= -50.0:
        scale_findings.append(
            "Normalized J_total drops by more than 50%, supporting real drift reduction."
        )
    else:
        scale_findings.append(
            "Normalized J_total does not drop by more than 50%; scale artifacts remain possible."
        )
    if imagined_trace_change is not None and imagined_trace_change <= -50.0:
        scale_findings.append(
            "Imagined deter covariance trace drops by more than 50%, flagging latent compression risk."
        )

    if reward["reward_prediction_error_available_all_seeds"]:
        reward_text = "Separate reward prediction arrays are available for all included seeds."
    else:
        reward_text = (
            "Separate true/imagined reward arrays are unavailable for at least one included seed; "
            "reward prediction error is not claimed from current rollouts."
        )

    early_segments = ["0-5", "5-15"]
    late_segments = ["50-100", "100-200"]
    early_obs = [
        horizon[segment]["obs_mse_change_pct"]["mean"]
        for segment in early_segments
        if horizon[segment]["obs_mse_change_pct"]["mean"] is not None
    ]
    late_obs = [
        horizon[segment]["obs_mse_change_pct"]["mean"]
        for segment in late_segments
        if horizon[segment]["obs_mse_change_pct"]["mean"] is not None
    ]
    early_good = bool(early_obs) and max(early_obs) <= 25.0
    late_bad = bool(late_obs) and max(late_obs) >= 50.0
    if early_good and late_bad:
        horizon_text = (
            "Early [0,15) obs errors are mild while late [50,200) errors worsen, "
            "suggesting lower actor-horizon risk but higher long-horizon fidelity risk."
        )
    elif not early_obs:
        horizon_text = "No completed seed pairs for horizon verdict."
    else:
        horizon_text = "Horizon pattern is mixed; inspect per-seed segment tables."

    step1 = get_nested(proxy, ["1", "sharp_over_baseline_ratio", "mean"])
    later_values = [
        get_nested(proxy, [str(step), "sharp_over_baseline_ratio", "mean"])
        for step in [2, 5, 10, 20, 50, 100, 200]
    ]
    later_values = [value for value in later_values if value is not None]
    max_later = max(later_values) if later_values else None
    if step1 is not None and step1 > 2.0:
        proxy_text = (
            "Step-1 obs error ratio exceeds 2, flagging decoder or representation concern."
        )
    elif step1 is not None and step1 <= 1.5 and max_later is not None and max_later > 2.0:
        proxy_text = (
            "Step-1 obs error is mild but later obs error exceeds 2x, suggesting dynamics "
            "or unfamiliar-region growth."
        )
    elif step1 is None:
        proxy_text = "No completed seed pairs for dynamics-vs-head proxy verdict."
    else:
        proxy_text = "Dynamics-vs-head proxy is mixed; inspect horizon ratios."

    return {
        "scale_normalized_drift": " ".join(scale_findings),
        "reward_error_decomposition": reward_text,
        "horizon_decomposition": horizon_text,
        "dynamics_vs_head_proxy": proxy_text,
    }


def summarize_nested(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    values = []
    for row in rows:
        value = get_nested(row, keys)
        if value is not None:
            values.append(value)
    return summarize_values(values)


def get_nested(data: dict[str, Any] | None, keys: list[str]) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def summarize_values(values: list[float | None]) -> dict[str, Any]:
    clean = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not clean:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    array = np.asarray(clean, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=0)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def percent_change(baseline: float | None, sharp: float | None) -> float | None:
    if baseline is None or sharp is None or float(baseline) == 0.0:
        return None
    return (float(sharp) - float(baseline)) / float(baseline) * 100.0


def subtract(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or float(denominator) == 0.0:
        return None
    return float(numerator) / float(denominator)


def safe_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size < 2:
        return None
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def safe_spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or len(y) < 2:
        return None
    return safe_corr(rankdata_average(np.asarray(x)), rankdata_average(np.asarray(y)))


def rankdata_average(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + end - 1) / 2.0 + 1.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def print_report(result: dict[str, Any]) -> None:
    title = f"SHARP Fidelity Diagnostics: {result['task'].title()}"
    print(f"\n{title}")
    print("=" * len(title))
    print(f"Root: {result['root']}")
    print(f"Rollout template: {result['rollout_template']}")
    print(f"Training seeds: {result['training_seeds']}")
    print("Percent change is SHARP relative to baseline; negative means lower.")

    for row in result["per_seed"]:
        print(f"\nSeed {row['seed']} [{row['status']}], matched rollouts: {row['n_matched_rollouts']}")
        if row.get("errors"):
            print(f"  Errors: {'; '.join(row['errors'])}")
        if not row.get("diagnostics"):
            continue
        print_scale_table(row["diagnostics"]["scale_normalized_drift"])
        print_reward_tables(row["diagnostics"]["reward_error_decomposition"])
        print_horizon_table(row["diagnostics"]["horizon_decomposition"])
        print_proxy_table(row["diagnostics"]["dynamics_vs_head_proxy"])

    print("\nCross-Seed Summary")
    print_cross_seed_summary(result["cross_seed_summary"])
    print("\nVerdict")
    for key, value in result["verdict"].items():
        print(f"- {key}: {value}")
    print(f"\nSaved JSON: {result['output_path']}")


def print_scale_table(scale: dict[str, Any]) -> None:
    baseline = scale["baseline"]
    sharp = scale["sharp"]
    comparison = scale["comparison"]
    print("\nDiagnostic 1: Scale-Normalized Drift")
    print_table(
        ["Metric", "Baseline", "SHARP", "Change"],
        [
            [
                "Raw J_total",
                fmt_num(baseline["raw_J_total"]["mean"]),
                fmt_num(sharp["raw_J_total"]["mean"]),
                fmt_pct(comparison["raw_J_total_change_pct"]),
            ],
            [
                "Normalized J_total",
                fmt_num(baseline["normalized_J_total"]["mean"]),
                fmt_num(sharp["normalized_J_total"]["mean"]),
                fmt_pct(comparison["normalized_J_total_change_pct"]),
            ],
            [
                "True deter trace",
                fmt_num(baseline["true_deter_spectrum"]["trace"]["mean"]),
                fmt_num(sharp["true_deter_spectrum"]["trace"]["mean"]),
                fmt_pct(comparison["true_deter_trace_change_pct"]),
            ],
            [
                "Imag deter trace",
                fmt_num(baseline["imagined_deter_spectrum"]["trace"]["mean"]),
                fmt_num(sharp["imagined_deter_spectrum"]["trace"]["mean"]),
                fmt_pct(comparison["imagined_deter_trace_change_pct"]),
            ],
            [
                "True effective rank",
                fmt_num(baseline["true_deter_spectrum"]["effective_rank"]["mean"]),
                fmt_num(sharp["true_deter_spectrum"]["effective_rank"]["mean"]),
                fmt_signed_num(comparison["true_effective_rank_delta"]),
            ],
            [
                "Imag effective rank",
                fmt_num(baseline["imagined_deter_spectrum"]["effective_rank"]["mean"]),
                fmt_num(sharp["imagined_deter_spectrum"]["effective_rank"]["mean"]),
                fmt_signed_num(comparison["imagined_effective_rank_delta"]),
            ],
        ],
    )
    print("Top-20 covariance eigenvalues are saved in JSON.")


def print_reward_tables(reward: dict[str, Any]) -> None:
    baseline = reward["baseline"]
    sharp = reward["sharp"]
    comparison = reward["comparison"]
    print("\nDiagnostic 2: Reward Error Decomposition")
    print_table(
        ["Condition", "Mean Reward", "Reward Var", "Cum@15", "Cum@30", "Cum@50", "Cum@100", "Cum@200"],
        [
            reward_distribution_row("Baseline", baseline),
            reward_distribution_row("SHARP", sharp),
        ],
    )
    print_table(
        ["Metric", "Baseline", "SHARP", "Change"],
        [
            [
                "Mean reward",
                fmt_num(baseline["available_rewards"]["mean_reward"]["mean"]),
                fmt_num(sharp["available_rewards"]["mean_reward"]["mean"]),
                fmt_pct(comparison["available_reward_distribution_change_pct"]["mean_reward"]),
            ],
            [
                "Reward variance",
                fmt_num(baseline["available_rewards"]["reward_variance"]["mean"]),
                fmt_num(sharp["available_rewards"]["reward_variance"]["mean"]),
                fmt_pct(comparison["available_reward_distribution_change_pct"]["reward_variance"]),
            ],
            [
                "Cumulative reward @200",
                fmt_num(baseline["available_rewards"]["cumulative_by_horizon"]["200"]["mean"]),
                fmt_num(sharp["available_rewards"]["cumulative_by_horizon"]["200"]["mean"]),
                fmt_pct(
                    comparison["available_reward_distribution_change_pct"][
                        "cumulative_reward_h200"
                    ]
                ),
            ],
        ],
    )
    print("Reward prediction error:")
    if comparison["prediction_error_available"]:
        print_table(
            ["Condition", "Reward MSE", "Bias", "Error Var", "Pearson"],
            [
                reward_prediction_row("Baseline", baseline),
                reward_prediction_row("SHARP", sharp),
            ],
        )
    else:
        print(f"  N/A - {comparison['prediction_error_note']}")
    print_table(
        ["Horizon", "Baseline-vs-SHARP realized Spearman"],
        [
            [horizon, fmt_num(comparison["baseline_vs_sharp_realized_rank_spearman"][str(horizon)])]
            for horizon in HORIZONS
        ],
    )


def reward_distribution_row(label: str, summary: dict[str, Any]) -> list[Any]:
    available = summary["available_rewards"]
    return [
        label,
        fmt_num(available["mean_reward"]["mean"]),
        fmt_num(available["reward_variance"]["mean"]),
        *[
            fmt_num(available["cumulative_by_horizon"][str(horizon)]["mean"])
            for horizon in HORIZONS
        ],
    ]


def reward_prediction_row(label: str, summary: dict[str, Any]) -> list[Any]:
    prediction = summary["prediction_error"]
    return [
        label,
        fmt_num(prediction["per_step_reward_mse"]["mean"]),
        fmt_num(prediction["reward_bias"]["mean"]),
        fmt_num(prediction["reward_error_variance"]["mean"]),
        fmt_num(prediction["reward_correlation_pearson"]["mean"]),
    ]


def print_horizon_table(horizon: dict[str, Any]) -> None:
    print("\nDiagnostic 3: Horizon Decomposition")
    rows = []
    for segment in SEGMENTS:
        base = horizon["baseline"][segment]
        sharp = horizon["sharp"][segment]
        comp = horizon["comparison"][segment]
        rows.append(
            [
                segment,
                fmt_num(base["latent_mse"]["mean"]),
                fmt_num(sharp["latent_mse"]["mean"]),
                fmt_pct(comp["latent_mse_change_pct"]),
                fmt_num(base["obs_mse"]["mean"]),
                fmt_num(sharp["obs_mse"]["mean"]),
                fmt_pct(comp["obs_mse_change_pct"]),
                fmt_num(base["reward_mse"]["mean"]),
                fmt_num(sharp["reward_mse"]["mean"]),
                fmt_num(base["cumulative_reward_error"]["mean"]),
                fmt_num(sharp["cumulative_reward_error"]["mean"]),
            ]
        )
    print_table(
        [
            "Segment",
            "Base Lat",
            "SHARP Lat",
            "Lat Δ",
            "Base Obs",
            "SHARP Obs",
            "Obs Δ",
            "Base R-MSE",
            "SHARP R-MSE",
            "Base Cum R Err",
            "SHARP Cum R Err",
        ],
        rows,
    )


def print_proxy_table(proxy: dict[str, Any]) -> None:
    print("\nDiagnostic 4: Dynamics vs Head Attribution Proxy")
    print_table(
        ["Step", "Used Idx", "Base Obs MSE", "SHARP Obs MSE", "SHARP/Base"],
        [
            [
                step,
                proxy["baseline"][str(step)]["used_indices"][0]
                if proxy["baseline"][str(step)]["used_indices"]
                else "N/A",
                fmt_num(proxy["baseline"][str(step)]["obs_mse"]["mean"]),
                fmt_num(proxy["sharp"][str(step)]["obs_mse"]["mean"]),
                fmt_num(proxy["sharp_over_baseline_ratio"][str(step)]),
            ]
            for step in PROBE_STEPS
        ],
    )


def print_cross_seed_summary(summary: dict[str, Any]) -> None:
    print(f"Included seeds: {summary['seeds']} (n={summary['n_included']})")
    scale = summary["scale_normalized_drift"]
    print_table(
        ["Scale Metric", "Mean Change", "Std", "N"],
        [
            summary_pct_row("Raw J_total", scale["raw_J_total_change_pct"]),
            summary_pct_row("Normalized J_total", scale["normalized_J_total_change_pct"]),
            summary_pct_row("Imag trace", scale["imagined_deter_trace_change_pct"]),
        ],
    )
    reward = summary["reward_error_decomposition"]
    print_table(
        ["Reward Metric", "Mean Change", "Std", "N"],
        [
            summary_pct_row("Cum reward @200", reward["cumulative_reward_h200_change_pct"]),
            summary_pct_row("Reward variance", reward["reward_variance_change_pct"]),
            summary_num_row("Spearman @200", reward["spearman_h200"]),
        ],
    )
    print_table(
        ["Segment", "Latent MSE Δ", "Obs MSE Δ"],
        [
            [
                segment,
                fmt_summary_pct(summary["horizon_decomposition"][segment]["latent_mse_change_pct"]),
                fmt_summary_pct(summary["horizon_decomposition"][segment]["obs_mse_change_pct"]),
            ]
            for segment in SEGMENTS
        ],
    )
    print_table(
        ["Probe Step", "Mean SHARP/Base Obs Ratio", "Std", "N"],
        [
            summary_num_row(
                step,
                summary["dynamics_vs_head_proxy"][str(step)]["sharp_over_baseline_ratio"],
            )
            for step in PROBE_STEPS
        ],
    )


def print_table(headers: list[Any], rows: list[list[Any]]) -> None:
    string_headers = [str(header) for header in headers]
    string_rows = [[str(cell) for cell in row] for row in rows]
    widths = [
        max(len(string_headers[idx]), *(len(row[idx]) for row in string_rows))
        if string_rows
        else len(string_headers[idx])
        for idx in range(len(string_headers))
    ]
    print("  ".join(header.ljust(widths[idx]) for idx, header in enumerate(string_headers)))
    print("  ".join("-" * width for width in widths))
    for row in string_rows:
        print("  ".join(row[idx].rjust(widths[idx]) for idx in range(len(widths))))


def fmt_num(value: Any) -> str:
    if value is None:
        return "N/A"
    value = float(value)
    if value == 0.0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:.3e}"
    return f"{value:.5f}"


def fmt_signed_num(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):+.3g}"


def fmt_pct(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):+.1f}%"


def fmt_summary_pct(summary: dict[str, Any] | None) -> str:
    if not summary or summary.get("n", 0) == 0:
        return "N/A"
    return f"{summary['mean']:+.1f}% +/- {summary['std']:.1f}%"


def summary_pct_row(label: str, summary: dict[str, Any] | None) -> list[Any]:
    if not summary or summary.get("n", 0) == 0:
        return [label, "N/A", "N/A", 0]
    return [label, fmt_pct(summary["mean"]), f"{float(summary['std']):.1f}%", int(summary["n"])]


def summary_num_row(label: Any, summary: dict[str, Any] | None) -> list[Any]:
    if not summary or summary.get("n", 0) == 0:
        return [label, "N/A", "N/A", 0]
    return [label, fmt_num(summary["mean"]), fmt_num(summary["std"]), int(summary["n"])]


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return relative_path(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    template = rollout_template(args.task)
    per_seed = [
        evaluate_seed(
            root=args.root,
            task=args.task,
            train_seed=int(seed),
            template=template,
            num_rollout_seeds=int(args.num_rollout_seeds),
        )
        for seed in args.seeds
    ]
    cross_seed_summary = build_cross_seed_summary(per_seed)
    result = {
        "task": args.task,
        "root": relative_path(args.root),
        "training_seeds": [int(seed) for seed in args.seeds],
        "num_rollout_seeds": int(args.num_rollout_seeds),
        "rollout_template": template,
        "metric_notes": {
            "latent_slice": "Use deter slice 1024:1536 for 1536-dim latents; use all dims for 512.",
            "trim_window_half_open": [DEFAULT_TRIM[0], DEFAULT_TRIM[1]],
            "percent_change": "SHARP relative to baseline; negative means lower.",
            "reward_handling": (
                "Current rollouts contain realized environment rewards only. "
                "Prediction-error metrics require separate true/imagined reward arrays."
            ),
        },
        "per_seed": per_seed,
        "cross_seed_summary": cross_seed_summary,
        "verdict": build_verdict(cross_seed_summary),
        "output_path": relative_path(args.output),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(to_jsonable(result), indent=2) + "\n")
    print_report(result)


if __name__ == "__main__":
    main()
