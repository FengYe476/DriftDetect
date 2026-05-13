#!/usr/bin/env python3
"""Analyze how SHARP drift reduction relates to return-side behavior.

This script compares same-task V3 multi-seed baseline and SHARP rollout pairs.
It focuses on diagnostics that can explain a small eval_return change despite
large latent drift reductions: per-band error changes, observation-space error,
reward statistics, latent variance, and early/mid/late horizon behavior.
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

from src.diagnostics.error_curves import aggregate_curves  # noqa: E402
from src.diagnostics.freq_decompose import DEFAULT_BANDS  # noqa: E402


BAND_ORDER = ["dc_trend", "very_low", "low", "mid", "high"]
DEFAULT_SEEDS = [43, 44, 45]
DEFAULT_NUM_ROLLOUT_SEEDS = 5
DEFAULT_ROOT = REPO_ROOT / "results" / "v3_multiseed"
DETER_SLICE = slice(1024, 1536)
VAR_THRESHOLD = 0.95
FILTER_TYPE = "butterworth"
SEGMENTS = {
    "early": (0, 50),
    "mid": (50, 100),
    "late": (100, 200),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze V3 SHARP return impact for Cheetah or Cartpole."
    )
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
            / f"sharp_return_impact_analysis_{args.task}.json"
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
        "baseline_rollout_dir": relative_path(baseline_dir),
        "sharp_rollout_dir": relative_path(sharp_dir),
        "missing_baseline_rollout_seeds": [int(seed) for seed in missing_baseline],
        "missing_sharp_rollout_seeds": [int(seed) for seed in missing_sharp],
        "n_matched_rollouts": int(len(matched)),
        "errors": [],
        "per_band": None,
        "obs_space": None,
        "rewards": None,
        "latent_variance": None,
        "horizon_segments": None,
    }
    if not matched:
        return result

    try:
        result["per_band"] = compare_per_band(baseline_paths, sharp_paths)
        result["obs_space"] = compare_obs_space(baseline_paths, sharp_paths)
        result["rewards"] = compare_rewards(baseline_paths, sharp_paths)
        result["latent_variance"] = compare_latent_variance(baseline_paths, sharp_paths)
        result["horizon_segments"] = compare_horizon_segments(
            baseline_paths,
            sharp_paths,
        )
        result["included_in_summary"] = True
    except Exception as exc:  # keep one bad seed from hiding other partial results
        result["status"] = "error"
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    return result


def compare_per_band(
    baseline_paths: list[Path],
    sharp_paths: list[Path],
) -> dict[str, Any]:
    baseline = frequency_band_mse(baseline_paths)
    sharp = frequency_band_mse(sharp_paths)
    return {
        "baseline": baseline,
        "sharp": sharp,
        "reduction_pct": {
            band: percent_reduction(baseline["per_band_mse"][band], sharp["per_band_mse"][band])
            for band in BAND_ORDER
        },
    }


def frequency_band_mse(paths: list[Path]) -> dict[str, Any]:
    curves = aggregate_curves(
        paths,
        signal_key_true="true_latent",
        signal_key_imag="imagined_latent",
        metric="mse",
        bands=DEFAULT_BANDS,
        var_threshold=VAR_THRESHOLD,
        filter_type=FILTER_TYPE,
    )
    per_band = {
        band: float(np.mean(np.asarray(curves[band]["mean"], dtype=np.float64)))
        for band in BAND_ORDER
    }
    total = float(sum(per_band.values()))
    info = curves["_info"]["decompose_info"]
    return {
        "J_total": total,
        "per_band_mse": per_band,
        "band_share": {
            band: safe_divide(value, total)
            for band, value in per_band.items()
        },
        "pca": {
            "n_components": int(info["n_pcs"]),
            "var_explained": float(info["var_explained"]),
        },
        "n_rollouts": int(len(paths)),
    }


def compare_obs_space(
    baseline_paths: list[Path],
    sharp_paths: list[Path],
) -> dict[str, Any]:
    baseline_values = [mean_pair_mse(path, "true_obs", "imagined_obs") for path in baseline_paths]
    sharp_values = [mean_pair_mse(path, "true_obs", "imagined_obs") for path in sharp_paths]
    baseline = summarize_values(baseline_values)
    sharp = summarize_values(sharp_values)
    return {
        "baseline_mse": baseline,
        "sharp_mse": sharp,
        "reduction_pct": percent_reduction(baseline["mean"], sharp["mean"]),
    }


def compare_rewards(
    baseline_paths: list[Path],
    sharp_paths: list[Path],
) -> dict[str, Any]:
    baseline = reward_summary(baseline_paths)
    sharp = reward_summary(sharp_paths)
    return {
        "baseline": baseline,
        "sharp": sharp,
        "change_pct": {
            "mean_reward": percent_change(
                baseline["mean_reward"]["mean"],
                sharp["mean_reward"]["mean"],
            ),
            "reward_variance": percent_change(
                baseline["reward_variance"]["mean"],
                sharp["reward_variance"]["mean"],
            ),
            "cumulative_reward": percent_change(
                baseline["cumulative_reward"]["mean"],
                sharp["cumulative_reward"]["mean"],
            ),
        },
    }


def reward_summary(paths: list[Path]) -> dict[str, Any]:
    mean_rewards = []
    reward_variances = []
    cumulative_rewards = []
    for path in paths:
        rewards = load_time_features(path, "rewards")
        per_step_reward = np.mean(rewards, axis=1)
        mean_rewards.append(float(np.mean(per_step_reward)))
        reward_variances.append(float(np.var(per_step_reward)))
        cumulative_rewards.append(float(np.sum(per_step_reward)))
    return {
        "mean_reward": summarize_values(mean_rewards),
        "reward_variance": summarize_values(reward_variances),
        "cumulative_reward": summarize_values(cumulative_rewards),
    }


def compare_latent_variance(
    baseline_paths: list[Path],
    sharp_paths: list[Path],
) -> dict[str, Any]:
    baseline = latent_variance_summary(baseline_paths)
    sharp = latent_variance_summary(sharp_paths)
    return {
        "baseline": baseline,
        "sharp": sharp,
        "change_pct": {
            "inter_seed_variance": percent_change(
                baseline["inter_seed_variance"],
                sharp["inter_seed_variance"],
            ),
            "intra_trajectory_variance": percent_change(
                baseline["intra_trajectory_variance"]["mean"],
                sharp["intra_trajectory_variance"]["mean"],
            ),
        },
        "note": (
            "For 1536-dim Dreamer features, variance is computed on the "
            "deterministic 512-dim slice; 512-dim inputs are used directly."
        ),
    }


def latent_variance_summary(paths: list[Path]) -> dict[str, Any]:
    latents = [latent_features(load_time_features(path, "imagined_latent")) for path in paths]
    reference_shape = latents[0].shape
    for path, latent in zip(paths, latents):
        if latent.shape != reference_shape:
            raise ValueError(
                f"imagined_latent shape mismatch: expected {reference_shape}, "
                f"got {latent.shape} in {relative_path(path)}"
            )
    stack = np.stack(latents, axis=0)
    inter_seed_variance = float(np.mean(np.var(stack, axis=0))) if len(latents) > 1 else 0.0
    intra_values = [float(np.mean(np.var(latent, axis=0))) for latent in latents]
    return {
        "n_rollouts": int(len(latents)),
        "time_steps": int(reference_shape[0]),
        "latent_dim_used": int(reference_shape[1]),
        "inter_seed_variance": inter_seed_variance,
        "intra_trajectory_variance": summarize_values(intra_values),
    }


def compare_horizon_segments(
    baseline_paths: list[Path],
    sharp_paths: list[Path],
) -> dict[str, Any]:
    baseline = segment_mse_summary(baseline_paths)
    sharp = segment_mse_summary(sharp_paths)
    return {
        "baseline": baseline,
        "sharp": sharp,
        "reduction_pct": {
            segment: percent_reduction(
                baseline[segment]["mean"],
                sharp[segment]["mean"],
            )
            for segment in SEGMENTS
        },
        "segments": {name: [int(start), int(end)] for name, (start, end) in SEGMENTS.items()},
        "metric": "latent deterministic MSE",
    }


def segment_mse_summary(paths: list[Path]) -> dict[str, Any]:
    values: dict[str, list[float]] = {segment: [] for segment in SEGMENTS}
    for path in paths:
        per_step = pair_mse_curve(path, "true_latent", "imagined_latent", use_latent=True)
        for segment, (start, end) in SEGMENTS.items():
            clipped_start = min(start, per_step.shape[0])
            clipped_end = min(end, per_step.shape[0])
            if clipped_start < clipped_end:
                values[segment].append(float(np.mean(per_step[clipped_start:clipped_end])))
    return {segment: summarize_values(segment_values) for segment, segment_values in values.items()}


def mean_pair_mse(path: Path, true_key: str, imagined_key: str) -> float:
    return float(np.mean(pair_mse_curve(path, true_key, imagined_key, use_latent=False)))


def pair_mse_curve(
    path: Path,
    true_key: str,
    imagined_key: str,
    *,
    use_latent: bool,
) -> np.ndarray:
    with np.load(path, allow_pickle=True) as data:
        if true_key not in data:
            raise KeyError(f"{relative_path(path)} does not contain {true_key!r}.")
        if imagined_key not in data:
            raise KeyError(f"{relative_path(path)} does not contain {imagined_key!r}.")
        true_signal = as_time_features(data[true_key], true_key)
        imagined_signal = as_time_features(data[imagined_key], imagined_key)
    if true_signal.shape != imagined_signal.shape:
        raise ValueError(
            f"{relative_path(path)} shape mismatch for {true_key}/{imagined_key}: "
            f"{true_signal.shape} vs {imagined_signal.shape}"
        )
    if use_latent:
        true_signal = latent_features(true_signal)
        imagined_signal = latent_features(imagined_signal)
    return np.mean((imagined_signal - true_signal) ** 2, axis=1)


def load_time_features(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=True) as data:
        if key not in data:
            raise KeyError(f"{relative_path(path)} does not contain {key!r}.")
        return as_time_features(data[key], key)


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


def latent_features(latent: np.ndarray) -> np.ndarray:
    latent = as_time_features(latent, "latent")
    if latent.shape[1] == 1536:
        return latent[:, DETER_SLICE]
    if latent.shape[1] == 512:
        return latent
    raise ValueError(f"Expected latent dimension 1536 or 512, got {latent.shape[1]}.")


def build_cross_seed_summary(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    included = [row for row in seed_results if row.get("included_in_summary")]
    summary: dict[str, Any] = {
        "n_included": int(len(included)),
        "seeds": [int(row["seed"]) for row in included],
    }
    summary["per_band_reduction_pct"] = {
        band: summarize_nested(included, ["per_band", "reduction_pct", band])
        for band in BAND_ORDER
    }
    summary["obs_mse_reduction_pct"] = summarize_nested(
        included,
        ["obs_space", "reduction_pct"],
    )
    summary["reward_change_pct"] = {
        key: summarize_nested(included, ["rewards", "change_pct", key])
        for key in ["mean_reward", "reward_variance", "cumulative_reward"]
    }
    summary["latent_variance_change_pct"] = {
        key: summarize_nested(included, ["latent_variance", "change_pct", key])
        for key in ["inter_seed_variance", "intra_trajectory_variance"]
    }
    summary["horizon_reduction_pct"] = {
        segment: summarize_nested(included, ["horizon_segments", "reduction_pct", segment])
        for segment in SEGMENTS
    }
    return summary


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


def percent_reduction(baseline: float | None, sharp: float | None) -> float | None:
    if baseline is None or sharp is None or float(baseline) == 0.0:
        return None
    return (float(baseline) - float(sharp)) / float(baseline) * 100.0


def percent_change(baseline: float | None, sharp: float | None) -> float | None:
    if baseline is None or sharp is None or float(baseline) == 0.0:
        return None
    return (float(sharp) - float(baseline)) / float(baseline) * 100.0


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def print_report(result: dict[str, Any]) -> None:
    title = f"SHARP Return Impact Analysis: {result['task'].title()}"
    print(f"\n{title}")
    print("=" * len(title))
    print(f"Root: {result['root']}")
    print(f"Rollout template: {result['rollout_template']}")
    print(f"Training seeds: {result['training_seeds']}")
    print(f"Matched rollout seeds requested: 0..{result['num_rollout_seeds'] - 1}")

    print("\n1. Per-Band Drift Reduction vs Baseline")
    print_table(
        ["Seed", "N", *BAND_ORDER],
        [
            [
                row["seed"],
                row["n_matched_rollouts"],
                *[
                    fmt_pct(get_nested(row, ["per_band", "reduction_pct", band]))
                    for band in BAND_ORDER
                ],
            ]
            for row in result["per_seed"]
        ],
    )

    print("\n2. Observation-Space Error")
    print_table(
        ["Seed", "N", "Base MSE", "SHARP MSE", "Reduction"],
        [
            [
                row["seed"],
                row["n_matched_rollouts"],
                fmt_num(get_nested(row, ["obs_space", "baseline_mse", "mean"])),
                fmt_num(get_nested(row, ["obs_space", "sharp_mse", "mean"])),
                fmt_pct(get_nested(row, ["obs_space", "reduction_pct"])),
            ]
            for row in result["per_seed"]
        ],
    )

    print("\n3. Reward Statistics")
    print_table(
        ["Seed", "Base Mean", "SHARP Mean", "Mean Δ", "Base Cum", "SHARP Cum", "Cum Δ"],
        [
            [
                row["seed"],
                fmt_num(get_nested(row, ["rewards", "baseline", "mean_reward", "mean"])),
                fmt_num(get_nested(row, ["rewards", "sharp", "mean_reward", "mean"])),
                fmt_pct(get_nested(row, ["rewards", "change_pct", "mean_reward"])),
                fmt_num(get_nested(row, ["rewards", "baseline", "cumulative_reward", "mean"])),
                fmt_num(get_nested(row, ["rewards", "sharp", "cumulative_reward", "mean"])),
                fmt_pct(get_nested(row, ["rewards", "change_pct", "cumulative_reward"])),
            ]
            for row in result["per_seed"]
        ],
    )
    print_table(
        ["Seed", "Base Var", "SHARP Var", "Var Δ"],
        [
            [
                row["seed"],
                fmt_num(get_nested(row, ["rewards", "baseline", "reward_variance", "mean"])),
                fmt_num(get_nested(row, ["rewards", "sharp", "reward_variance", "mean"])),
                fmt_pct(get_nested(row, ["rewards", "change_pct", "reward_variance"])),
            ]
            for row in result["per_seed"]
        ],
    )

    print("\n4. Imagined Latent Variance")
    print_table(
        ["Seed", "Base Inter", "SHARP Inter", "Inter Δ", "Base Intra", "SHARP Intra", "Intra Δ"],
        [
            [
                row["seed"],
                fmt_num(get_nested(row, ["latent_variance", "baseline", "inter_seed_variance"])),
                fmt_num(get_nested(row, ["latent_variance", "sharp", "inter_seed_variance"])),
                fmt_pct(get_nested(row, ["latent_variance", "change_pct", "inter_seed_variance"])),
                fmt_num(
                    get_nested(
                        row,
                        ["latent_variance", "baseline", "intra_trajectory_variance", "mean"],
                    )
                ),
                fmt_num(
                    get_nested(
                        row,
                        ["latent_variance", "sharp", "intra_trajectory_variance", "mean"],
                    )
                ),
                fmt_pct(
                    get_nested(row, ["latent_variance", "change_pct", "intra_trajectory_variance"])
                ),
            ]
            for row in result["per_seed"]
        ],
    )

    print("\n5. Early vs Late Latent Error")
    print_table(
        ["Seed", "Segment", "Base MSE", "SHARP MSE", "Reduction"],
        [
            [
                row["seed"],
                segment,
                fmt_num(get_nested(row, ["horizon_segments", "baseline", segment, "mean"])),
                fmt_num(get_nested(row, ["horizon_segments", "sharp", segment, "mean"])),
                fmt_pct(get_nested(row, ["horizon_segments", "reduction_pct", segment])),
            ]
            for row in result["per_seed"]
            for segment in SEGMENTS
        ],
    )

    print("\nCross-Seed Summary")
    summary = result["cross_seed_summary"]
    print(f"Included seeds: {summary['seeds']} (n={summary['n_included']})")
    print_table(
        ["Metric", "Mean", "Std", "N"],
        [summary_pct_row(f"band:{band}", summary["per_band_reduction_pct"][band]) for band in BAND_ORDER]
        + [
            summary_pct_row("obs_mse_reduction", summary["obs_mse_reduction_pct"]),
            summary_pct_row(
                "reward_cumulative_change",
                summary["reward_change_pct"]["cumulative_reward"],
            ),
            summary_pct_row(
                "latent_intra_variance_change",
                summary["latent_variance_change_pct"]["intra_trajectory_variance"],
            ),
        ],
    )

    errored = [row for row in result["per_seed"] if row.get("errors")]
    if errored:
        print("\nSeed Errors")
        for row in errored:
            print(f"  seed {row['seed']}: {'; '.join(row['errors'])}")

    print(f"\nSaved JSON: {result['output_path']}")


def print_table(headers: list[Any], rows: list[list[Any]]) -> None:
    string_rows = [[str(cell) for cell in row] for row in rows]
    string_headers = [str(header) for header in headers]
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


def fmt_pct(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):+.1f}%"


def fmt_pct_summary(summary: dict[str, Any] | None) -> str:
    if not summary or summary.get("n", 0) == 0:
        return "N/A"
    return f"{summary['mean']:+.1f}% +/- {summary['std']:.1f}% (n={summary['n']})"


def summary_pct_row(label: str, summary: dict[str, Any] | None) -> list[Any]:
    if not summary or summary.get("n", 0) == 0:
        return [label, "N/A", "N/A", 0]
    return [label, fmt_pct(summary["mean"]), f"{float(summary['std']):.1f}%", int(summary["n"])]


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
    result = {
        "task": args.task,
        "root": relative_path(args.root),
        "training_seeds": [int(seed) for seed in args.seeds],
        "num_rollout_seeds": int(args.num_rollout_seeds),
        "rollout_template": template,
        "frequency": {
            "module": "src.diagnostics.freq_decompose",
            "bands": {band: list(DEFAULT_BANDS[band]) for band in BAND_ORDER},
            "var_threshold": VAR_THRESHOLD,
            "filter_type": FILTER_TYPE,
        },
        "questions": {
            "per_band": "Does SHARP uniformly reduce all bands, or selectively?",
            "obs_space": "Does latent drift reduction translate to obs-space improvement?",
            "rewards": "Does SHARP affect the reward structure of imagined rollouts?",
            "latent_variance": "Does SHARP compress imagined latents too much?",
            "horizon_segments": "Does SHARP help uniformly across the horizon?",
        },
        "per_seed": per_seed,
        "cross_seed_summary": build_cross_seed_summary(per_seed),
        "output_path": relative_path(args.output),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(to_jsonable(result), indent=2) + "\n")
    print_report(result)


if __name__ == "__main__":
    main()
