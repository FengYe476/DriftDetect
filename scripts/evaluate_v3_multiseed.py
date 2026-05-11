#!/usr/bin/env python3
"""Evaluate V3 multi-seed baseline/SHARP rollout pairs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.diagnostics.error_curves import aggregate_curves  # noqa: E402
from src.diagnostics.freq_decompose import DEFAULT_BANDS  # noqa: E402


BAND_ORDER = list(DEFAULT_BANDS.keys())
DEFAULT_ROOT = REPO_ROOT / "results" / "v3_multiseed"
DEFAULT_SEEDS = [43, 44, 45]
DEFAULT_TRIM = (25, 175)
DETER_SLICE = slice(1024, 1536)
VAR_THRESHOLD = 0.95
FILTER_TYPE = "butterworth"


@dataclass(frozen=True)
class SeedPair:
    seed: int
    label: str
    baseline_rollout_dir: Path
    sharp_rollout_dir: Path
    baseline_run_dir: Path | None = None
    sharp_run_dir: Path | None = None
    reference: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate V3 multi-seed SHARP.")
    parser.add_argument("--task", default="dmc_cheetah_run")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument(
        "--seed-dirs",
        type=Path,
        nargs="*",
        default=None,
        help=(
            "Optional baseline or SHARP run dirs, e.g. "
            "results/v3_multiseed/cheetah_baseline_seed43. "
            "Sibling dirs are inferred from the seed number."
        ),
    )
    parser.add_argument(
        "--include_seed42",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include Month 8 seed=42 reference rollouts when available.",
    )
    parser.add_argument("--num_rollout_seeds", type=int, default=20)
    parser.add_argument("--trim_start", type=int, default=DEFAULT_TRIM[0])
    parser.add_argument("--trim_end", type=int, default=DEFAULT_TRIM[1])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    args.task_short = task_short(args.task)
    args.rollout_template = f"{args.task_short}_v3_seed{{seed}}_v2.npz"
    args.root = resolve_path(args.root)
    if args.output is None:
        args.output = (
            REPO_ROOT
            / "results"
            / "tables"
            / f"v3_{args.task_short}_multiseed_results.json"
        )
    else:
        args.output = resolve_path(args.output)
    if args.seed_dirs is not None:
        args.seed_dirs = [resolve_path(path) for path in args.seed_dirs]
    if args.num_rollout_seeds <= 0:
        raise ValueError("--num_rollout_seeds must be positive.")
    if not 0 <= args.trim_start < args.trim_end:
        raise ValueError("Expected 0 <= trim_start < trim_end.")
    return args


def task_short(task: str) -> str:
    if task == "dmc_cheetah_run" or task == "cheetah":
        return "cheetah"
    if task == "dmc_cartpole_swingup" or task == "cartpole":
        return "cartpole"
    cleaned = task.removeprefix("dmc_")
    return cleaned.split("_", 1)[0]


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


def seed_pairs_from_args(args: argparse.Namespace) -> list[SeedPair]:
    pairs: list[SeedPair] = []
    if args.include_seed42:
        pairs.append(reference_pair(args.task_short))

    if args.seed_dirs:
        seen = {(pair.seed, pair.reference) for pair in pairs}
        for seed_dir in args.seed_dirs:
            pair = infer_pair_from_seed_dir(seed_dir, args.task_short)
            key = (pair.seed, pair.reference)
            if key not in seen:
                pairs.append(pair)
                seen.add(key)
        return pairs

    for seed in args.seeds:
        pairs.append(default_pair(args.root, int(seed), args.task_short))
    return pairs


def reference_pair(short_name: str) -> SeedPair:
    if short_name == "cartpole":
        preferred_baseline = REPO_ROOT / "results/month8_controlled/rollouts_baseline_cartpole"
        baseline_rollout_dir = (
            preferred_baseline if preferred_baseline.exists() else REPO_ROOT / "results/rollouts"
        )
        # Month 8 Cartpole SHARP rollouts may use cheetah_v3_* naming; rename
        # them to cartpole_v3_* if needed before evaluation.
        return SeedPair(
            seed=42,
            label="seed42_month8_reference",
            baseline_rollout_dir=baseline_rollout_dir,
            sharp_rollout_dir=REPO_ROOT / "results/month8_controlled/rollouts_moment_match_cartpole",
            baseline_run_dir=REPO_ROOT / "results/checkpoints",
            sharp_run_dir=REPO_ROOT / "results/month8_moment_match_cartpole",
            reference=True,
        )
    return SeedPair(
        seed=42,
        label="seed42_month8_reference",
        baseline_rollout_dir=REPO_ROOT / "results/month8_controlled/rollouts_baseline",
        sharp_rollout_dir=REPO_ROOT / "results/month8_controlled/rollouts_moment_match",
        baseline_run_dir=REPO_ROOT / "results/month8_controlled/baseline",
        sharp_run_dir=REPO_ROOT / "results/month8_moment_match",
        reference=True,
    )


def default_pair(root: Path, seed: int, short_name: str) -> SeedPair:
    baseline_run_dir = root / f"{short_name}_baseline_seed{seed}"
    sharp_run_dir = root / f"{short_name}_sharp_seed{seed}"
    return SeedPair(
        seed=seed,
        label=f"seed{seed}",
        baseline_rollout_dir=baseline_run_dir / "rollouts",
        sharp_rollout_dir=sharp_run_dir / "rollouts",
        baseline_run_dir=baseline_run_dir,
        sharp_run_dir=sharp_run_dir,
    )


def infer_pair_from_seed_dir(seed_dir: Path, short_name: str) -> SeedPair:
    match = re.search(r"seed(\d+)", seed_dir.name)
    if match is None:
        raise ValueError(f"Could not infer seed from directory name: {seed_dir}")
    seed = int(match.group(1))
    root = seed_dir.parent
    return default_pair(root, seed, short_name)


def discover_rollouts(
    rollout_dir: Path,
    *,
    num_rollout_seeds: int,
    rollout_template: str,
) -> dict[str, Any]:
    paths = [
        rollout_dir / rollout_template.format(seed=seed)
        for seed in range(num_rollout_seeds)
    ]
    existing = [path for path in paths if path.exists()]
    missing = [seed for seed, path in enumerate(paths) if not path.exists()]
    return {
        "dir": relative_path(rollout_dir),
        "expected_count": int(num_rollout_seeds),
        "found_count": int(len(existing)),
        "complete": len(existing) == num_rollout_seeds,
        "missing_rollout_seeds": missing,
        "paths": [relative_path(path) for path in existing],
        "paths_abs": [str(path) for path in existing],
    }


def matched_rollout_paths(
    baseline_dir: Path,
    sharp_dir: Path,
    *,
    num_rollout_seeds: int,
    rollout_template: str,
) -> tuple[list[int], list[Path], list[Path]]:
    seeds: list[int] = []
    baseline_paths: list[Path] = []
    sharp_paths: list[Path] = []
    for rollout_seed in range(num_rollout_seeds):
        baseline_path = baseline_dir / rollout_template.format(seed=rollout_seed)
        sharp_path = sharp_dir / rollout_template.format(seed=rollout_seed)
        if baseline_path.exists() and sharp_path.exists():
            seeds.append(rollout_seed)
            baseline_paths.append(baseline_path)
            sharp_paths.append(sharp_path)
    return seeds, baseline_paths, sharp_paths


def evaluate_seed_pair(
    pair: SeedPair,
    *,
    num_rollout_seeds: int,
    rollout_template: str,
    trim: tuple[int, int],
) -> dict[str, Any]:
    baseline_status = discover_rollouts(
        pair.baseline_rollout_dir,
        num_rollout_seeds=num_rollout_seeds,
        rollout_template=rollout_template,
    )
    sharp_status = discover_rollouts(
        pair.sharp_rollout_dir,
        num_rollout_seeds=num_rollout_seeds,
        rollout_template=rollout_template,
    )
    matched_seeds, baseline_paths, sharp_paths = matched_rollout_paths(
        pair.baseline_rollout_dir,
        pair.sharp_rollout_dir,
        num_rollout_seeds=num_rollout_seeds,
        rollout_template=rollout_template,
    )
    complete = len(matched_seeds) == num_rollout_seeds

    result: dict[str, Any] = {
        "seed": int(pair.seed),
        "label": pair.label,
        "reference": bool(pair.reference),
        "status": "complete" if complete else ("partial" if matched_seeds else "missing"),
        "included_in_cross_seed_summary": bool(complete),
        "matched_rollout_seeds": [int(seed) for seed in matched_seeds],
        "baseline_rollouts": baseline_status,
        "sharp_rollouts": sharp_status,
        "baseline_eval_return": extract_eval_returns(
            pair.baseline_run_dir / "metrics.jsonl" if pair.baseline_run_dir else None
        ),
        "sharp_eval_return": extract_eval_returns(
            pair.sharp_run_dir / "metrics.jsonl" if pair.sharp_run_dir else None
        ),
        "baseline": None,
        "sharp": None,
        "comparison": None,
    }

    if not matched_seeds:
        return result

    baseline_metrics = compute_rollout_metrics(baseline_paths, trim=trim)
    sharp_metrics = compute_rollout_metrics(sharp_paths, trim=trim)
    result["baseline"] = baseline_metrics
    result["sharp"] = sharp_metrics
    result["comparison"] = compare_seed_metrics(
        baseline_metrics,
        sharp_metrics,
        result["baseline_eval_return"],
        result["sharp_eval_return"],
    )
    return result


def compute_rollout_metrics(paths: list[Path], *, trim: tuple[int, int]) -> dict[str, Any]:
    raw_values = [raw_j_total(path, trim=trim) for path in paths]
    frequency = compute_frequency_metrics(paths, trim=trim)
    return {
        "n_rollouts": int(len(paths)),
        "raw_J_total": summarize_values(raw_values),
        "frequency": frequency,
    }


def raw_j_total(path: Path, *, trim: tuple[int, int]) -> float:
    with np.load(path, allow_pickle=True) as data:
        true_latent = np.asarray(data["true_latent"], dtype=np.float64)
        imagined_latent = np.asarray(data["imagined_latent"], dtype=np.float64)
    if true_latent.shape != imagined_latent.shape:
        raise ValueError(f"{relative_path(path)} latent shape mismatch.")
    true_deter = extract_deter(true_latent)
    imagined_deter = extract_deter(imagined_latent)
    window = slice(trim[0], trim[1])
    mse_per_step = np.mean((imagined_deter - true_deter) ** 2, axis=1)
    return float(np.mean(mse_per_step[window]))


def extract_deter(latent: np.ndarray) -> np.ndarray:
    if latent.ndim != 2:
        raise ValueError(f"Expected 2D latent array, got {latent.shape}.")
    if latent.shape[1] == 1536:
        return latent[:, DETER_SLICE]
    if latent.shape[1] == 512:
        return latent
    raise ValueError(f"Expected latent dimension 1536 or 512, got {latent.shape[1]}.")


def compute_frequency_metrics(paths: list[Path], *, trim: tuple[int, int]) -> dict[str, Any]:
    curves = aggregate_curves(
        paths,
        signal_key_true="true_latent",
        signal_key_imag="imagined_latent",
        metric="mse",
        bands=DEFAULT_BANDS,
        var_threshold=VAR_THRESHOLD,
        filter_type=FILTER_TYPE,
    )
    window = slice(trim[0], trim[1])
    band_mse = {
        band: float(np.mean(np.asarray(curves[band]["mean"])[window]))
        for band in BAND_ORDER
    }
    total = float(sum(band_mse.values()))
    info = curves["_info"]["decompose_info"]
    return {
        "J_total": total,
        "per_band_mse": band_mse,
        "band_share": {
            band: float(value / total) if total > 0 else 0.0
            for band, value in band_mse.items()
        },
        "pca": {
            "n_components": int(info["n_pcs"]),
            "var_explained": float(info["var_explained"]),
        },
    }


def compare_seed_metrics(
    baseline: dict[str, Any],
    sharp: dict[str, Any],
    baseline_eval: dict[str, Any],
    sharp_eval: dict[str, Any],
) -> dict[str, Any]:
    baseline_raw = baseline["raw_J_total"]["mean"]
    sharp_raw = sharp["raw_J_total"]["mean"]
    baseline_freq = baseline["frequency"]["J_total"]
    sharp_freq = sharp["frequency"]["J_total"]
    comparison = {
        "raw_J_total_reduction_pct": percent_reduction(baseline_raw, sharp_raw),
        "frequency_J_total_reduction_pct": percent_reduction(baseline_freq, sharp_freq),
        "eval_return_peak_change_pct": percent_change(
            get_nested(baseline_eval, "peak", "value"),
            get_nested(sharp_eval, "peak", "value"),
        ),
        "per_band_reduction_pct": {},
    }
    for band in BAND_ORDER:
        comparison["per_band_reduction_pct"][band] = percent_reduction(
            baseline["frequency"]["per_band_mse"][band],
            sharp["frequency"]["per_band_mse"][band],
        )
    return comparison


def extract_eval_returns(metrics_path: Path | None) -> dict[str, Any]:
    result = {
        "metrics_path": relative_path(metrics_path),
        "metrics_exists": bool(metrics_path and metrics_path.exists()),
        "key": None,
        "history": [],
        "peak": None,
        "final": None,
    }
    if metrics_path is None or not metrics_path.exists():
        return result

    candidates = ("eval_return", "test_return", "test/return")
    history = []
    with metrics_path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Skipping malformed metrics line {line_number}: {exc}", flush=True)
                continue
            key = next((candidate for candidate in candidates if candidate in row), None)
            if key is None:
                continue
            history.append(
                {
                    "step": int(row.get("step", -1)),
                    "value": float(row[key]),
                    "key": key,
                }
            )

    if not history:
        return result
    peak = max(history, key=lambda item: item["value"])
    final = history[-1]
    result["key"] = history[0]["key"]
    result["history"] = history
    result["peak"] = {"step": int(peak["step"]), "value": float(peak["value"])}
    result["final"] = {"step": int(final["step"]), "value": float(final["value"])}
    return result


def build_cross_seed_summary(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    included = [row for row in seed_results if row["included_in_cross_seed_summary"]]
    summary: dict[str, Any] = {
        "n_complete": int(len(included)),
        "seeds": [int(row["seed"]) for row in included],
    }
    if not included:
        summary["raw_J_total_reduction_pct"] = None
        summary["frequency_J_total_reduction_pct"] = None
        summary["eval_return_peak_change_pct"] = None
        return summary

    summary["baseline_raw_J_total"] = summarize_values(
        [row["baseline"]["raw_J_total"]["mean"] for row in included]
    )
    summary["sharp_raw_J_total"] = summarize_values(
        [row["sharp"]["raw_J_total"]["mean"] for row in included]
    )
    summary["raw_J_total_reduction_pct"] = summarize_values(
        [row["comparison"]["raw_J_total_reduction_pct"] for row in included]
    )
    summary["frequency_J_total_reduction_pct"] = summarize_values(
        [row["comparison"]["frequency_J_total_reduction_pct"] for row in included]
    )
    eval_changes = [
        row["comparison"]["eval_return_peak_change_pct"]
        for row in included
        if row["comparison"]["eval_return_peak_change_pct"] is not None
    ]
    summary["eval_return_peak_change_pct"] = (
        summarize_values(eval_changes) if eval_changes else None
    )
    return summary


def summarize_values(values: list[float]) -> dict[str, float | int | None]:
    clean = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not clean:
        return {"n": 0, "mean": None, "std": None}
    array = np.asarray(clean, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=0)),
    }


def percent_reduction(baseline: float | None, sharp: float | None) -> float | None:
    if baseline is None or sharp is None or float(baseline) == 0.0:
        return None
    return (float(baseline) - float(sharp)) / float(baseline) * 100.0


def percent_change(baseline: float | None, sharp: float | None) -> float | None:
    if baseline is None or sharp is None or float(baseline) == 0.0:
        return None
    return (float(sharp) - float(baseline)) / float(baseline) * 100.0


def get_nested(data: dict[str, Any] | None, *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def print_summary(result: dict[str, Any]) -> None:
    print(f"\nV3 {result['task_short'].title()} Multi-Seed SHARP Evaluation")
    print("=" * 88)
    print(
        f"{'Seed':>6}  {'Status':<9} {'N':>3} "
        f"{'Base J':>10} {'SHARP J':>10} {'Red.':>9} {'Eval Δ':>9}"
    )
    print("-" * 88)
    for row in result["per_seed"]:
        baseline_j = get_nested(row, "baseline", "raw_J_total", "mean")
        sharp_j = get_nested(row, "sharp", "raw_J_total", "mean")
        reduction = get_nested(row, "comparison", "raw_J_total_reduction_pct")
        eval_change = get_nested(row, "comparison", "eval_return_peak_change_pct")
        n_rollouts = len(row.get("matched_rollout_seeds", []))
        print(
            f"{row['seed']:>6}  {row['status']:<9} {n_rollouts:>3} "
            f"{fmt_value(baseline_j):>10} {fmt_value(sharp_j):>10} "
            f"{fmt_pct(reduction):>9} {fmt_pct(eval_change):>9}"
        )

    summary = result["cross_seed_summary"]
    raw_reduction = summary.get("raw_J_total_reduction_pct")
    freq_reduction = summary.get("frequency_J_total_reduction_pct")
    print("-" * 88)
    print(f"Complete seeds: {summary['n_complete']} {summary['seeds']}")
    print(f"Raw J_total reduction:       {fmt_mean_std(raw_reduction)}")
    print(f"Frequency J_total reduction: {fmt_mean_std(freq_reduction)}")
    print(f"Saved: {relative_path(Path(result['output_path']))}")


def fmt_value(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def fmt_pct(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):+.1f}%"


def fmt_mean_std(summary: dict[str, Any] | None) -> str:
    if not summary or summary.get("n", 0) == 0:
        return "N/A"
    return f"{summary['mean']:.1f}% +- {summary['std']:.1f}% (n={summary['n']})"


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
    trim = (int(args.trim_start), int(args.trim_end))
    pairs = seed_pairs_from_args(args)
    per_seed = [
        evaluate_seed_pair(
            pair,
            num_rollout_seeds=args.num_rollout_seeds,
            rollout_template=args.rollout_template,
            trim=trim,
        )
        for pair in pairs
    ]
    result = {
        "task": args.task,
        "task_short": args.task_short,
        "metric": {
            "primary": "raw deterministic latent J_total",
            "trim_window_half_open": [trim[0], trim[1]],
            "rollout_template": args.rollout_template,
        },
        "frequency_eval": {
            "module": "src.diagnostics.freq_decompose",
            "bands": {band: list(bounds) for band, bounds in DEFAULT_BANDS.items()},
            "var_threshold": VAR_THRESHOLD,
            "filter_type": FILTER_TYPE,
        },
        "include_seed42": bool(args.include_seed42),
        "per_seed": per_seed,
        "cross_seed_summary": build_cross_seed_summary(per_seed),
        "output_path": str(args.output),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(to_jsonable(result), indent=2) + "\n")
    print_summary(result)


if __name__ == "__main__":
    main()
