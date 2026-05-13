#!/usr/bin/env python3
"""Compare incremental one-step proxy errors against multi-step drift.

SHARP optimizes one-step error statistics, but DriftDetect reports accumulated
open-loop drift. This script measures the closest one-step proxy available from
open-loop rollouts,

    delta_t = (z_hat[t + 1] - z_hat[t]) - (z[t + 1] - z[t]),

then compares its frequency-band energy against accumulated drift,

    drift_t = z_hat[t] - z[t].

The output highlights whether SHARP reduces one-step errors uniformly and
whether multi-step amplification factors differ by temporal band.
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

from src.diagnostics.freq_decompose import DEFAULT_BANDS, decompose, fit_pca  # noqa: E402


BAND_ORDER = ["dc_trend", "very_low", "low", "mid", "high"]
DEFAULT_ROOT = REPO_ROOT / "results" / "v3_multiseed"
DEFAULT_SEEDS = [43, 44, 45]
DEFAULT_NUM_ROLLOUT_SEEDS = 5
DETER_SLICE = slice(1024, 1536)
VAR_THRESHOLD = 0.95
FILTER_TYPE = "butterworth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze one-step proxy errors versus multi-step SHARP drift."
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
            / f"onestep_vs_multistep_{args.task}.json"
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
        "baseline": None,
        "sharp": None,
        "comparison": None,
    }
    if not matched:
        return result

    try:
        comparison = compare_conditions(baseline_paths, sharp_paths)
        result["baseline"] = comparison["baseline"]
        result["sharp"] = comparison["sharp"]
        result["comparison"] = comparison["comparison"]
        result["included_in_summary"] = True
    except Exception as exc:
        result["status"] = "error"
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    return result


def compare_conditions(baseline_paths: list[Path], sharp_paths: list[Path]) -> dict[str, Any]:
    baseline_rollouts = [load_rollout_components(path) for path in baseline_paths]
    sharp_rollouts = [load_rollout_components(path) for path in sharp_paths]

    delta_pca = fit_shared_basis(
        [row["delta"] for row in baseline_rollouts + sharp_rollouts],
    )
    drift_pca = fit_shared_basis(
        [row["drift"] for row in baseline_rollouts + sharp_rollouts],
    )

    baseline = condition_summary(baseline_rollouts, delta_pca=delta_pca, drift_pca=drift_pca)
    sharp = condition_summary(sharp_rollouts, delta_pca=delta_pca, drift_pca=drift_pca)
    return {
        "baseline": baseline,
        "sharp": sharp,
        "comparison": compare_summaries(baseline, sharp),
    }


def load_rollout_components(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        if "true_latent" not in data:
            raise KeyError(f"{relative_path(path)} does not contain 'true_latent'.")
        if "imagined_latent" not in data:
            raise KeyError(f"{relative_path(path)} does not contain 'imagined_latent'.")
        true_latent = latent_features(data["true_latent"])
        imagined_latent = latent_features(data["imagined_latent"])
    if true_latent.shape != imagined_latent.shape:
        raise ValueError(
            f"{relative_path(path)} latent shape mismatch: "
            f"{true_latent.shape} vs {imagined_latent.shape}"
        )
    if true_latent.shape[0] < 2:
        raise ValueError(f"{relative_path(path)} needs at least two timesteps.")
    delta = np.diff(imagined_latent, axis=0) - np.diff(true_latent, axis=0)
    drift = imagined_latent - true_latent
    return {
        "path": path,
        "true_latent_shape": tuple(int(x) for x in true_latent.shape),
        "delta": delta,
        "drift": drift,
    }


def latent_features(latent: np.ndarray) -> np.ndarray:
    values = np.asarray(latent, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    elif values.ndim > 2:
        values = values.reshape(values.shape[0], -1)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"Expected non-empty 2D latent, got shape {values.shape}.")
    if values.shape[1] == 1536:
        return values[:, DETER_SLICE]
    if values.shape[1] == 512:
        return values
    raise ValueError(f"Expected latent dimension 1536 or 512, got {values.shape[1]}.")


def fit_shared_basis(signals: list[np.ndarray]) -> Any:
    if not signals:
        return None
    reference_dim = signals[0].shape[1]
    for signal in signals:
        if signal.shape[1] != reference_dim:
            raise ValueError(
                f"Signal dimension mismatch: expected {reference_dim}, got {signal.shape[1]}."
            )
    if reference_dim <= 50:
        return None
    pooled = np.concatenate(signals, axis=0)
    pca, _ = fit_pca(pooled, var_threshold=VAR_THRESHOLD)
    return pca


def condition_summary(
    rollouts: list[dict[str, Any]],
    *,
    delta_pca: Any,
    drift_pca: Any,
) -> dict[str, Any]:
    deltas = [row["delta"] for row in rollouts]
    drifts = [row["drift"] for row in rollouts]
    return {
        "n_rollouts": int(len(rollouts)),
        "latent_shape": list(rollouts[0]["true_latent_shape"]),
        "one_step_proxy": {
            "definition": "(imagined[t+1] - imagined[t]) - (true[t+1] - true[t])",
            "raw_stats": one_step_raw_stats(deltas),
            "band_energy": band_energy_summary(deltas, pca=delta_pca),
        },
        "multi_step_drift": {
            "definition": "imagined[t] - true[t]",
            "raw_J_total": summarize_values([float(np.mean(drift**2)) for drift in drifts]),
            "band_energy": band_energy_summary(drifts, pca=drift_pca),
        },
        "amplification_factor": amplification_summary(deltas, drifts, delta_pca, drift_pca),
    }


def one_step_raw_stats(deltas: list[np.ndarray]) -> dict[str, Any]:
    return {
        "mean_squared_error": summarize_values(
            [float(np.mean(delta**2)) for delta in deltas]
        ),
        "mean_norm_sq": summarize_values(
            [float(np.mean(np.sum(delta**2, axis=1))) for delta in deltas]
        ),
        "mean_vector_mse": summarize_values(
            [float(np.mean(np.mean(delta, axis=0) ** 2)) for delta in deltas]
        ),
        "mean_vector_norm_sq": summarize_values(
            [float(np.sum(np.mean(delta, axis=0) ** 2)) for delta in deltas]
        ),
        "temporal_variance_mean": summarize_values(
            [float(np.mean(np.var(delta, axis=0))) for delta in deltas]
        ),
        "temporal_variance_sum": summarize_values(
            [float(np.sum(np.var(delta, axis=0))) for delta in deltas]
        ),
    }


def band_energy_summary(signals: list[np.ndarray], *, pca: Any) -> dict[str, Any]:
    per_rollout = []
    for signal in signals:
        decomposed = decompose(
            signal,
            bands=DEFAULT_BANDS,
            pca=pca,
            var_threshold=VAR_THRESHOLD,
            filter_type=FILTER_TYPE,
        )
        per_rollout.append(
            {
                band: float(np.mean(np.asarray(decomposed[band], dtype=np.float64) ** 2))
                for band in BAND_ORDER
            }
        )

    per_band = {
        band: summarize_values([row[band] for row in per_rollout])
        for band in BAND_ORDER
    }
    means = {band: per_band[band]["mean"] for band in BAND_ORDER}
    total = float(sum(value for value in means.values() if value is not None))
    return {
        "per_band": per_band,
        "J_total": total,
        "band_share": {
            band: safe_divide(means[band], total)
            for band in BAND_ORDER
        },
        "pca": pca_info(pca),
    }


def amplification_summary(
    deltas: list[np.ndarray],
    drifts: list[np.ndarray],
    delta_pca: Any,
    drift_pca: Any,
) -> dict[str, Any]:
    delta_energy = band_energy_summary(deltas, pca=delta_pca)
    drift_energy = band_energy_summary(drifts, pca=drift_pca)
    return {
        "per_band": {
            band: safe_ratio(
                drift_energy["per_band"][band]["mean"],
                delta_energy["per_band"][band]["mean"],
            )
            for band in BAND_ORDER
        },
        "definition": "multi_step_band_energy / one_step_incremental_band_energy",
    }


def pca_info(pca: Any) -> dict[str, Any]:
    if pca is None:
        return {"applied": False, "n_components": None, "var_explained": 1.0}
    return {
        "applied": True,
        "n_components": int(pca.n_components_),
        "var_explained": float(np.sum(pca.explained_variance_ratio_)),
    }


def compare_summaries(baseline: dict[str, Any], sharp: dict[str, Any]) -> dict[str, Any]:
    band_rows = {}
    for band in BAND_ORDER:
        base_one = baseline["one_step_proxy"]["band_energy"]["per_band"][band]["mean"]
        sharp_one = sharp["one_step_proxy"]["band_energy"]["per_band"][band]["mean"]
        base_multi = baseline["multi_step_drift"]["band_energy"]["per_band"][band]["mean"]
        sharp_multi = sharp["multi_step_drift"]["band_energy"]["per_band"][band]["mean"]
        band_rows[band] = {
            "baseline_one_step_band_energy": base_one,
            "sharp_one_step_band_energy": sharp_one,
            "one_step_change_pct": percent_change(base_one, sharp_one),
            "baseline_multi_step_band_energy": base_multi,
            "sharp_multi_step_band_energy": sharp_multi,
            "multi_step_change_pct": percent_change(base_multi, sharp_multi),
            "baseline_amplification": baseline["amplification_factor"]["per_band"][band],
            "sharp_amplification": sharp["amplification_factor"]["per_band"][band],
        }

    return {
        "band_table": band_rows,
        "one_step_raw_stats_change_pct": {
            "mean_squared_error": percent_change(
                baseline["one_step_proxy"]["raw_stats"]["mean_squared_error"]["mean"],
                sharp["one_step_proxy"]["raw_stats"]["mean_squared_error"]["mean"],
            ),
            "mean_vector_mse": percent_change(
                baseline["one_step_proxy"]["raw_stats"]["mean_vector_mse"]["mean"],
                sharp["one_step_proxy"]["raw_stats"]["mean_vector_mse"]["mean"],
            ),
            "temporal_variance_mean": percent_change(
                baseline["one_step_proxy"]["raw_stats"]["temporal_variance_mean"]["mean"],
                sharp["one_step_proxy"]["raw_stats"]["temporal_variance_mean"]["mean"],
            ),
        },
        "multi_step_raw_J_total_change_pct": percent_change(
            baseline["multi_step_drift"]["raw_J_total"]["mean"],
            sharp["multi_step_drift"]["raw_J_total"]["mean"],
        ),
    }


def build_cross_seed_summary(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    included = [row for row in seed_results if row.get("included_in_summary")]
    summary: dict[str, Any] = {
        "n_included": int(len(included)),
        "seeds": [int(row["seed"]) for row in included],
        "band_table": {},
        "one_step_raw_stats_change_pct": {
            key: summarize_nested(included, ["comparison", "one_step_raw_stats_change_pct", key])
            for key in ["mean_squared_error", "mean_vector_mse", "temporal_variance_mean"]
        },
        "multi_step_raw_J_total_change_pct": summarize_nested(
            included,
            ["comparison", "multi_step_raw_J_total_change_pct"],
        ),
    }
    for band in BAND_ORDER:
        summary["band_table"][band] = {
            "baseline_one_step_band_energy": summarize_nested(
                included,
                ["comparison", "band_table", band, "baseline_one_step_band_energy"],
            ),
            "sharp_one_step_band_energy": summarize_nested(
                included,
                ["comparison", "band_table", band, "sharp_one_step_band_energy"],
            ),
            "one_step_change_pct": summarize_nested(
                included,
                ["comparison", "band_table", band, "one_step_change_pct"],
            ),
            "baseline_multi_step_band_energy": summarize_nested(
                included,
                ["comparison", "band_table", band, "baseline_multi_step_band_energy"],
            ),
            "sharp_multi_step_band_energy": summarize_nested(
                included,
                ["comparison", "band_table", band, "sharp_multi_step_band_energy"],
            ),
            "multi_step_change_pct": summarize_nested(
                included,
                ["comparison", "band_table", band, "multi_step_change_pct"],
            ),
            "baseline_amplification": summarize_nested(
                included,
                ["comparison", "band_table", band, "baseline_amplification"],
            ),
            "sharp_amplification": summarize_nested(
                included,
                ["comparison", "band_table", band, "sharp_amplification"],
            ),
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


def percent_change(baseline: float | None, sharp: float | None) -> float | None:
    if baseline is None or sharp is None or float(baseline) == 0.0:
        return None
    return (float(sharp) - float(baseline)) / float(baseline) * 100.0


def safe_divide(numerator: float | None, denominator: float | None) -> float:
    if numerator is None or denominator is None or float(denominator) == 0.0:
        return 0.0
    return float(numerator) / float(denominator)


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or float(denominator) == 0.0:
        return None
    return float(numerator) / float(denominator)


def print_report(result: dict[str, Any]) -> None:
    title = f"One-Step vs Multi-Step Error Analysis: {result['task'].title()}"
    print(f"\n{title}")
    print("=" * len(title))
    print(f"Root: {result['root']}")
    print(f"Rollout template: {result['rollout_template']}")
    print(f"Training seeds: {result['training_seeds']}")
    print("Delta columns are SHARP vs baseline percent change; negative means reduction.")

    for row in result["per_seed"]:
        print(f"\nSeed {row['seed']} [{row['status']}], matched rollouts: {row['n_matched_rollouts']}")
        if row.get("errors"):
            print(f"  Errors: {'; '.join(row['errors'])}")
        print_band_table(row.get("comparison", {}).get("band_table") if row.get("comparison") else None)
        print_raw_stats_table(row)

    print("\nCross-Seed Mean Band Table")
    print_band_table(result["cross_seed_summary"]["band_table"], summary=True)
    print("\nCross-Seed Raw Proxy Summary")
    raw = result["cross_seed_summary"]["one_step_raw_stats_change_pct"]
    print_table(
        ["Metric", "Change Mean", "Std", "N"],
        [
            summary_pct_row("1-step MSE", raw["mean_squared_error"]),
            summary_pct_row("1-step mean-vector MSE", raw["mean_vector_mse"]),
            summary_pct_row("1-step temporal variance", raw["temporal_variance_mean"]),
            summary_pct_row(
                "multi-step raw J_total",
                result["cross_seed_summary"]["multi_step_raw_J_total_change_pct"],
            ),
        ],
    )
    print(f"\nSaved JSON: {result['output_path']}")


def print_band_table(band_table: dict[str, Any] | None, *, summary: bool = False) -> None:
    headers = [
        "Band",
        "Base 1-step",
        "SHARP 1-step",
        "1-step Δ",
        "Base Multi",
        "SHARP Multi",
        "Multi Δ",
        "Base Amp",
        "SHARP Amp",
    ]
    rows = []
    for band in BAND_ORDER:
        row = band_table.get(band) if band_table else None
        if summary:
            rows.append(
                [
                    band,
                    fmt_summary_num(row["baseline_one_step_band_energy"]) if row else "N/A",
                    fmt_summary_num(row["sharp_one_step_band_energy"]) if row else "N/A",
                    fmt_summary_pct(row["one_step_change_pct"]) if row else "N/A",
                    fmt_summary_num(row["baseline_multi_step_band_energy"]) if row else "N/A",
                    fmt_summary_num(row["sharp_multi_step_band_energy"]) if row else "N/A",
                    fmt_summary_pct(row["multi_step_change_pct"]) if row else "N/A",
                    fmt_summary_num(row["baseline_amplification"]) if row else "N/A",
                    fmt_summary_num(row["sharp_amplification"]) if row else "N/A",
                ]
            )
        else:
            rows.append(
                [
                    band,
                    fmt_num(row.get("baseline_one_step_band_energy") if row else None),
                    fmt_num(row.get("sharp_one_step_band_energy") if row else None),
                    fmt_pct(row.get("one_step_change_pct") if row else None),
                    fmt_num(row.get("baseline_multi_step_band_energy") if row else None),
                    fmt_num(row.get("sharp_multi_step_band_energy") if row else None),
                    fmt_pct(row.get("multi_step_change_pct") if row else None),
                    fmt_num(row.get("baseline_amplification") if row else None),
                    fmt_num(row.get("sharp_amplification") if row else None),
                ]
            )
    print_table(headers, rows)


def print_raw_stats_table(row: dict[str, Any]) -> None:
    if not row.get("comparison"):
        return
    base = row["baseline"]["one_step_proxy"]["raw_stats"]
    sharp = row["sharp"]["one_step_proxy"]["raw_stats"]
    change = row["comparison"]["one_step_raw_stats_change_pct"]
    print_table(
        ["Raw Proxy Stat", "Base", "SHARP", "Δ"],
        [
            [
                "mean squared error",
                fmt_num(base["mean_squared_error"]["mean"]),
                fmt_num(sharp["mean_squared_error"]["mean"]),
                fmt_pct(change["mean_squared_error"]),
            ],
            [
                "mean-vector MSE",
                fmt_num(base["mean_vector_mse"]["mean"]),
                fmt_num(sharp["mean_vector_mse"]["mean"]),
                fmt_pct(change["mean_vector_mse"]),
            ],
            [
                "temporal variance mean",
                fmt_num(base["temporal_variance_mean"]["mean"]),
                fmt_num(sharp["temporal_variance_mean"]["mean"]),
                fmt_pct(change["temporal_variance_mean"]),
            ],
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


def fmt_pct(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):+.1f}%"


def fmt_summary_num(summary: dict[str, Any] | None) -> str:
    if not summary or summary.get("n", 0) == 0:
        return "N/A"
    return f"{fmt_num(summary['mean'])} +/- {fmt_num(summary['std'])}"


def fmt_summary_pct(summary: dict[str, Any] | None) -> str:
    if not summary or summary.get("n", 0) == 0:
        return "N/A"
    return f"{summary['mean']:+.1f}% +/- {summary['std']:.1f}%"


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
            "basis": (
                "Shared PCA is fit per training seed across baseline and SHARP, "
                "separately for incremental deltas and accumulated drift."
            ),
        },
        "metric_notes": {
            "latent_slice": "Use deter slice 1024:1536 for 1536-dim latents; use all dims for 512.",
            "delta_pct_columns": "Percent change is SHARP relative to baseline; negative is reduction.",
            "amplification_factor": "multi_step_band_energy / one_step_incremental_band_energy",
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
