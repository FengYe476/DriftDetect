#!/usr/bin/env python3
"""Evaluate SMAD Phase 2 DreamerV3 runs.

This script intentionally treats rollout extraction as a placeholder for now:
it consumes already-extracted v2 rollout files from each run directory and
performs eval-return parsing, frequency decomposition, and representation
health checks. Missing checkpoints or rollouts are reported cleanly so the
script can be committed before Phase 2 training finishes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.diagnostics.error_curves import aggregate_curves
from src.diagnostics.freq_decompose import DEFAULT_BANDS, bandpass, extract_deter


DEFAULT_BASELINE_DIR = REPO_ROOT / "results" / "smad_phase2" / "baseline"
DEFAULT_SMAD_DIR = REPO_ROOT / "results" / "smad_phase2" / "smad_eta020"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "results" / "tables" / "phase2_metrics.json"
DEFAULT_U_PATH = REPO_ROOT / "results" / "smad" / "U_drift_cheetah_r10.npy"

BAND_ORDER = list(DEFAULT_BANDS.keys())
TARGET_EVAL_STEPS = (100_000, 200_000, 300_000, 400_000, 500_000)
ROLLOUT_TEMPLATE = "cheetah_v3_seed{seed}_v2.npz"
VAR_THRESHOLD = 0.95
FILTER_TYPE = "butterworth"
REPRESENTATION_ANCHOR_STEPS = (50, 100, 150)
MIN_DIRECTION_NORM = 1e-8


def main() -> None:
    args = parse_args()
    baseline = evaluate_run(
        run_name="baseline",
        run_dir=args.baseline_dir,
        n_seeds=args.n_seeds,
        trim=args.trim,
    )
    smad = evaluate_run(
        run_name="smad",
        run_dir=args.smad_dir,
        n_seeds=args.n_seeds,
        trim=args.trim,
    )
    comparison = compare_runs(baseline, smad)

    result = {
        "analysis": "smad_phase2_eval",
        "baseline_dir": relative_path(args.baseline_dir),
        "smad_dir": relative_path(args.smad_dir),
        "n_seeds_requested": int(args.n_seeds),
        "horizon": int(args.horizon),
        "trim_window_inclusive": [int(args.trim[0]), int(args.trim[1])],
        "frequency_eval": {
            "bands": {band: list(bounds) for band, bounds in DEFAULT_BANDS.items()},
            "var_threshold": VAR_THRESHOLD,
            "filter_type": FILTER_TYPE,
            "pca_fit_scope": "pooled true deter latents within each run",
            "metric": "per-band MSE averaged over trim window",
        },
        "smad_runtime": {
            "eta": float(args.smad_eta),
            "U_path": relative_path(args.smad_U_path),
            "rollout_extraction_status": (
                "placeholder only; this script consumes pre-extracted rollouts"
            ),
        },
        "runs": {
            "baseline": baseline,
            "smad": smad,
        },
        "comparison": comparison,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(to_jsonable(result), indent=2) + "\n")
    print_summary(result)
    print(f"\nSaved metrics: {relative_path(args.output_json)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SMAD Phase 2 runs.")
    parser.add_argument("--baseline_dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--smad_dir", type=Path, default=DEFAULT_SMAD_DIR)
    parser.add_argument("--n_seeds", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument(
        "--trim",
        nargs="+",
        default=["25", "175"],
        help="Inclusive trim window, either '25 175' or '25,175'.",
    )
    parser.add_argument("--output_json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--smad_eta", type=float, default=0.20)
    parser.add_argument("--smad_U_path", type=Path, default=DEFAULT_U_PATH)
    args = parser.parse_args()

    args.baseline_dir = resolve_path(args.baseline_dir)
    args.smad_dir = resolve_path(args.smad_dir)
    args.output_json = resolve_path(args.output_json)
    args.smad_U_path = resolve_path(args.smad_U_path)
    args.trim = parse_trim(args.trim)

    if args.n_seeds <= 0:
        raise ValueError("--n_seeds must be positive.")
    if args.horizon <= 0:
        raise ValueError("--horizon must be positive.")
    if not 0 <= args.trim[0] <= args.trim[1] < args.horizon:
        raise ValueError("--trim must be inside the rollout horizon.")
    if args.smad_eta < 0:
        raise ValueError("--smad_eta must be non-negative.")
    return args


def parse_trim(values: list[str]) -> tuple[int, int]:
    if len(values) == 1:
        cleaned = values[0].strip().strip("()[]")
        parts = [part.strip() for part in cleaned.split(",") if part.strip()]
    elif len(values) == 2:
        parts = values
    else:
        raise ValueError("--trim expects two integers, e.g. --trim 25 175.")
    if len(parts) != 2:
        raise ValueError("--trim expects two integers, e.g. --trim 25,175.")
    return int(parts[0]), int(parts[1])


def evaluate_run(
    *,
    run_name: str,
    run_dir: Path,
    n_seeds: int,
    trim: tuple[int, int],
) -> dict[str, Any]:
    checkpoint_path = run_dir / "latest.pt"
    rollout_status = discover_rollouts(run_dir, n_seeds)
    result: dict[str, Any] = {
        "run_name": run_name,
        "run_dir": relative_path(run_dir),
        "checkpoint_path": relative_path(checkpoint_path),
        "checkpoint_exists": checkpoint_path.exists(),
        "eval_return": extract_eval_returns(run_dir / "metrics.jsonl"),
        "rollouts": rollout_status,
        "frequency": None,
        "representation_health": None,
    }

    if not checkpoint_path.exists():
        print(
            f"[{run_name}] Checkpoint not found at {relative_path(checkpoint_path)}; "
            "rollout extraction will be skipped."
        )

    if not rollout_status["complete"]:
        print(
            f"[{run_name}] Rollouts not found, run extraction separately. "
            f"Expected {n_seeds} files in {relative_path(run_dir / 'rollouts')}."
        )
        return result

    paths = [Path(path) for path in rollout_status["paths_abs"]]
    result["frequency"] = compute_frequency_metrics(paths, trim)
    result["representation_health"] = compute_representation_health(paths)
    return result


def discover_rollouts(run_dir: Path, n_seeds: int) -> dict[str, Any]:
    rollout_dir = run_dir / "rollouts"
    paths = [rollout_dir / ROLLOUT_TEMPLATE.format(seed=seed) for seed in range(n_seeds)]
    existing = [path for path in paths if path.exists()]
    missing_seeds = [seed for seed, path in enumerate(paths) if not path.exists()]
    return {
        "rollout_dir": relative_path(rollout_dir),
        "expected_count": int(n_seeds),
        "found_count": int(len(existing)),
        "complete": len(existing) == n_seeds,
        "missing_seeds": missing_seeds,
        "paths": [relative_path(path) for path in existing],
        "paths_abs": [str(path) for path in existing],
    }


def extract_eval_returns(metrics_path: Path) -> dict[str, Any]:
    result = {
        "metrics_path": relative_path(metrics_path),
        "metrics_exists": metrics_path.exists(),
        "key": None,
        "history": [],
        "peak": None,
        "final": None,
        "at_steps": {str(step): None for step in TARGET_EVAL_STEPS},
    }
    if not metrics_path.exists():
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
                print(f"Skipping malformed metrics line {line_number}: {exc}")
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
    result["peak"] = {"step": peak["step"], "value": peak["value"]}
    result["final"] = {"step": final["step"], "value": final["value"]}
    result["at_steps"] = {
        str(target): nearest_eval_return(history, target) for target in TARGET_EVAL_STEPS
    }
    return result


def nearest_eval_return(history: list[dict[str, Any]], target_step: int) -> dict[str, Any]:
    nearest = min(history, key=lambda item: abs(item["step"] - target_step))
    return {
        "target_step": int(target_step),
        "nearest_step": int(nearest["step"]),
        "value": float(nearest["value"]),
    }


def compute_frequency_metrics(paths: list[Path], trim: tuple[int, int]) -> dict[str, Any]:
    curves = aggregate_curves(
        paths,
        signal_key_true="true_latent",
        signal_key_imag="imagined_latent",
        metric="mse",
        bands=DEFAULT_BANDS,
        var_threshold=VAR_THRESHOLD,
        filter_type=FILTER_TYPE,
    )
    window = slice(trim[0], trim[1] + 1)
    band_mse = {
        band: float(np.mean(np.asarray(curves[band]["mean"])[window]))
        for band in BAND_ORDER
    }
    total = float(sum(band_mse.values()))
    shares = {
        band: float(band_mse[band] / total) if total > 0 else 0.0
        for band in BAND_ORDER
    }
    info = curves["_info"]["decompose_info"]
    return {
        "band_mse": band_mse,
        "band_share": shares,
        "J_slow": float(band_mse["dc_trend"]),
        "J_total": total,
        "trim_window_inclusive": [int(trim[0]), int(trim[1])],
        "pca": {
            "n_components": int(info["n_pcs"]),
            "var_explained": float(info["var_explained"]),
        },
        "curve_summary": {
            band: {
                "mean_step25": value_at(curves[band]["mean"], 25),
                "mean_step100": value_at(curves[band]["mean"], 100),
                "mean_step175": value_at(curves[band]["mean"], 175),
            }
            for band in BAND_ORDER
        },
    }


def compute_representation_health(paths: list[Path]) -> dict[str, Any]:
    deter_by_seed = [load_true_deter(path) for path in paths]
    pooled = np.concatenate(deter_by_seed, axis=0).astype(np.float64, copy=False)
    total_variance = float(np.trace(np.cov(pooled, rowvar=False)))
    dc_vars = []
    skipped = 0
    for deter in deter_by_seed:
        for anchor_step in REPRESENTATION_ANCHOR_STEPS:
            if anchor_step >= deter.shape[0]:
                skipped += 1
                continue
            direction, norm = dc_trend_direction(deter, anchor_step)
            if norm < MIN_DIRECTION_NORM:
                skipped += 1
                continue
            projection = deter @ direction
            dc_vars.append(float(np.var(projection, ddof=1)))
    return {
        "dc_trend_direction_variance": float(np.mean(dc_vars)) if dc_vars else None,
        "dc_trend_direction_variance_std": float(np.std(dc_vars, ddof=1))
        if len(dc_vars) > 1
        else 0.0,
        "total_deter_variance": total_variance,
        "n_dc_variance_samples": int(len(dc_vars)),
        "skipped_dc_variance_samples": int(skipped),
        "direction_method": (
            "For each seed and anchor step, lowpass-filter true deter in the "
            "dc_trend band, use filtered[anchor] - filtered.mean as a unit "
            "direction, then compute var(true_deter @ direction)."
        ),
        "anchor_steps": list(REPRESENTATION_ANCHOR_STEPS),
    }


def load_true_deter(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=True) as data:
        if "true_latent" not in data:
            raise KeyError(f"{path} does not contain 'true_latent'.")
        latent = np.asarray(data["true_latent"])
    if latent.ndim != 2:
        raise ValueError(f"{path}: true_latent must be 2D, got {latent.shape}.")
    if latent.shape[1] == 1536:
        return extract_deter(latent).astype(np.float64, copy=False)
    if latent.shape[1] == 512:
        return latent.astype(np.float64, copy=False)
    raise ValueError(f"{path}: expected true_latent dim 1536 or 512, got {latent.shape}.")


def dc_trend_direction(deter: np.ndarray, anchor_step: int) -> tuple[np.ndarray, float]:
    filtered = np.empty_like(deter, dtype=np.float64)
    low, high = DEFAULT_BANDS["dc_trend"]
    for dim in range(deter.shape[1]):
        filtered[:, dim] = bandpass(
            deter[:, dim],
            low=low,
            high=high,
            filter_type=FILTER_TYPE,
        )
    direction = filtered[anchor_step] - filtered.mean(axis=0)
    norm = float(np.linalg.norm(direction))
    if norm < MIN_DIRECTION_NORM:
        return np.zeros(deter.shape[1], dtype=np.float64), norm
    return direction / norm, norm


def compare_runs(baseline: dict[str, Any], smad: dict[str, Any]) -> dict[str, Any]:
    comparison: dict[str, Any] = {
        "eval_return_change_pct": percent_change(
            get_nested(baseline, "eval_return", "peak", "value"),
            get_nested(smad, "eval_return", "peak", "value"),
        ),
        "J_slow_reduction_pct": percent_reduction(
            get_nested(baseline, "frequency", "J_slow"),
            get_nested(smad, "frequency", "J_slow"),
        ),
        "J_total_reduction_pct": percent_reduction(
            get_nested(baseline, "frequency", "J_total"),
            get_nested(smad, "frequency", "J_total"),
        ),
        "per_band": {},
        "representation_health": {},
        "verdict": {},
    }

    for band in BAND_ORDER:
        base_value = get_nested(baseline, "frequency", "band_mse", band)
        smad_value = get_nested(smad, "frequency", "band_mse", band)
        comparison["per_band"][band] = {
            "baseline": base_value,
            "smad": smad_value,
            "change_pct": percent_change(base_value, smad_value),
            "reduction_pct": percent_reduction(base_value, smad_value),
        }

    base_dc_var = get_nested(
        baseline,
        "representation_health",
        "dc_trend_direction_variance",
    )
    smad_dc_var = get_nested(
        smad,
        "representation_health",
        "dc_trend_direction_variance",
    )
    base_total_var = get_nested(
        baseline,
        "representation_health",
        "total_deter_variance",
    )
    smad_total_var = get_nested(
        smad,
        "representation_health",
        "total_deter_variance",
    )
    dc_ratio = safe_ratio(smad_dc_var, base_dc_var)
    total_ratio = safe_ratio(smad_total_var, base_total_var)
    comparison["representation_health"] = {
        "dc_trend_variance_ratio": dc_ratio,
        "total_deter_variance_ratio": total_ratio,
        "collapse_warning": None if dc_ratio is None else dc_ratio < 0.70,
        "collapse_rule": "Warning if SMAD dc_trend variance drops more than 30%.",
    }

    slow_ok = threshold_yes_no(comparison["J_slow_reduction_pct"], minimum=20.0)
    return_ok = threshold_yes_no(comparison["eval_return_change_pct"], minimum=-5.0)
    comparison["verdict"] = {
        "J_slow_reduction_ge_20pct": slow_ok,
        "eval_return_drop_le_5pct": return_ok,
        "overall": overall_verdict(slow_ok, return_ok),
    }
    return comparison


def print_summary(result: dict[str, Any]) -> None:
    baseline = result["runs"]["baseline"]
    smad = result["runs"]["smad"]
    comparison = result["comparison"]

    print("============================================")
    print("SMAD Phase 2 Evaluation Summary")
    print("============================================")
    print("")
    print("Eval Return:")
    print(
        "  Baseline peak: "
        f"{fmt_metric(get_nested(baseline, 'eval_return', 'peak', 'value'))}  |  "
        "SMAD peak: "
        f"{fmt_metric(get_nested(smad, 'eval_return', 'peak', 'value'))}  |  "
        f"Change: {fmt_pct(comparison['eval_return_change_pct'])}"
    )
    print(
        "  Baseline final: "
        f"{fmt_metric(get_nested(baseline, 'eval_return', 'final', 'value'))} |  "
        "SMAD final: "
        f"{fmt_metric(get_nested(smad, 'eval_return', 'final', 'value'))}"
    )
    print("")
    print("Imagination Fidelity:")
    print("  J_slow (dc_trend MSE):")
    print(
        "    Baseline: "
        f"{fmt_metric(get_nested(baseline, 'frequency', 'J_slow'))}  |  "
        f"SMAD: {fmt_metric(get_nested(smad, 'frequency', 'J_slow'))}  |  "
        f"Reduction: {fmt_pct(comparison['J_slow_reduction_pct'])}"
    )
    print("  J_total (all-band MSE):")
    print(
        "    Baseline: "
        f"{fmt_metric(get_nested(baseline, 'frequency', 'J_total'))}  |  "
        f"SMAD: {fmt_metric(get_nested(smad, 'frequency', 'J_total'))}  |  "
        f"Reduction: {fmt_pct(comparison['J_total_reduction_pct'])}"
    )
    print("")
    print("Per-Band MSE:")
    print("  Band        Baseline    SMAD        Change%")
    for band in BAND_ORDER:
        band_result = comparison["per_band"][band]
        print(
            f"  {band:<10}  "
            f"{fmt_metric(band_result['baseline']):>8}    "
            f"{fmt_metric(band_result['smad']):>8}    "
            f"{fmt_pct(band_result['change_pct']):>8}"
        )
    print("")
    health = comparison["representation_health"]
    print("Representation Health:")
    print(
        "  DC_trend variance: "
        f"Baseline={fmt_metric(get_nested(baseline, 'representation_health', 'dc_trend_direction_variance'))}  "
        f"SMAD={fmt_metric(get_nested(smad, 'representation_health', 'dc_trend_direction_variance'))}  "
        f"Ratio={fmt_metric(health['dc_trend_variance_ratio'], digits=2)}"
    )
    print(
        "  Total deter variance: "
        f"Baseline={fmt_metric(get_nested(baseline, 'representation_health', 'total_deter_variance'))}  "
        f"SMAD={fmt_metric(get_nested(smad, 'representation_health', 'total_deter_variance'))}  "
        f"Ratio={fmt_metric(health['total_deter_variance_ratio'], digits=2)}"
    )
    print(f"  Collapse warning: {yes_no_unknown(health['collapse_warning'])}")
    print("")
    verdict = comparison["verdict"]
    print("Finding D Verdict:")
    print(f"  J_slow reduction >= 20%: {verdict['J_slow_reduction_ge_20pct']}")
    print(f"  eval_return drop <= 5%: {verdict['eval_return_drop_le_5pct']}")
    print(f"  VERDICT: {verdict['overall']}")
    print("============================================")


def percent_reduction(original: Any, new: Any) -> float | None:
    if original is None or new is None:
        return None
    original = float(original)
    new = float(new)
    if original == 0:
        return None
    return (original - new) / original * 100.0


def percent_change(original: Any, new: Any) -> float | None:
    if original is None or new is None:
        return None
    original = float(original)
    new = float(new)
    if original == 0:
        return None
    return (new - original) / original * 100.0


def safe_ratio(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or denominator is None:
        return None
    denominator = float(denominator)
    if denominator == 0:
        return None
    return float(numerator) / denominator


def threshold_yes_no(value: float | None, *, minimum: float) -> str:
    if value is None:
        return "UNKNOWN"
    return "YES" if value >= minimum else "NO"


def overall_verdict(slow_ok: str, return_ok: str) -> str:
    if "UNKNOWN" in {slow_ok, return_ok}:
        return "INCOMPLETE"
    if slow_ok == "YES" and return_ok == "YES":
        return "SUPPORTED"
    if slow_ok == "NO" and return_ok == "NO":
        return "NOT SUPPORTED"
    return "MIXED"


def yes_no_unknown(value: Any) -> str:
    if value is None:
        return "UNKNOWN"
    return "YES" if bool(value) else "NO"


def get_nested(container: dict[str, Any], *keys: str) -> Any:
    current: Any = container
    for key in keys:
        if current is None or not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def value_at(array: Any, index: int) -> float | None:
    values = np.asarray(array)
    if index < 0 or index >= values.shape[0]:
        return None
    return float(values[index])


def fmt_metric(value: Any, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def fmt_pct(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):+.1f}%"


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


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


if __name__ == "__main__":
    main()
