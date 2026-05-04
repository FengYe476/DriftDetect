#!/usr/bin/env python3
"""Evaluate HIAD rollouts against baseline.

Computes J_slow (dc_trend MSE), J_total (all-band MSE), and per-band MSE for
baseline and HIAD rollouts, then reports the percentage change.

Usage:
    python scripts/run_hiad_eval.py \
        --baseline_dir results/rollouts/ \
        --hiad_dir results/rollouts_hiad/ \
        --output results/tables/hiad_first_run.json
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


DEFAULT_BASELINE_DIR = REPO_ROOT / "results" / "rollouts"
DEFAULT_HIAD_DIR = REPO_ROOT / "results" / "rollouts_hiad"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "tables" / "hiad_first_run.json"
BASELINE_TEMPLATE = "cheetah_v3_seed{seed}_v2.npz"
HIAD_TEMPLATE = "cheetah_hiad_seed{seed}.npz"
BAND_ORDER = list(DEFAULT_BANDS.keys())
VAR_THRESHOLD = 0.95
FILTER_TYPE = "butterworth"
DEFAULT_TRIM = (25, 175)
CONTINUE_THRESHOLD = 25.0
MARGINAL_THRESHOLD = 15.0


def main() -> None:
    args = parse_args()
    baseline_paths, hiad_paths = discover_matched_rollouts(
        baseline_dir=args.baseline_dir,
        hiad_dir=args.hiad_dir,
        seeds=args.seeds,
        baseline_template=args.baseline_template,
        hiad_template=args.hiad_template,
    )

    baseline = compute_frequency_metrics(
        paths=baseline_paths,
        trim=args.trim,
        signal_key_true=args.signal_key_true,
        signal_key_imag=args.signal_key_imag,
    )
    hiad = compute_frequency_metrics(
        paths=hiad_paths,
        trim=args.trim,
        signal_key_true=args.signal_key_true,
        signal_key_imag=args.signal_key_imag,
    )
    comparison = compare_metrics(baseline, hiad)
    verdict = verdict_for_change(comparison["J_slow_pct"])
    hiad_config = read_hiad_config(hiad_paths[0])

    result = {
        "n_seeds": len(args.seeds),
        "seeds": [int(seed) for seed in args.seeds],
        "baseline_dir": relative_path(args.baseline_dir),
        "hiad_dir": relative_path(args.hiad_dir),
        "baseline_files": [relative_path(path) for path in baseline_paths],
        "hiad_files": [relative_path(path) for path in hiad_paths],
        "hiad_config": hiad_config,
        "frequency_eval": {
            "bands": {band: list(bounds) for band, bounds in DEFAULT_BANDS.items()},
            "signal_keys": {
                "true": args.signal_key_true,
                "imagined": args.signal_key_imag,
            },
            "trim_window_inclusive": [int(args.trim[0]), int(args.trim[1])],
            "filter_type": FILTER_TYPE,
            "var_threshold": VAR_THRESHOLD,
        },
        "baseline": baseline,
        "hiad": hiad,
        "reduction": comparison,
        "verdict": verdict,
        "threshold": {
            "continue": CONTINUE_THRESHOLD,
            "marginal": MARGINAL_THRESHOLD,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(to_jsonable(result), indent=2) + "\n")
    print_summary(result, args.output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HIAD first-run rollouts.")
    parser.add_argument("--baseline_dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--hiad_dir", type=Path, default=DEFAULT_HIAD_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    parser.add_argument("--baseline_template", default=BASELINE_TEMPLATE)
    parser.add_argument("--hiad_template", default=HIAD_TEMPLATE)
    parser.add_argument("--signal_key_true", default="true_obs")
    parser.add_argument("--signal_key_imag", default="imagined_obs")
    parser.add_argument(
        "--trim",
        nargs="+",
        default=[str(DEFAULT_TRIM[0]), str(DEFAULT_TRIM[1])],
        help="Inclusive trim window, either '25 175' or '25,175'.",
    )
    args = parser.parse_args()
    args.baseline_dir = resolve_path(args.baseline_dir)
    args.hiad_dir = resolve_path(args.hiad_dir)
    args.output = resolve_path(args.output)
    args.trim = parse_trim(args.trim)
    if not args.seeds:
        raise ValueError("--seeds must contain at least one seed.")
    if not 0 <= args.trim[0] <= args.trim[1]:
        raise ValueError("--trim must be an inclusive non-empty window.")
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


def discover_matched_rollouts(
    *,
    baseline_dir: Path,
    hiad_dir: Path,
    seeds: list[int],
    baseline_template: str,
    hiad_template: str,
) -> tuple[list[Path], list[Path]]:
    baseline_paths = []
    hiad_paths = []
    missing = []
    for seed in seeds:
        baseline_path = baseline_dir / baseline_template.format(seed=seed)
        hiad_path = hiad_dir / hiad_template.format(seed=seed)
        if not baseline_path.exists():
            missing.append(relative_path(baseline_path))
        if not hiad_path.exists():
            missing.append(relative_path(hiad_path))
        baseline_paths.append(baseline_path)
        hiad_paths.append(hiad_path)
    if missing:
        raise FileNotFoundError("Missing rollout files:\n" + "\n".join(missing))
    return baseline_paths, hiad_paths


def compute_frequency_metrics(
    *,
    paths: list[Path],
    trim: tuple[int, int],
    signal_key_true: str,
    signal_key_imag: str,
) -> dict[str, Any]:
    curves = aggregate_curves(
        paths,
        signal_key_true=signal_key_true,
        signal_key_imag=signal_key_imag,
        metric="mse",
        bands=DEFAULT_BANDS,
        var_threshold=VAR_THRESHOLD,
        filter_type=FILTER_TYPE,
    )
    horizon = len(np.asarray(curves[BAND_ORDER[0]]["mean"]))
    if trim[1] >= horizon:
        raise ValueError(f"--trim end {trim[1]} outside rollout horizon {horizon}.")
    window = slice(trim[0], trim[1] + 1)
    per_band = {
        band: float(np.mean(np.asarray(curves[band]["mean"])[window]))
        for band in BAND_ORDER
    }
    return {
        "J_slow": float(per_band["dc_trend"]),
        "J_total": float(sum(per_band.values())),
        "per_band": per_band,
    }


def compare_metrics(
    baseline: dict[str, Any],
    hiad: dict[str, Any],
) -> dict[str, Any]:
    return {
        "J_slow_pct": percent_change(baseline["J_slow"], hiad["J_slow"]),
        "J_total_pct": percent_change(baseline["J_total"], hiad["J_total"]),
        "per_band_pct": {
            band: percent_change(baseline["per_band"][band], hiad["per_band"][band])
            for band in BAND_ORDER
        },
    }


def read_hiad_config(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {
            "eta": scalar_or_none(data, "hiad_eta"),
            "r": scalar_or_none(data, "hiad_r"),
            "re_est_freq": scalar_or_none(data, "hiad_re_est_freq"),
            "basis_mode": scalar_or_none(data, "hiad_basis_mode"),
        }


def scalar_or_none(data: np.lib.npyio.NpzFile, key: str) -> Any:
    if key not in data:
        return None
    value = data[key]
    if value.shape == ():
        return value.item()
    return value.tolist()


def percent_change(original: float, new: float) -> float | None:
    original = float(original)
    new = float(new)
    if original == 0.0:
        return None
    return (new - original) / original * 100.0


def verdict_for_change(j_slow_change_pct: float | None) -> str:
    if j_slow_change_pct is None:
        return "PIVOT"
    reduction = -float(j_slow_change_pct)
    if reduction >= CONTINUE_THRESHOLD:
        return "CONTINUE"
    if reduction >= MARGINAL_THRESHOLD:
        return "MARGINAL"
    return "PIVOT"


def print_summary(result: dict[str, Any], output_path: Path) -> None:
    config = result["hiad_config"]
    eta = config.get("eta")
    rank = config.get("r")
    re_est_freq = config.get("re_est_freq")
    print(
        "HIAD First Run Results "
        f"(eta={fmt_value(eta)}, r={fmt_value(rank, digits=0)}, "
        f"re_est_freq={fmt_value(re_est_freq, digits=0)})"
    )
    print("Metric          Baseline    HIAD        Change")
    print("--------------  ----------  ----------  --------")
    print_metric("J_slow", result["baseline"]["J_slow"], result["hiad"]["J_slow"], result["reduction"]["J_slow_pct"])
    print_metric("J_total", result["baseline"]["J_total"], result["hiad"]["J_total"], result["reduction"]["J_total_pct"])
    for band in BAND_ORDER:
        print_metric(
            band,
            result["baseline"]["per_band"][band],
            result["hiad"]["per_band"][band],
            result["reduction"]["per_band_pct"][band],
        )

    verdict = result["verdict"]
    if verdict == "CONTINUE":
        print("VERDICT: CONTINUE - HIAD clears threshold")
    elif verdict == "MARGINAL":
        print("VERDICT: MARGINAL - proceed with caution")
    else:
        print("VERDICT: PIVOT - HIAD below threshold, consider Path A")
    print(f"Saved: {relative_path(output_path)}")


def print_metric(name: str, baseline: float, hiad: float, change_pct: float | None) -> None:
    print(
        f"{name:<14}  "
        f"{fmt_value(baseline):>10}  "
        f"{fmt_value(hiad):>10}  "
        f"{fmt_pct(change_pct):>8}"
    )


def fmt_value(value: Any, digits: int = 3) -> str:
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
