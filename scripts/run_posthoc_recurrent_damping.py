#!/usr/bin/env python3
"""Post-hoc recurrent damping test with arbitrary U basis.

Applies damping to existing baseline rollouts in latent space, then evaluates
J_slow and J_total in both latent-space frequency decomposition and raw MSE.

Tests both non-recurrent and recurrent modes across multiple eta values.

Usage:
    python scripts/run_posthoc_recurrent_damping.py \
        --basis results/smad/U_drift_cheetah_r10.npy \
        --rollout_dir results/rollouts/ \
        --output results/tables/posthoc_recurrent_U_drift.json

    python scripts/run_posthoc_recurrent_damping.py \
        --basis results/smad/current_U_adaptive_smad.npy \
        --rollout_dir results/rollouts/ \
        --output results/tables/posthoc_recurrent_U_adaptive.json

Recurrent mode uses original model deltas, not re-run img_step. This is a
linear approximation of rollout-time damping because the saved rollouts do not
include a callable RSSM transition model.
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

from src.diagnostics.freq_decompose import (  # noqa: E402
    DEFAULT_BANDS,
    DETER_SLICE,
    apply_pca,
    bandpass,
    fit_pca,
)


DEFAULT_ROLLOUT_DIR = REPO_ROOT / "results" / "rollouts"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "tables"
DEFAULT_ETAS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.50)
DEFAULT_SEEDS = tuple(range(20))
DEFAULT_TRIM = (25, 175)
ROLLOUT_TEMPLATE = "cheetah_v3_seed{seed}_v2.npz"
MODES = ("non_recurrent", "recurrent")
CAVEAT = (
    "Recurrent mode uses original model deltas, not re-run img_step. "
    "This is a linear approximation."
)
LATENT_DIM = 1536
DETER_DIM = 512
HORIZON = 200
VAR_THRESHOLD = 0.95
FILTER_TYPE = "butterworth"
ORTHONORMAL_ATOL = 1e-5


def main() -> None:
    args = parse_args()
    basis, basis_info = load_basis(args.basis, rank=args.rank)
    rollout_paths = discover_rollouts(args.rollout_dir, args.seeds)
    true_deter, imag_deter = load_deter_rollouts(rollout_paths)
    projector = basis @ basis.T

    eval_context = build_eval_context(true_deter)
    baseline = evaluate_candidate(
        true_deter=true_deter,
        candidate_deter=imag_deter,
        eval_context=eval_context,
        trim=args.trim,
    )

    results = []
    for eta in args.etas:
        for mode in MODES:
            damped = apply_damping(
                imag_deter=imag_deter,
                projector=projector,
                eta=float(eta),
                mode=mode,
            )
            metrics = evaluate_candidate(
                true_deter=true_deter,
                candidate_deter=damped,
                eval_context=eval_context,
                trim=args.trim,
            )
            results.append(build_result_entry(float(eta), mode, metrics, baseline))

    output = {
        "basis_path": relative_path(args.basis),
        "basis_shape": list(basis_info["original_shape"]),
        "basis_rank_used": int(basis.shape[1]),
        "basis_validation": {
            "orthonormal_max_abs_error": basis_info["orthonormal_error"],
            "orthonormal_atol": ORTHONORMAL_ATOL,
            "orthonormal_within_atol": basis_info["orthonormal_within_atol"],
        },
        "rollout_dir": relative_path(args.rollout_dir),
        "rollout_files": [relative_path(path) for path in rollout_paths],
        "n_seeds": int(len(args.seeds)),
        "seeds": [int(seed) for seed in args.seeds],
        "trim_window": [int(args.trim[0]), int(args.trim[1])],
        "etas": [float(eta) for eta in args.etas],
        "modes": list(MODES),
        "frequency_eval": {
            "bands": {band: list(bounds) for band, bounds in DEFAULT_BANDS.items()},
            "var_threshold": VAR_THRESHOLD,
            "filter_type": FILTER_TYPE,
            "pca_source": "pooled true_deter",
            "retained_components": int(eval_context["pca"].n_components_),
            "retained_variance_explained": float(
                np.sum(eval_context["pca"].explained_variance_ratio_)
            ),
        },
        "caveat": CAVEAT,
        "baseline": baseline,
        "results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(to_jsonable(output), indent=2) + "\n")
    print_summary(output, args.output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run post-hoc recurrent latent damping with an arbitrary U basis."
    )
    parser.add_argument("--basis", type=Path, required=True)
    parser.add_argument("--rollout_dir", type=Path, default=DEFAULT_ROLLOUT_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--etas", type=float, nargs="+", default=list(DEFAULT_ETAS))
    parser.add_argument(
        "--trim",
        nargs="+",
        default=[str(DEFAULT_TRIM[0]), str(DEFAULT_TRIM[1])],
        help="Inclusive trim window, either '25 175' or '25,175'.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Optional basis rank. Uses all basis columns when omitted.",
    )
    args = parser.parse_args()
    args.basis = resolve_path(args.basis)
    args.rollout_dir = resolve_path(args.rollout_dir)
    args.output = (
        default_output_path(args.basis)
        if args.output is None
        else resolve_path(args.output)
    )
    args.trim = parse_trim(args.trim)
    if not args.seeds:
        raise ValueError("--seeds must contain at least one seed.")
    if not args.etas:
        raise ValueError("--etas must contain at least one eta.")
    if any(eta < 0.0 for eta in args.etas):
        raise ValueError("--etas must be non-negative.")
    if args.rank is not None and args.rank <= 0:
        raise ValueError("--rank must be positive when provided.")
    if not 0 <= args.trim[0] <= args.trim[1] < HORIZON:
        raise ValueError(
            f"--trim must be an inclusive window inside 0..{HORIZON - 1}."
        )
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


def default_output_path(basis_path: Path) -> Path:
    return DEFAULT_OUTPUT_DIR / f"posthoc_recurrent_{basis_path.stem}.json"


def load_basis(path: Path, *, rank: int | None) -> tuple[np.ndarray, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Basis file not found: {path}")
    basis = np.load(path).astype(np.float64, copy=False)
    if basis.ndim != 2:
        raise ValueError(f"Basis must be 2D, got shape {basis.shape}.")
    if basis.shape[0] != DETER_DIM:
        raise ValueError(
            f"Basis leading dimension must be {DETER_DIM}, got {basis.shape[0]}."
        )
    if basis.shape[1] <= 0:
        raise ValueError("Basis must contain at least one column.")
    if not np.all(np.isfinite(basis)):
        raise ValueError("Basis contains non-finite values.")
    if rank is not None:
        if rank > basis.shape[1]:
            raise ValueError(
                f"Requested rank={rank}, but basis has only {basis.shape[1]} columns."
            )
        basis = basis[:, :rank]

    gram_error = orthonormal_error(basis)
    within_atol = bool(gram_error <= ORTHONORMAL_ATOL)
    if not within_atol:
        print(
            "Warning: basis columns are not orthonormal within "
            f"atol={ORTHONORMAL_ATOL:g}; max_abs_error={gram_error:.3e}.",
            flush=True,
        )
    return basis.copy(), {
        "original_shape": tuple(np.load(path, mmap_mode="r").shape),
        "orthonormal_error": float(gram_error),
        "orthonormal_within_atol": within_atol,
    }


def orthonormal_error(basis: np.ndarray) -> float:
    gram = basis.T @ basis
    return float(np.max(np.abs(gram - np.eye(gram.shape[0]))))


def discover_rollouts(rollout_dir: Path, seeds: list[int]) -> list[Path]:
    missing = []
    paths = []
    for seed in seeds:
        path = rollout_dir / ROLLOUT_TEMPLATE.format(seed=int(seed))
        if not path.exists():
            missing.append(relative_path(path))
        paths.append(path)
    if missing:
        raise FileNotFoundError("Missing rollout files:\n" + "\n".join(missing))
    return paths


def load_deter_rollouts(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    true_deters = []
    imag_deters = []
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            true_latent = require_latent(data, "true_latent", path)
            imag_latent = require_latent(data, "imagined_latent", path)
        true_deters.append(extract_deter_copy(true_latent))
        imag_deters.append(extract_deter_copy(imag_latent))
    return np.stack(true_deters, axis=0), np.stack(imag_deters, axis=0)


def require_latent(data: np.lib.npyio.NpzFile, key: str, path: Path) -> np.ndarray:
    if key not in data:
        raise KeyError(f"{relative_path(path)} does not contain {key!r}.")
    latent = np.asarray(data[key])
    expected = (HORIZON, LATENT_DIM)
    if latent.shape != expected:
        raise ValueError(
            f"{relative_path(path)}:{key} expected shape {expected}, got {latent.shape}."
        )
    if not np.all(np.isfinite(latent)):
        raise ValueError(f"{relative_path(path)}:{key} contains non-finite values.")
    return latent


def extract_deter_copy(latent: np.ndarray) -> np.ndarray:
    deter = np.asarray(latent[:, DETER_SLICE], dtype=np.float64)
    if deter.shape != (HORIZON, DETER_DIM):
        raise ValueError(f"Expected deter shape ({HORIZON}, {DETER_DIM}), got {deter.shape}.")
    return deter.copy()


def build_eval_context(true_deter: np.ndarray) -> dict[str, Any]:
    pca, _n_components = fit_pca(
        true_deter.reshape(-1, DETER_DIM),
        var_threshold=VAR_THRESHOLD,
    )
    true_projected = project_rollouts(true_deter, pca)
    true_bands = filter_all_bands(true_projected)
    return {
        "pca": pca,
        "true_projected": true_projected,
        "true_bands": true_bands,
    }


def project_rollouts(rollouts: np.ndarray, pca: Any) -> np.ndarray:
    n_seeds, horizon, deter_dim = rollouts.shape
    projected = apply_pca(rollouts.reshape(-1, deter_dim), pca)
    return projected.reshape(n_seeds, horizon, -1)


def filter_all_bands(projected: np.ndarray) -> dict[str, np.ndarray]:
    return {
        band: filter_projected(projected, low=low, high=high)
        for band, (low, high) in DEFAULT_BANDS.items()
    }


def filter_projected(projected: np.ndarray, *, low: float, high: float) -> np.ndarray:
    out = np.empty_like(projected, dtype=np.float64)
    for seed in range(projected.shape[0]):
        for dim in range(projected.shape[2]):
            out[seed, :, dim] = bandpass(
                projected[seed, :, dim],
                low=low,
                high=high,
                filter_type=FILTER_TYPE,
            )
    return out


def apply_damping(
    *,
    imag_deter: np.ndarray,
    projector: np.ndarray,
    eta: float,
    mode: str,
) -> np.ndarray:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}.")
    damped = imag_deter.copy() if mode == "non_recurrent" else np.zeros_like(imag_deter)
    if mode == "recurrent":
        damped[:, 0, :] = imag_deter[:, 0, :]

    deltas = imag_deter[:, 1:, :] - imag_deter[:, :-1, :]
    damped_deltas = deltas - eta * (deltas @ projector.T)
    if mode == "non_recurrent":
        damped[:, 1:, :] = imag_deter[:, :-1, :] + damped_deltas
    else:
        for t in range(HORIZON - 1):
            damped[:, t + 1, :] = damped[:, t, :] + damped_deltas[:, t, :]
    return damped


def evaluate_candidate(
    *,
    true_deter: np.ndarray,
    candidate_deter: np.ndarray,
    eval_context: dict[str, Any],
    trim: tuple[int, int],
) -> dict[str, Any]:
    candidate_projected = project_rollouts(candidate_deter, eval_context["pca"])
    candidate_bands = filter_all_bands(candidate_projected)
    per_band_seed = per_seed_band_mse(
        eval_context["true_bands"],
        candidate_bands,
        trim=trim,
    )
    j_slow_seed = per_band_seed["dc_trend"]
    j_total_seed = np.sum(
        np.stack([per_band_seed[band] for band in DEFAULT_BANDS], axis=0),
        axis=0,
    )
    raw_seed = per_seed_raw_mse(true_deter, candidate_deter, trim=trim)

    return {
        "J_slow": summarize(j_slow_seed),
        "J_total": summarize(j_total_seed),
        "raw_mse": summarize(raw_seed),
        "per_band": {
            band: summarize(values)
            for band, values in per_band_seed.items()
        },
    }


def per_seed_band_mse(
    true_bands: dict[str, np.ndarray],
    candidate_bands: dict[str, np.ndarray],
    *,
    trim: tuple[int, int],
) -> dict[str, np.ndarray]:
    window = slice(trim[0], trim[1] + 1)
    return {
        band: np.mean(
            (candidate_bands[band][:, window, :] - true_bands[band][:, window, :])
            ** 2,
            axis=(1, 2),
        )
        for band in DEFAULT_BANDS
    }


def per_seed_raw_mse(
    true_deter: np.ndarray,
    candidate_deter: np.ndarray,
    *,
    trim: tuple[int, int],
) -> np.ndarray:
    window = slice(trim[0], trim[1] + 1)
    return np.mean((candidate_deter[:, window, :] - true_deter[:, window, :]) ** 2, axis=(1, 2))


def summarize(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "per_seed": values.tolist(),
    }


def build_result_entry(
    eta: float,
    mode: str,
    metrics: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    entry = {
        "eta": float(eta),
        "mode": mode,
        "J_slow": with_baseline(metrics["J_slow"], baseline["J_slow"]),
        "J_total": with_baseline(metrics["J_total"], baseline["J_total"]),
        "raw_mse": with_baseline(metrics["raw_mse"], baseline["raw_mse"]),
        "J_slow_change_pct": percent_change(
            baseline["J_slow"]["mean"],
            metrics["J_slow"]["mean"],
        ),
        "J_total_change_pct": percent_change(
            baseline["J_total"]["mean"],
            metrics["J_total"]["mean"],
        ),
        "raw_mse_change_pct": percent_change(
            baseline["raw_mse"]["mean"],
            metrics["raw_mse"]["mean"],
        ),
        "per_band": {
            band: with_baseline(metrics["per_band"][band], baseline["per_band"][band])
            for band in DEFAULT_BANDS
        },
        "per_band_change_pct": {
            band: percent_change(
                baseline["per_band"][band]["mean"],
                metrics["per_band"][band]["mean"],
            )
            for band in DEFAULT_BANDS
        },
    }
    return entry


def with_baseline(metric: dict[str, Any], baseline_metric: dict[str, Any]) -> dict[str, Any]:
    return {
        **metric,
        "baseline_mean": float(baseline_metric["mean"]),
        "baseline_std": float(baseline_metric["std"]),
    }


def percent_change(original: float, new: float) -> float | None:
    original = float(original)
    new = float(new)
    if original == 0.0:
        return None
    return (new - original) / original * 100.0


def print_summary(result: dict[str, Any], output_path: Path) -> None:
    print(f"Post-hoc Damping: {Path(result['basis_path']).name}")
    print(f"Basis: {result['basis_path']} shape={result['basis_shape']}")
    print(f"Rollouts: {result['n_seeds']} Cheetah baseline seeds")
    print(f"Caveat: {result['caveat']}")
    print("eta    mode            J_slow     chg%     J_total    chg%     raw_mse    chg%")
    print("-----  --------------  ---------  -------  ---------  -------  ---------  -------")
    for entry in result["results"]:
        print(
            f"{entry['eta']:<5.2f}  "
            f"{entry['mode']:<14}  "
            f"{fmt_metric(entry['J_slow']['mean']):>9}  "
            f"{fmt_pct(entry['J_slow_change_pct']):>7}  "
            f"{fmt_metric(entry['J_total']['mean']):>9}  "
            f"{fmt_pct(entry['J_total_change_pct']):>7}  "
            f"{fmt_metric(entry['raw_mse']['mean']):>9}  "
            f"{fmt_pct(entry['raw_mse_change_pct']):>7}"
        )
    print(f"Saved: {relative_path(output_path)}")


def fmt_metric(value: float) -> str:
    return f"{float(value):.6f}"


def fmt_pct(value: float | None) -> str:
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
