"""Run filter-choice ablation for DriftDetect frequency diagnostics."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.diagnostics.error_curves import aggregate_curves  # noqa: E402
from src.diagnostics.freq_decompose import DEFAULT_BANDS  # noqa: E402


FILTER_TYPES = ["butterworth", "chebyshev1", "fir"]
FILTER_LABELS = {
    "butterworth": "Butterworth",
    "chebyshev1": "Chebyshev1",
    "fir": "FIR",
}
BAND_ORDER = list(DEFAULT_BANDS.keys())
ROLLOUT_DIR = REPO_ROOT / "results" / "rollouts"
TABLE_DIR = REPO_ROOT / "results" / "tables"
PARQUET_PATH = TABLE_DIR / "filter_ablation.parquet"
CSV_PATH = TABLE_DIR / "filter_ablation.csv"
N_SEEDS = 20
WINDOW = slice(25, 176)
ROBUST_DEVIATION_THRESHOLD = 20.0


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    rollout_paths = [ROLLOUT_DIR / f"cheetah_v3_seed{seed}_v2.npz" for seed in range(N_SEEDS)]
    missing = [path for path in rollout_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing rollout files: {missing}")

    print("Running filter ablation on 20 Cheetah v2 rollouts...")
    latent_results = run_space(
        rollout_paths,
        signal_space="latent",
        signal_key_true="true_latent",
        signal_key_imag="imagined_latent",
    )
    obs_results = run_space(
        rollout_paths,
        signal_space="obs",
        signal_key_true="true_obs",
        signal_key_imag="imagined_obs",
    )

    rows = build_rows(latent_results, signal_space="latent")
    rows.extend(build_rows(obs_results, signal_space="obs"))
    saved_path = save_rows(rows)

    latent_summary = summarize_space(latent_results)
    obs_summary = summarize_space(obs_results)
    print_space_table(
        "LATENT-SPACE",
        latent_summary,
        latent_results,
    )
    print_space_table(
        "OBS-SPACE",
        obs_summary,
        obs_results,
    )
    print_summary(latent_summary, obs_summary)
    print(f"\nSaved raw ablation data: {display_path(saved_path)}")


def run_space(
    rollout_paths: list[Path],
    *,
    signal_space: str,
    signal_key_true: str,
    signal_key_imag: str,
) -> dict[str, dict[str, Any]]:
    results = {}
    for filter_type in FILTER_TYPES:
        print(f"  {signal_space}: {filter_type}", flush=True)
        results[filter_type] = aggregate_curves(
            rollout_paths=rollout_paths,
            signal_key_true=signal_key_true,
            signal_key_imag=signal_key_imag,
            metric="mse",
            filter_type=filter_type,
        )
    return results


def summarize_space(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    band_values: dict[str, dict[str, float]] = {}
    deviations: dict[str, float] = {}
    rankings: dict[str, list[str]] = {}

    for filter_type, result in results.items():
        filter_values = {
            band: float(np.mean(result[band]["mean"][WINDOW])) for band in BAND_ORDER
        }
        rankings[filter_type] = sorted(
            BAND_ORDER,
            key=lambda band: filter_values[band],
            reverse=True,
        )
        for band, value in filter_values.items():
            band_values.setdefault(band, {})[filter_type] = value

    for band, values_by_filter in band_values.items():
        values = np.asarray([values_by_filter[filter_type] for filter_type in FILTER_TYPES])
        mean_value = float(np.mean(values))
        deviations[band] = 0.0
        if mean_value > 0:
            deviations[band] = float((np.max(values) - np.min(values)) / mean_value * 100.0)

    reference_ranking = rankings[FILTER_TYPES[0]]
    rankings_identical = all(rankings[filter_type] == reference_ranking for filter_type in FILTER_TYPES)
    max_deviation = max(deviations.values()) if deviations else 0.0
    return {
        "band_values": band_values,
        "deviations": deviations,
        "rankings": rankings,
        "rankings_identical": rankings_identical,
        "max_deviation": max_deviation,
    }


def print_space_table(
    label: str,
    summary: dict[str, Any],
    results: dict[str, dict[str, Any]],
) -> None:
    print()
    print(f"=== FILTER ABLATION: {label} MEAN MSE [25:175] ===")
    print()
    print(
        f"{'Band':<12}  {'Butterworth':>12}    {'Chebyshev1':>10}    "
        f"{'FIR':>10}    {'Max Deviation (%)':>17}"
    )
    for band in BAND_ORDER:
        values = summary["band_values"][band]
        print(
            f"{band:<12}  "
            f"{values['butterworth']:>12.4f}    "
            f"{values['chebyshev1']:>10.4f}    "
            f"{values['fir']:>10.4f}    "
            f"{summary['deviations'][band]:>16.1f}%"
        )
    print()
    for filter_type in FILTER_TYPES:
        spacing = " " if filter_type != "chebyshev1" else ""
        print(
            f"Band ranking ({FILTER_LABELS[filter_type]}):{spacing} "
            f"{' > '.join(summary['rankings'][filter_type])}"
        )
    print(f"Rankings identical: {yes_no(summary['rankings_identical'])}")
    if "latent" in label.lower():
        n_pcs = results["butterworth"]["_info"]["decompose_info"]["n_pcs"]
        print(f"PCA components used: {n_pcs}")


def print_summary(
    latent_summary: dict[str, Any],
    obs_summary: dict[str, Any],
) -> None:
    rankings_identical = (
        latent_summary["rankings_identical"] and obs_summary["rankings_identical"]
    )
    max_deviation = max(latent_summary["max_deviation"], obs_summary["max_deviation"])
    robust = max_deviation < ROBUST_DEVIATION_THRESHOLD and rankings_identical

    print()
    print("=== FILTER ABLATION SUMMARY ===")
    print(
        "Latent-space max deviation across all bands: "
        f"{latent_summary['max_deviation']:.1f}%"
    )
    print(
        "Obs-space max deviation across all bands: "
        f"{obs_summary['max_deviation']:.1f}%"
    )
    print(f"Band rankings identical across filters: {yes_no(rankings_identical)}")
    print(f"Conclusion: Results are {'ROBUST' if robust else 'SENSITIVE'} to filter choice.")


def build_rows(
    results: dict[str, dict[str, Any]],
    *,
    signal_space: str,
) -> list[dict[str, Any]]:
    rows = []
    for filter_type, result in results.items():
        for band in BAND_ORDER:
            mean = result[band]["mean"]
            std = result[band]["std"]
            for step, (mean_mse, std_mse) in enumerate(zip(mean, std)):
                rows.append(
                    {
                        "filter_type": filter_type,
                        "band": band,
                        "step": step,
                        "mean_mse": float(mean_mse),
                        "std_mse": float(std_mse),
                        "signal_space": signal_space,
                    }
                )
    return rows


def save_rows(rows: list[dict[str, Any]]) -> Path:
    try:
        import pandas as pd

        frame = pd.DataFrame(rows)
        try:
            frame.to_parquet(PARQUET_PATH, index=False)
            return PARQUET_PATH
        except Exception as exc:
            print(f"Parquet write failed ({exc}); falling back to CSV.", flush=True)
            frame.to_csv(CSV_PATH, index=False)
            return CSV_PATH
    except ImportError:
        with CSV_PATH.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "filter_type",
                    "band",
                    "step",
                    "mean_mse",
                    "std_mse",
                    "signal_space",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        return CSV_PATH


def display_path(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def yes_no(value: bool) -> str:
    return "YES" if value else "NO"


if __name__ == "__main__":
    main()
