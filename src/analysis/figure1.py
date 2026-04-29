"""Generate Figure 1 for DriftDetect frequency-vs-horizon error analysis."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "driftdetect-matplotlib-cache"),
)

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

sys.path.insert(0, str(REPO_ROOT))

from src.diagnostics.error_curves import aggregate_curves  # noqa: E402
from src.diagnostics.freq_decompose import DEFAULT_BANDS  # noqa: E402


ROLLOUT_DIR = REPO_ROOT / "results" / "rollouts"
FIGURE_DIR = REPO_ROOT / "results" / "figures"

LATENT_FIG = FIGURE_DIR / "figure1_cheetah_latent_v1.pdf"
LATENT_FIG_LOG = FIGURE_DIR / "figure1_cheetah_latent_v1_log.pdf"
OBS_FIG = FIGURE_DIR / "figure1_cheetah_obs_v1.pdf"
OBS_FIG_LOG = FIGURE_DIR / "figure1_cheetah_obs_v1_log.pdf"

TASK = "dmc_cheetah_run"
N_SEEDS = 20
HORIZON = 200
PLOT_START = 10
PLOT_END = 189
STATS_START = 25
STATS_END = 175
VAR_THRESHOLD = 0.95

BAND_ORDER = list(DEFAULT_BANDS.keys())
BAND_LABELS = {
    "dc_trend": "DC/Trend (0-0.02)",
    "very_low": "Very Low (0.02-0.05)",
    "low": "Low (0.05-0.10)",
    "mid": "Mid (0.10-0.20)",
    "high": "High (0.20-0.50)",
}
BAND_COLORS = {
    "dc_trend": "#7f1d1d",
    "very_low": "#d97706",
    "low": "#1d4ed8",
    "mid": "#15803d",
    "high": "#7c3aed",
}


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    rollout_paths = collect_rollout_paths()
    latent_result = aggregate_curves(
        rollout_paths=rollout_paths,
        signal_key_true="true_latent",
        signal_key_imag="imagined_latent",
        metric="mse",
        var_threshold=VAR_THRESHOLD,
    )
    obs_result = aggregate_curves(
        rollout_paths=rollout_paths,
        signal_key_true="true_obs",
        signal_key_imag="imagined_obs",
        metric="mse",
        var_threshold=VAR_THRESHOLD,
    )

    plot_result(
        latent_result,
        LATENT_FIG,
        title="DreamerV3 Cheetah Run - Latent-Space Drift by Frequency Band",
        ylabel="Per-Band MSE",
        annotation="20 seeds, 95% CI, Butterworth order 4, PCA 95% var",
        log_scale=False,
    )
    plot_result(
        latent_result,
        LATENT_FIG_LOG,
        title="DreamerV3 Cheetah Run - Latent-Space Drift by Frequency Band",
        ylabel="Per-Band MSE",
        annotation="20 seeds, 95% CI, Butterworth order 4, PCA 95% var",
        log_scale=True,
    )
    plot_result(
        obs_result,
        OBS_FIG,
        title="DreamerV3 Cheetah Run - Obs-Space Drift by Frequency Band",
        ylabel="Per-Band MSE (obs)",
        annotation="20 seeds, 95% CI, Butterworth order 4, no PCA",
        log_scale=False,
    )
    plot_result(
        obs_result,
        OBS_FIG_LOG,
        title="DreamerV3 Cheetah Run - Obs-Space Drift by Frequency Band",
        ylabel="Per-Band MSE (obs)",
        annotation="20 seeds, 95% CI, Butterworth order 4, no PCA",
        log_scale=True,
    )

    print_summary(latent_result, obs_result)
    print_generated_files()


def collect_rollout_paths() -> list[Path]:
    paths = [ROLLOUT_DIR / f"cheetah_v3_seed{seed}_v2.npz" for seed in range(N_SEEDS)]
    missing = [path for path in paths if not path.exists()]
    if missing:
        missing_str = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing rollout files: {missing_str}")
    return paths


def plot_result(
    result: dict[str, Any],
    output_path: Path,
    *,
    title: str,
    ylabel: str,
    annotation: str,
    log_scale: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    x = np.arange(HORIZON)
    plot_slice = slice(PLOT_START, PLOT_END + 1)
    x_plot = x[plot_slice]
    positive_floor = positive_plot_floor(result, plot_slice)

    for band in BAND_ORDER:
        mean = np.asarray(result[band]["mean"], dtype=np.float64)[plot_slice]
        ci_low = np.asarray(result[band]["ci_low"], dtype=np.float64)[plot_slice]
        ci_high = np.asarray(result[band]["ci_high"], dtype=np.float64)[plot_slice]

        if log_scale:
            mean = np.clip(mean, positive_floor, None)
            ci_low = np.clip(ci_low, positive_floor, None)
            ci_high = np.clip(ci_high, positive_floor, None)

        ax.plot(
            x_plot,
            mean,
            color=BAND_COLORS[band],
            linewidth=1.5,
            label=BAND_LABELS[band],
        )
        ax.fill_between(
            x_plot,
            ci_low,
            ci_high,
            color=BAND_COLORS[band],
            alpha=0.2,
            linewidth=0,
        )

    ax.set_xlim(PLOT_START, PLOT_END)
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Imagination Horizon Step", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(loc="best", frameon=True, fontsize=9)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(
        0.98,
        0.05,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        bbox={
            "facecolor": "#f3f4f6",
            "edgecolor": "#d1d5db",
            "alpha": 0.7,
            "boxstyle": "round,pad=0.25",
        },
    )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def positive_plot_floor(result: dict[str, Any], plot_slice: slice) -> float:
    positive_values: list[float] = []
    for band in BAND_ORDER:
        for key in ("mean", "ci_low", "ci_high"):
            values = np.asarray(result[band][key], dtype=np.float64)[plot_slice]
            positive_values.extend(values[values > 0].tolist())
    if not positive_values:
        return 1e-8
    return max(1e-8, min(positive_values) * 0.5)


def print_summary(latent_result: dict[str, Any], obs_result: dict[str, Any]) -> None:
    latent_summary = summarize_result(latent_result)
    obs_summary = summarize_result(obs_result)

    print("=== FIGURE 1 STATISTICAL SUMMARY ===")
    print()
    print(
        "Latent-space "
        f"({latent_result['_info']['n_seeds']} seeds, "
        f"n_pcs={latent_result['_info']['decompose_info']['n_pcs']}):"
    )
    print(
        "  Band          Mean MSE [25:175]   Growth ratio (step25->step175)   "
        "p-value (slope > 0)"
    )
    for band in BAND_ORDER:
        stats = latent_summary["bands"][band]
        print(
            f"  {band:<12}"
            f"{stats['window_mean']:>10.4f}              "
            f"{stats['growth_ratio']:>6.2f}                        "
            f"{stats['p_value_one_sided']:>7.4f}"
        )
    print()
    print(f"Band ranking by mean MSE: {latent_summary['ranking_text']}")
    print(
        "Dominant band carries "
        f"{latent_summary['dominant_share'] * 100:.1f}% of total mean MSE."
    )
    print()
    print("Finding A assessment:")
    print(
        "  - Frequency asymmetry present: "
        f"{yes_no(latent_summary['frequency_asymmetry'])}"
    )
    print(
        "  - DC/trend dominates: "
        f"{yes_no(latent_summary['dc_dominates'])} "
        f"({latent_summary['dc_share'] * 100:.1f}% of total)"
    )
    print(
        "  - High-freq suppressed: "
        f"{yes_no(latent_summary['high_suppressed'])} "
        f"({latent_summary['high_share'] * 100:.1f}% of total)"
    )
    print(
        "  - Slope significance: "
        f"{latent_summary['positive_slope_count']}/5 bands have p < 0.05 for positive slope"
    )
    print()
    print(
        "Obs-space "
        f"({obs_result['_info']['n_seeds']} seeds, "
        f"n_dims={obs_result['_info']['decompose_info']['output_dim']}):"
    )
    print(
        "  Band          Mean MSE [25:175]   Growth ratio (step25->step175)   "
        "p-value (slope > 0)"
    )
    for band in BAND_ORDER:
        stats = obs_summary["bands"][band]
        print(
            f"  {band:<12}"
            f"{stats['window_mean']:>10.4f}              "
            f"{stats['growth_ratio']:>6.2f}                        "
            f"{stats['p_value_one_sided']:>7.4f}"
        )
    print()
    print(f"Obs band ranking by mean MSE: {obs_summary['ranking_text']}")
    print(
        "Obs dominant band carries "
        f"{obs_summary['dominant_share'] * 100:.1f}% of total mean MSE."
    )


def summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    stats_slice = slice(STATS_START, STATS_END + 1)
    bands: dict[str, dict[str, float]] = {}
    band_means: dict[str, float] = {}
    for band in BAND_ORDER:
        mean_curve = np.asarray(result[band]["mean"], dtype=np.float64)
        window_curve = mean_curve[stats_slice]
        slope, _, _, p_value_two_sided, _ = linregress(
            np.arange(STATS_START, STATS_END + 1),
            window_curve,
        )
        growth_ratio = safe_ratio(mean_curve[STATS_END], mean_curve[STATS_START])
        p_value_one_sided = one_sided_positive_p_value(slope, p_value_two_sided)
        window_mean = float(np.mean(window_curve))
        bands[band] = {
            "window_mean": window_mean,
            "growth_ratio": growth_ratio,
            "slope": float(slope),
            "p_value_one_sided": p_value_one_sided,
        }
        band_means[band] = window_mean

    ranking = sorted(band_means, key=band_means.get, reverse=True)
    total_mean = float(sum(band_means.values()))
    dominant_band = ranking[0]
    dominant_share = band_means[dominant_band] / total_mean if total_mean > 0 else 0.0
    dc_share = band_means["dc_trend"] / total_mean if total_mean > 0 else 0.0
    high_share = band_means["high"] / total_mean if total_mean > 0 else 0.0
    min_mean = min(band_means.values()) if band_means else 0.0
    max_mean = max(band_means.values()) if band_means else 0.0

    return {
        "bands": bands,
        "ranking": ranking,
        "ranking_text": " > ".join(ranking),
        "dominant_band": dominant_band,
        "dominant_share": dominant_share,
        "dc_share": dc_share,
        "high_share": high_share,
        "dc_dominates": dominant_band == "dc_trend" and dc_share >= 0.35,
        "high_suppressed": high_share <= 0.10,
        "frequency_asymmetry": min_mean > 0 and (max_mean / min_mean) >= 2.0,
        "positive_slope_count": sum(
            1 for band in BAND_ORDER if bands[band]["p_value_one_sided"] < 0.05
        ),
    }


def one_sided_positive_p_value(slope: float, p_value_two_sided: float) -> float:
    if slope > 0:
        return float(p_value_two_sided / 2.0)
    return float(1.0 - p_value_two_sided / 2.0)


def safe_ratio(numerator: float, denominator: float) -> float:
    numerator = float(numerator)
    denominator = float(denominator)
    if abs(denominator) < 1e-12:
        return float("inf") if abs(numerator) >= 1e-12 else 1.0
    return numerator / denominator


def yes_no(value: bool) -> str:
    return "YES" if value else "NO"


def print_generated_files() -> None:
    print()
    print("Generated figures:")
    for path in (LATENT_FIG, LATENT_FIG_LOG, OBS_FIG, OBS_FIG_LOG):
        size_kib = path.stat().st_size / 1024.0
        print(f"  {path.relative_to(REPO_ROOT)} ({size_kib:.1f} KiB)")


if __name__ == "__main__":
    main()
