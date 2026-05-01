"""Generate Figure 2: V3 vs V4 frequency-band drift comparison."""

from __future__ import annotations

import json
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

sys.path.insert(0, str(REPO_ROOT))

from src.diagnostics.error_curves import aggregate_curves  # noqa: E402
from src.diagnostics.freq_decompose import DEFAULT_BANDS  # noqa: E402


V3_ROLLOUT_DIR = REPO_ROOT / "results" / "rollouts"
V4_TABLE = REPO_ROOT / "results" / "tables" / "v4_band_errors.json"
FIGURE_DIR = REPO_ROOT / "results" / "figures"
OUTPUT_PATH = FIGURE_DIR / "figure2_v3_vs_v4_v1.pdf"

N_SEEDS = 20
HORIZON = 200
PLOT_START = 25
PLOT_END = 175
VAR_THRESHOLD = 0.95

BAND_ORDER = list(DEFAULT_BANDS.keys())
BAND_LABELS = {
    "dc_trend": "DC/Trend",
    "very_low": "Very Low",
    "low": "Low",
    "mid": "Mid",
    "high": "High",
}
BAND_COLORS = {
    "dc_trend": "#7f1d1d",
    "very_low": "#d97706",
    "low": "#1d4ed8",
    "mid": "#15803d",
    "high": "#7c3aed",
}


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    v4 = load_v4_table()
    v4_latent = v4["latent-space"]

    print("Computing V3 per-band curves from v2 rollouts...")
    v3_curves = aggregate_curves(
        rollout_paths=collect_v3_rollout_paths(),
        signal_key_true="true_latent",
        signal_key_imag="imagined_latent",
        metric="mse",
        var_threshold=VAR_THRESHOLD,
    )
    v3_summary = summarize_curves(v3_curves)
    v4_summary = summarize_json_section(v4_latent)

    plot_figure(v3_curves, v4_latent, v3_summary, v4_summary)

    size_kib = OUTPUT_PATH.stat().st_size / 1024.0
    print(f"Saved {OUTPUT_PATH.relative_to(REPO_ROOT)} ({size_kib:.1f} KiB)")
    print("V3 ranking:", " > ".join(v3_summary["ranking"]))
    print("V4 ranking:", v4_latent["ranking_text"])


def collect_v3_rollout_paths() -> list[Path]:
    paths = [V3_ROLLOUT_DIR / f"cheetah_v3_seed{seed}_v2.npz" for seed in range(N_SEEDS)]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing V3 rollouts: " + ", ".join(str(path) for path in missing))
    return paths


def load_v4_table() -> dict[str, Any]:
    if not V4_TABLE.exists():
        raise FileNotFoundError(
            f"Missing {V4_TABLE.relative_to(REPO_ROOT)}. Run scripts/run_v4_freq_decompose.py first."
        )
    return json.loads(V4_TABLE.read_text())


def plot_figure(
    v3_curves: dict[str, Any],
    v4_latent: dict[str, Any],
    v3_summary: dict[str, Any],
    v4_summary: dict[str, Any],
) -> None:
    fig = plt.figure(figsize=(10, 5.8), dpi=300, constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.25, 1.0], wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])

    x = np.arange(HORIZON)
    plot_slice = slice(PLOT_START, PLOT_END + 1)
    x_plot = x[plot_slice]
    floor = positive_plot_floor(v3_curves, v4_latent, plot_slice)

    for band in BAND_ORDER:
        color = BAND_COLORS[band]
        v3_mean = np.clip(np.asarray(v3_curves[band]["mean"], dtype=np.float64), floor, None)
        v3_low = np.clip(np.asarray(v3_curves[band]["ci_low"], dtype=np.float64), floor, None)
        v3_high = np.clip(np.asarray(v3_curves[band]["ci_high"], dtype=np.float64), floor, None)

        v4_band = v4_latent["bands"][band]
        v4_mean = np.clip(np.asarray(v4_band["curve_mean"], dtype=np.float64), floor, None)
        v4_low = np.clip(np.asarray(v4_band["curve_ci95_low"], dtype=np.float64), floor, None)
        v4_high = np.clip(np.asarray(v4_band["curve_ci95_high"], dtype=np.float64), floor, None)

        ax.plot(
            x_plot,
            v3_mean[plot_slice],
            color=color,
            linewidth=1.5,
            linestyle="-",
            label=f"{BAND_LABELS[band]} V3",
        )
        ax.fill_between(
            x_plot,
            v3_low[plot_slice],
            v3_high[plot_slice],
            color=color,
            alpha=0.12,
            linewidth=0,
        )
        ax.plot(
            x_plot,
            v4_mean[plot_slice],
            color=color,
            linewidth=1.7,
            linestyle="--",
            label=f"{BAND_LABELS[band]} V4",
        )
        ax.fill_between(
            x_plot,
            v4_low[plot_slice],
            v4_high[plot_slice],
            color=color,
            alpha=0.08,
            linewidth=0,
        )

    ax.set_yscale("log")
    ax.set_xlim(PLOT_START, PLOT_END)
    ax.set_xlabel("Imagination Horizon Step", fontsize=11)
    ax.set_ylabel("Per-Band MSE", fontsize=11)
    ax.set_title("Cross-Architecture Frequency-Band Error: DreamerV3 vs DreamerV4", fontsize=12)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(ncol=2, fontsize=7.5, frameon=True, loc="upper left")

    x_bar = np.arange(len(BAND_ORDER))
    width = 0.36
    v3_shares = [v3_summary["shares"][band] * 100.0 for band in BAND_ORDER]
    v4_shares = [v4_summary["shares"][band] * 100.0 for band in BAND_ORDER]
    ax_bar.bar(x_bar - width / 2, v3_shares, width, color="#4b5563", label="V3")
    ax_bar.bar(x_bar + width / 2, v4_shares, width, color="#2563eb", label="V4")
    ax_bar.set_xticks(x_bar)
    ax_bar.set_xticklabels([BAND_LABELS[band] for band in BAND_ORDER], rotation=35, ha="right")
    ax_bar.set_ylabel("Band Share (%)", fontsize=10)
    ax_bar.set_title("Trim-Window Share", fontsize=11)
    ax_bar.legend(frameon=True, fontsize=8)
    ax_bar.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)


def summarize_curves(result: dict[str, Any]) -> dict[str, Any]:
    stats_slice = slice(PLOT_START, PLOT_END + 1)
    means = {
        band: float(np.asarray(result[band]["mean"], dtype=np.float64)[stats_slice].mean())
        for band in BAND_ORDER
    }
    total = float(sum(means.values()))
    shares = {band: means[band] / total if total > 0 else 0.0 for band in BAND_ORDER}
    ranking = sorted(BAND_ORDER, key=lambda band: means[band], reverse=True)
    return {"means": means, "shares": shares, "ranking": ranking}


def summarize_json_section(section: dict[str, Any]) -> dict[str, Any]:
    means = {
        band: float(section["bands"][band]["window_mean_mse"])
        for band in BAND_ORDER
    }
    shares = {
        band: float(section["bands"][band]["share"])
        for band in BAND_ORDER
    }
    ranking = list(section["ranking"])
    return {"means": means, "shares": shares, "ranking": ranking}


def positive_plot_floor(
    v3_curves: dict[str, Any],
    v4_latent: dict[str, Any],
    plot_slice: slice,
) -> float:
    positive: list[float] = []
    for band in BAND_ORDER:
        for key in ("mean", "ci_low", "ci_high"):
            values = np.asarray(v3_curves[band][key], dtype=np.float64)[plot_slice]
            positive.extend(values[values > 0].tolist())
        for key in ("curve_mean", "curve_ci95_low", "curve_ci95_high"):
            values = np.asarray(v4_latent["bands"][band][key], dtype=np.float64)[plot_slice]
            positive.extend(values[values > 0].tolist())
    if not positive:
        return 1e-10
    return max(1e-10, min(positive) * 0.5)


if __name__ == "__main__":
    main()
