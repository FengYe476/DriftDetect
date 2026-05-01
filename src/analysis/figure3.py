"""Generate Figure 3 for DreamerV3 Cheetah training dynamics."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "driftdetect-matplotlib-cache"),
)

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

sys.path.insert(0, str(REPO_ROOT))


DEFAULT_INPUT = REPO_ROOT / "results" / "tables" / "v3_cheetah_training_dynamics.npz"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "figures"

CURVES_FIG = "figure3_curves_v1.pdf"
HEATMAP_FIG = "figure3_heatmap_v1.pdf"
SHARE_FIG = "figure3_share_v1.pdf"

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
SUMMARY_STEPS = (5000, 250000, 510000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Figure 3 training-dynamics visualizations."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input training dynamics .npz file.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated figure PDFs.",
    )
    return parser.parse_args()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_training_dynamics(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing training dynamics input: {display_path(path)}")

    with np.load(path, allow_pickle=True) as data:
        required = ("training_steps", "band_mse", "band_names")
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"{display_path(path)} is missing keys: {missing}")

        training_steps = np.asarray(data["training_steps"], dtype=np.int64)
        band_mse = np.asarray(data["band_mse"], dtype=np.float64)
        band_names = [str(name) for name in data["band_names"].tolist()]

    if training_steps.ndim != 1:
        raise ValueError(f"training_steps must be 1D, got {training_steps.shape}.")
    if band_mse.ndim != 2:
        raise ValueError(f"band_mse must be 2D, got {band_mse.shape}.")
    if band_mse.shape != (training_steps.size, len(band_names)):
        raise ValueError(
            "Expected band_mse shape "
            f"({training_steps.size}, {len(band_names)}), got {band_mse.shape}."
        )
    if np.any(band_mse < 0):
        raise ValueError("band_mse contains negative values.")

    order = np.argsort(training_steps)
    return training_steps[order], band_mse[order], band_names


def step_formatter(value: float, _pos: int | None = None) -> str:
    if value == 0:
        return "0"
    return f"{int(round(value / 1000.0))}k"


def set_step_ticks(ax: plt.Axes, training_steps: np.ndarray) -> None:
    tick_indices = np.arange(0, training_steps.size, 10)
    if tick_indices[-1] != training_steps.size - 1:
        tick_indices = np.append(tick_indices, training_steps.size - 1)
    ax.set_xticks(training_steps[tick_indices])
    ax.xaxis.set_major_formatter(FuncFormatter(step_formatter))


def compute_share(band_mse: np.ndarray) -> np.ndarray:
    totals = band_mse.sum(axis=1, keepdims=True)
    return np.divide(
        band_mse,
        totals,
        out=np.zeros_like(band_mse, dtype=np.float64),
        where=totals > 0,
    )


def positive_floor(values: np.ndarray) -> float:
    positive = values[values > 0]
    if positive.size == 0:
        return 1e-8
    return max(1e-8, float(positive.min()) * 0.5)


def plot_curves(
    training_steps: np.ndarray,
    band_mse: np.ndarray,
    band_names: list[str],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    for band_index, band in enumerate(band_names):
        ax.plot(
            training_steps,
            band_mse[:, band_index],
            color=BAND_COLORS.get(band),
            linewidth=1.8,
            marker="o",
            markersize=3.5,
            markevery=10,
            label=BAND_LABELS.get(band, band),
        )

    ax.set_yscale("log")
    ax.set_ylim(bottom=positive_floor(band_mse))
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Latent MSE")
    ax.set_title("Per-Band Latent MSE Over Training (Cheetah Run)")
    set_step_ticks(ax, training_steps)
    ax.legend(loc="best", frameon=True, fontsize=9)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(
        0.99,
        0.04,
        "Log-scale y-axis; markers shown every 10 checkpoints",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={
            "facecolor": "#f3f4f6",
            "edgecolor": "#d1d5db",
            "alpha": 0.75,
            "boxstyle": "round,pad=0.25",
        },
    )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(
    training_steps: np.ndarray,
    band_mse: np.ndarray,
    band_names: list[str],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4), dpi=300)
    image = ax.imshow(
        band_mse.T,
        aspect="auto",
        interpolation="nearest",
        cmap="inferno",
        origin="upper",
    )

    tick_indices = np.arange(0, training_steps.size, 10)
    if tick_indices[-1] != training_steps.size - 1:
        tick_indices = np.append(tick_indices, training_steps.size - 1)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([step_formatter(step) for step in training_steps[tick_indices]])
    ax.set_yticks(np.arange(len(band_names)))
    ax.set_yticklabels([BAND_LABELS.get(band, band) for band in band_names])
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Frequency Band")
    ax.set_title("Drift Error by Frequency Band and Training Step")

    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Mean Latent MSE")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_share(
    training_steps: np.ndarray,
    share: np.ndarray,
    band_names: list[str],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    colors = [BAND_COLORS.get(band, "#6b7280") for band in band_names]
    labels = [BAND_LABELS.get(band, band) for band in band_names]
    ax.stackplot(
        training_steps,
        share.T,
        colors=colors,
        labels=labels,
        alpha=0.7,
        linewidth=0.4,
        edgecolor="#ffffff",
    )
    ax.axhline(
        0.5,
        color="#111827",
        linestyle="--",
        linewidth=1.1,
        label="Majority threshold",
    )

    ax.set_xlim(training_steps[0], training_steps[-1])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Fractional Share of Total MSE")
    ax.set_title("Frequency Band Share Over Training (Cheetah Run)")
    set_step_ticks(ax, training_steps)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True, fontsize=9)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def nearest_step_index(training_steps: np.ndarray, target_step: int) -> int:
    return int(np.argmin(np.abs(training_steps - target_step)))


def ranking(values: np.ndarray, band_names: list[str]) -> list[str]:
    order = np.argsort(values)[::-1]
    return [band_names[index] for index in order]


def print_checkpoint_summary(
    training_steps: np.ndarray,
    band_mse: np.ndarray,
    share: np.ndarray,
    band_names: list[str],
    target_step: int,
) -> None:
    index = nearest_step_index(training_steps, target_step)
    step = int(training_steps[index])
    rank = ranking(band_mse[index], band_names)
    dc_index = band_names.index("dc_trend")
    dc_share = float(share[index, dc_index])
    total_mse = float(band_mse[index].sum())

    print(f"Step {step_formatter(step)}:")
    print(f"  Band ranking: {' > '.join(rank)}")
    print(f"  dc_trend share: {dc_share * 100:.1f}%")
    print(f"  Total MSE: {total_mse:.6f}")


def print_summary(
    training_steps: np.ndarray,
    band_mse: np.ndarray,
    share: np.ndarray,
    band_names: list[str],
) -> None:
    print("=== FIGURE 3 TRAINING DYNAMICS SUMMARY ===")
    print()
    for target_step in SUMMARY_STEPS:
        print_checkpoint_summary(
            training_steps,
            band_mse,
            share,
            band_names,
            target_step,
        )
        print()

    dc_index = band_names.index("dc_trend")
    dominant_indices = np.argmax(band_mse, axis=1)
    dc_dominant_count = int(np.sum(dominant_indices == dc_index))
    dc_always_dominant = dc_dominant_count == training_steps.size
    print(
        "dc_trend #1 checkpoints: "
        f"{dc_dominant_count}/{training_steps.size}"
    )
    print(f"dc_trend always #1: {'YES' if dc_always_dominant else 'NO'}")


def print_generated_files(output_dir: Path) -> None:
    print()
    print("Generated figures:")
    for name in (CURVES_FIG, HEATMAP_FIG, SHARE_FIG):
        print(f"  - {display_path(output_dir / name)}")


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()

    plt.style.use("seaborn-v0_8-whitegrid")
    output_dir.mkdir(parents=True, exist_ok=True)

    training_steps, band_mse, band_names = load_training_dynamics(input_path)
    share = compute_share(band_mse)

    plot_curves(
        training_steps,
        band_mse,
        band_names,
        output_dir / CURVES_FIG,
    )
    plot_heatmap(
        training_steps,
        band_mse,
        band_names,
        output_dir / HEATMAP_FIG,
    )
    plot_share(
        training_steps,
        share,
        band_names,
        output_dir / SHARE_FIG,
    )

    print_summary(training_steps, band_mse, share, band_names)
    print_generated_files(output_dir)


if __name__ == "__main__":
    main()
