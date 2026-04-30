"""Generate Cartpole Figure 1 and cross-task DriftDetect comparisons."""

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

sys.path.insert(0, str(REPO_ROOT))

from src.analysis.figure1 import (  # noqa: E402
    BAND_ORDER,
    HORIZON,
    VAR_THRESHOLD,
    plot_result,
    print_summary,
    summarize_result,
    yes_no,
)
from src.diagnostics.error_curves import aggregate_curves  # noqa: E402


ROLLOUT_DIR = REPO_ROOT / "results" / "rollouts"
FIGURE_DIR = REPO_ROOT / "results" / "figures"

LATENT_FIG = FIGURE_DIR / "figure1_cartpole_latent_v1.pdf"
LATENT_FIG_LOG = FIGURE_DIR / "figure1_cartpole_latent_v1_log.pdf"
OBS_FIG = FIGURE_DIR / "figure1_cartpole_obs_v1.pdf"
OBS_FIG_LOG = FIGURE_DIR / "figure1_cartpole_obs_v1_log.pdf"

N_SEEDS = 20
STATS_WINDOW_LABEL = "[25:175]"


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    cartpole_paths = collect_rollout_paths("cartpole")
    cheetah_paths = collect_rollout_paths("cheetah")

    cartpole_latent = aggregate_curves(
        rollout_paths=cartpole_paths,
        signal_key_true="true_latent",
        signal_key_imag="imagined_latent",
        metric="mse",
        var_threshold=VAR_THRESHOLD,
    )
    cartpole_obs = aggregate_curves(
        rollout_paths=cartpole_paths,
        signal_key_true="true_obs",
        signal_key_imag="imagined_obs",
        metric="mse",
        var_threshold=VAR_THRESHOLD,
    )

    plot_result(
        cartpole_latent,
        LATENT_FIG,
        title="DreamerV3 Cartpole Swingup - Latent-Space Drift by Frequency Band",
        ylabel="Per-Band MSE",
        annotation="20 seeds, 95% CI, Butterworth order 4, PCA 95% var",
        log_scale=False,
    )
    plot_result(
        cartpole_latent,
        LATENT_FIG_LOG,
        title="DreamerV3 Cartpole Swingup - Latent-Space Drift by Frequency Band",
        ylabel="Per-Band MSE",
        annotation="20 seeds, 95% CI, Butterworth order 4, PCA 95% var",
        log_scale=True,
    )
    plot_result(
        cartpole_obs,
        OBS_FIG,
        title="DreamerV3 Cartpole Swingup - Obs-Space Drift by Frequency Band",
        ylabel="Per-Band MSE (obs)",
        annotation="20 seeds, 95% CI, Butterworth order 4, no PCA",
        log_scale=False,
    )
    plot_result(
        cartpole_obs,
        OBS_FIG_LOG,
        title="DreamerV3 Cartpole Swingup - Obs-Space Drift by Frequency Band",
        ylabel="Per-Band MSE (obs)",
        annotation="20 seeds, 95% CI, Butterworth order 4, no PCA",
        log_scale=True,
    )

    print("=== CARTPOLE FIGURE 1 ===")
    print()
    print_summary(cartpole_latent, cartpole_obs)
    print_generated_files()
    print()

    cheetah_latent = aggregate_curves(
        rollout_paths=cheetah_paths,
        signal_key_true="true_latent",
        signal_key_imag="imagined_latent",
        metric="mse",
        var_threshold=VAR_THRESHOLD,
    )
    cheetah_obs = aggregate_curves(
        rollout_paths=cheetah_paths,
        signal_key_true="true_obs",
        signal_key_imag="imagined_obs",
        metric="mse",
        var_threshold=VAR_THRESHOLD,
    )

    print_cross_task_comparison(
        cheetah_latent,
        cartpole_latent,
        space_label="latent-space",
    )
    print()
    print_cross_task_comparison(
        cheetah_obs,
        cartpole_obs,
        space_label="obs-space",
    )


def collect_rollout_paths(task_name: str) -> list[Path]:
    if task_name == "cartpole":
        pattern = "cartpole_v3_seed{seed}_v2.npz"
    elif task_name == "cheetah":
        pattern = "cheetah_v3_seed{seed}_v2.npz"
    else:
        raise ValueError(f"Unknown task_name {task_name!r}.")

    paths = [ROLLOUT_DIR / pattern.format(seed=seed) for seed in range(N_SEEDS)]
    missing = [path for path in paths if not path.exists()]
    if missing:
        missing_str = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing rollout files: {missing_str}")
    return paths


def print_generated_files() -> None:
    print()
    print("Generated Cartpole figures:")
    for path in (LATENT_FIG, LATENT_FIG_LOG, OBS_FIG, OBS_FIG_LOG):
        size_kib = path.stat().st_size / 1024.0
        print(f"  {path.relative_to(REPO_ROOT)} ({size_kib:.1f} KiB)")


def print_cross_task_comparison(
    cheetah_result: dict[str, Any],
    cartpole_result: dict[str, Any],
    *,
    space_label: str,
) -> None:
    cheetah_summary = summarize_result(cheetah_result)
    cartpole_summary = summarize_result(cartpole_result)
    cheetah_rank = cheetah_summary["ranking"]
    cartpole_rank = cartpole_summary["ranking"]
    rankings_identical = cheetah_rank == cartpole_rank
    cheetah_positions = {band: index for index, band in enumerate(cheetah_rank)}
    cartpole_positions = {band: index for index, band in enumerate(cartpole_rank)}

    print(
        "=== CROSS-TASK COMPARISON "
        f"({space_label}, mean MSE {STATS_WINDOW_LABEL}) ==="
    )
    print()
    print("Band          Cheetah      Cartpole     Same Ranking?")
    for band in BAND_ORDER:
        cheetah_mean = cheetah_summary["bands"][band]["window_mean"]
        cartpole_mean = cartpole_summary["bands"][band]["window_mean"]
        same_rank_position = cheetah_positions[band] == cartpole_positions[band]
        print(
            f"{band:<12}"
            f"{cheetah_mean:>8.4f}      "
            f"{cartpole_mean:>8.4f}     "
            f"{yes_no(same_rank_position):>3}"
        )
    print()
    print(f"Cheetah ranking:  {cheetah_summary['ranking_text']}")
    print(f"Cartpole ranking: {cartpole_summary['ranking_text']}")
    print(f"Rankings identical: {yes_no(rankings_identical)}")
    print()

    dc_dominant_both = (
        cheetah_summary["dominant_band"] == "dc_trend"
        and cartpole_summary["dominant_band"] == "dc_trend"
    )
    high_suppressed_both = (
        cheetah_summary["high_suppressed"]
        and cartpole_summary["high_suppressed"]
    )
    if dc_dominant_both and high_suppressed_both and rankings_identical:
        conclusion = "SAME_PATTERN"
    elif dc_dominant_both or high_suppressed_both or rankings_identical:
        conclusion = "PARTIAL_OVERLAP"
    else:
        conclusion = "DIFFERENT_PATTERN"

    print("Cross-task Finding A assessment:")
    print(f"  - dc_trend dominant in both tasks: {yes_no(dc_dominant_both)}")
    print(f"  - high-freq suppressed in both tasks: {yes_no(high_suppressed_both)}")
    print(f"  - Band ranking identical: {yes_no(rankings_identical)}")
    print(f"  - Conclusion: {conclusion}")


if __name__ == "__main__":
    main()
