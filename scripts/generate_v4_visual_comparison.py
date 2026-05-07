"""Generate V4 baseline-vs-SHARP visual imagination comparison.

Uses pre-extracted V4 rollouts when available; otherwise re-runs extraction
through ``DreamerV4Adapter`` against the supplied checkpoints.

The DreamerV4 adapter's ``extract_rollout`` already decodes imagined latents
back to ``(T, 3, 128, 128)`` float frames in ``[0, 1]``, so the rollout
``.npz`` files contain everything needed for pixel-level visualization.

Outputs (all 300 dpi, written to ``results/figures/``):
- fig5_v4_frame_comparison.png  : 3x8 frame grid (True / Baseline / SHARP).
- fig5_v4_imagination.gif       : 200-frame triptych animation.
- fig6_v4_recon_error.png       : per-step decoded-frame MSE.
- fig7_v4_latent_drift.png      : per-step latent (packed flat) MSE.

Run on RunPod from the repo root:

    PYTHONPATH=external/dreamer4/dreamer4:external/dreamer4:$PYTHONPATH \
    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa WANDB_MODE=disabled \
    python scripts/generate_v4_visual_comparison.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "results" / "figures"

DEFAULT_BASELINE_NPZ = (
    REPO_ROOT
    / "results"
    / "month8_controlled"
    / "rollouts_v4_baseline"
    / "cheetah-run_v4_seed0_v1.npz"
)
DEFAULT_SHARP_NPZ = (
    REPO_ROOT
    / "results"
    / "month8_controlled"
    / "rollouts_v4_sharp"
    / "cheetah-run_v4_seed0_v1.npz"
)

DEFAULT_BASELINE_CKPT = REPO_ROOT / "external" / "dreamer4" / "checkpoints"
DEFAULT_SHARP_CKPT = REPO_ROOT / "results" / "v4_sharp" / "for_eval"
DEFAULT_DATASET_DIR = REPO_ROOT / "external" / "dreamer4" / "data"
DEFAULT_TOKENIZER_PATH = (
    REPO_ROOT / "external" / "dreamer4" / "checkpoints" / "tokenizer.pt"
)

GRID_STEPS = [0, 5, 15, 30, 50, 100, 150, 199]
N_STEPS = 200
ACTOR_HORIZON = 15
GIF_FPS = 15

COLOR_BASELINE = "#E53935"
COLOR_SHARP = "#1E88E5"


@dataclass
class V4Rollout:
    true_obs: np.ndarray
    imagined_obs: np.ndarray
    true_latent: np.ndarray
    imagined_latent: np.ndarray


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.color": "lightgray",
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
    })


def to_uint8_image(frame: np.ndarray) -> np.ndarray:
    """Convert a (3, H, W) float frame in [0, 1] to a (H, W, 3) uint8 image."""

    if frame.ndim != 3 or frame.shape[0] != 3:
        raise ValueError(f"expected (3,H,W) frame, got shape {frame.shape}")
    arr = np.clip(frame, 0.0, 1.0)
    arr = (arr * 255.0 + 0.5).astype(np.uint8)
    return np.transpose(arr, (1, 2, 0))


def load_existing_rollout(path: Path) -> V4Rollout:
    with np.load(path, allow_pickle=True) as data:
        true_obs = np.asarray(data["true_obs"])
        imagined_obs = np.asarray(data["imagined_obs"])
        true_latent = np.asarray(data["true_latent"]) if "true_latent" in data.files else None
        imagined_latent = (
            np.asarray(data["imagined_latent"]) if "imagined_latent" in data.files else None
        )
    if true_latent is None or imagined_latent is None:
        raise RuntimeError(
            f"{path} does not contain true_latent/imagined_latent — re-extract with include_latent=True."
        )
    if true_obs.shape[0] < N_STEPS or imagined_obs.shape[0] < N_STEPS:
        raise RuntimeError(
            f"{path} has only {true_obs.shape[0]} steps, need at least {N_STEPS}."
        )
    return V4Rollout(
        true_obs=true_obs[:N_STEPS].astype(np.float32),
        imagined_obs=imagined_obs[:N_STEPS].astype(np.float32),
        true_latent=true_latent[:N_STEPS].astype(np.float32),
        imagined_latent=imagined_latent[:N_STEPS].astype(np.float32),
    )


def _load_adapter_class():
    path = REPO_ROOT / "src" / "models" / "dreamerv4_adapter.py"
    spec = importlib.util.spec_from_file_location("_driftdetect_v4_adapter_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import DreamerV4Adapter from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.DreamerV4Adapter


def extract_rollout_via_adapter(
    *,
    checkpoint_dir: Path,
    dataset_dir: Path,
    task: str,
    seed: int,
    context_length: int,
    horizon: int,
    tokenizer_override: Path | None = None,
) -> V4Rollout:
    DreamerV4Adapter = _load_adapter_class()
    adapter = DreamerV4Adapter(task=task)
    adapter.load_checkpoint(
        str(checkpoint_dir),
        tokenizer_path=str(tokenizer_override) if tokenizer_override is not None else None,
    )
    rollout: dict[str, Any] = adapter.extract_rollout(
        dataset_path=dataset_dir,
        task_name=task,
        seed=seed,
        context_length=context_length,
        horizon=horizon,
        include_latent=True,
    )
    return V4Rollout(
        true_obs=np.asarray(rollout["true_obs"])[:N_STEPS].astype(np.float32),
        imagined_obs=np.asarray(rollout["imagined_obs"])[:N_STEPS].astype(np.float32),
        true_latent=np.asarray(rollout["true_latent"])[:N_STEPS].astype(np.float32),
        imagined_latent=np.asarray(rollout["imagined_latent"])[:N_STEPS].astype(np.float32),
    )


def get_rollouts(args: argparse.Namespace) -> tuple[V4Rollout, V4Rollout]:
    baseline_npz = Path(args.baseline_rollout)
    sharp_npz = Path(args.sharp_rollout)

    if baseline_npz.exists() and sharp_npz.exists() and not args.force_extract:
        print(f"loading existing baseline rollout: {baseline_npz}")
        baseline = load_existing_rollout(baseline_npz)
        print(f"loading existing SHARP rollout:    {sharp_npz}")
        sharp = load_existing_rollout(sharp_npz)
        return baseline, sharp

    print("rollout npz not found (or --force_extract); running extraction via adapter")
    baseline = extract_rollout_via_adapter(
        checkpoint_dir=Path(args.baseline_checkpoint_dir),
        dataset_dir=Path(args.dataset_dir),
        task=args.task,
        seed=args.seed,
        context_length=args.context_length,
        horizon=N_STEPS,
        tokenizer_override=None,
    )
    sharp = extract_rollout_via_adapter(
        checkpoint_dir=Path(args.sharp_checkpoint_dir),
        dataset_dir=Path(args.dataset_dir),
        task=args.task,
        seed=args.seed,
        context_length=args.context_length,
        horizon=N_STEPS,
        tokenizer_override=Path(args.tokenizer_path),
    )
    return baseline, sharp


def figure5_frame_grid(baseline: V4Rollout, sharp: V4Rollout) -> None:
    rows = ("True", "Baseline", "SHARP")
    n_cols = len(GRID_STEPS)

    fig, axes = plt.subplots(
        len(rows),
        n_cols,
        figsize=(1.6 * n_cols, 1.7 * len(rows) + 0.6),
        gridspec_kw={"wspace": 0.05, "hspace": 0.08},
    )

    for col, step in enumerate(GRID_STEPS):
        true_img = to_uint8_image(sharp.true_obs[step])
        base_img = to_uint8_image(baseline.imagined_obs[step])
        sharp_img = to_uint8_image(sharp.imagined_obs[step])

        for row_idx, img in enumerate((true_img, base_img, sharp_img)):
            ax = axes[row_idx, col]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        axes[0, col].set_title(f"t={step}", fontsize=10)

    for row_idx, label in enumerate(rows):
        axes[row_idx, 0].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=12,
                                    labelpad=18)

    fig.suptitle("V4 Imagination Quality: True vs Baseline vs SHARP", y=0.99)

    out = FIG_DIR / "fig5_v4_frame_comparison.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


def _label_strip(width: int, label: str, *, height: int = 14) -> np.ndarray:
    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.text(0.5, 0.5, label, ha="center", va="center", fontsize=9, family="serif")
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return rgba[..., :3].copy()


def figure5_imagination_gif(baseline: V4Rollout, sharp: V4Rollout) -> None:
    panel_w = baseline.imagined_obs.shape[-1]
    panel_h = baseline.imagined_obs.shape[-2]
    label_h = 16

    label_true = _label_strip(panel_w, "True", height=label_h)
    label_base = _label_strip(panel_w, "Baseline", height=label_h)
    label_sharp = _label_strip(panel_w, "SHARP", height=label_h)
    header = np.concatenate([label_true, label_base, label_sharp], axis=1)

    frames: list[Image.Image] = []
    for step in range(N_STEPS):
        true_img = to_uint8_image(sharp.true_obs[step])
        base_img = to_uint8_image(baseline.imagined_obs[step])
        sharp_img = to_uint8_image(sharp.imagined_obs[step])
        body = np.concatenate([true_img, base_img, sharp_img], axis=1)

        if header.shape[1] != body.shape[1]:
            header_resized = np.array(
                Image.fromarray(header).resize((body.shape[1], header.shape[0]), Image.BILINEAR)
            )
        else:
            header_resized = header

        composite = np.concatenate([header_resized, body], axis=0)
        frames.append(Image.fromarray(composite))

    out = FIG_DIR / "fig5_v4_imagination.gif"
    duration_ms = max(1, int(round(1000.0 / GIF_FPS)))
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )
    print(f"wrote {out} ({len(frames)} frames @ {GIF_FPS} fps, panel size {panel_w}x{panel_h})")


def figure6_recon_error(baseline: V4Rollout, sharp: V4Rollout) -> None:
    base_se = (baseline.imagined_obs - sharp.true_obs) ** 2
    sharp_se = (sharp.imagined_obs - sharp.true_obs) ** 2
    base_mse = base_se.reshape(N_STEPS, -1).mean(axis=1)
    sharp_mse = sharp_se.reshape(N_STEPS, -1).mean(axis=1)

    steps = np.arange(N_STEPS)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, base_mse, color=COLOR_BASELINE, linestyle="--", linewidth=1.6,
            label="Baseline DreamerV4")
    ax.plot(steps, sharp_mse, color=COLOR_SHARP, linewidth=1.6, label="SHARP V4")
    ax.axvline(ACTOR_HORIZON, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ymin, ymax = ax.get_ylim()
    ax.text(
        ACTOR_HORIZON + 2.0,
        ymin + 0.93 * (ymax - ymin),
        "Actor horizon",
        color="gray",
        fontsize=9,
        alpha=0.85,
    )

    ax.set_xlabel("Imagination step")
    ax.set_ylabel("Per-step decoded frame MSE")
    ax.set_title("V4 Decoded-Frame Reconstruction Error: Baseline vs SHARP (seed 0)")
    ax.legend(loc="upper left", frameon=False)

    out = FIG_DIR / "fig6_v4_recon_error.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


def figure7_latent_drift(baseline: V4Rollout, sharp: V4Rollout) -> None:
    base_mse = ((baseline.imagined_latent - baseline.true_latent) ** 2).mean(axis=1)
    sharp_mse = ((sharp.imagined_latent - sharp.true_latent) ** 2).mean(axis=1)

    steps = np.arange(N_STEPS)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(steps, base_mse, color=COLOR_BASELINE, linestyle="--", linewidth=1.7,
            label="Baseline DreamerV4")
    ax.plot(steps, sharp_mse, color=COLOR_SHARP, linewidth=1.7, label="SHARP V4")

    ax.axvline(ACTOR_HORIZON, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ymin, ymax = ax.get_ylim()
    ax.text(
        ACTOR_HORIZON + 2.0,
        ymin + 0.93 * (ymax - ymin),
        "Actor horizon",
        color="gray",
        fontsize=9,
        alpha=0.85,
    )

    target = 150
    sharp_y = sharp_mse[target]
    base_y = base_mse[target]
    text_y = (base_y + sharp_y) * 0.5
    ax.annotate(
        "J_total: -64.1%",
        xy=(target, sharp_y),
        xytext=(target - 60, text_y),
        fontsize=11,
        color=COLOR_SHARP,
        arrowprops=dict(
            arrowstyle="->",
            color=COLOR_SHARP,
            lw=1.2,
            connectionstyle="arc3,rad=-0.15",
        ),
    )

    ax.set_xlabel("Imagination step")
    ax.set_ylabel("Latent MSE (packed flat, 512 dims)")
    ax.set_title("V4 Latent Drift Accumulation: Baseline vs SHARP (seed 0)")
    ax.legend(loc="upper left", frameon=False)

    out = FIG_DIR / "fig7_v4_latent_drift.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_rollout", default=str(DEFAULT_BASELINE_NPZ))
    parser.add_argument("--sharp_rollout", default=str(DEFAULT_SHARP_NPZ))
    parser.add_argument("--baseline_checkpoint_dir", default=str(DEFAULT_BASELINE_CKPT))
    parser.add_argument("--sharp_checkpoint_dir", default=str(DEFAULT_SHARP_CKPT))
    parser.add_argument("--tokenizer_path", default=str(DEFAULT_TOKENIZER_PATH))
    parser.add_argument("--dataset_dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--task", default="cheetah-run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--context_length", type=int, default=8)
    parser.add_argument(
        "--force_extract",
        action="store_true",
        help="Ignore existing rollout npz files and re-extract via adapter.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    baseline, sharp = get_rollouts(args)
    print(
        f"baseline true_obs={baseline.true_obs.shape}, imagined_obs={baseline.imagined_obs.shape}, "
        f"latent={baseline.imagined_latent.shape}"
    )
    print(
        f"sharp    true_obs={sharp.true_obs.shape}, imagined_obs={sharp.imagined_obs.shape}, "
        f"latent={sharp.imagined_latent.shape}"
    )

    if not np.allclose(baseline.true_obs, sharp.true_obs, atol=1e-4):
        diff = float(np.abs(baseline.true_obs - sharp.true_obs).max())
        print(
            f"warning: baseline.true_obs and sharp.true_obs differ (max abs diff {diff:.4g}); "
            "using sharp.true_obs as the reference trajectory."
        )

    figure5_frame_grid(baseline, sharp)
    figure5_imagination_gif(baseline, sharp)
    figure6_recon_error(baseline, sharp)
    figure7_latent_drift(baseline, sharp)


if __name__ == "__main__":
    main()
