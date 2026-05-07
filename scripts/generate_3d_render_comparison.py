"""Render V3 Cheetah imagination rollouts in 3D via DMControl.

Each step's 17-dim observation vector is set back into the dm_control cheetah
physics state (``qpos[1:] = obs[:8]``, ``qvel[:] = obs[8:17]``), then rendered
through the MuJoCo OpenGL backend. Three trajectories are rendered in parallel
for a single seed:

- Ground truth (from ``true_obs``)
- Baseline DreamerV3 imagination (from baseline ``imagined_obs``)
- SHARP imagination (from SHARP ``imagined_obs``)

Outputs (300 dpi, ``results/figures/``):
- ``fig9_3d_render_comparison.png``      : 3x8 grid (True / Baseline / SHARP).
- ``fig9_3d_render_comparison_diff.png`` : 4x8 grid with a |True-SHARP| row.
- ``fig9_3d_render_comparison.gif``      : 200-frame triptych at 15 fps.

Run on RunPod (headless OSMesa) from the repo root::

    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \\
    python scripts/generate_3d_render_comparison.py

Optional: ``--sanity_only`` saves a single default-pose render for
``test_render.png`` and exits without touching the rollouts.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("WANDB_MODE", "disabled")

import argparse
from pathlib import Path
from typing import Sequence

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
    / "rollouts_drifthead_nocorr"
    / "cheetah_v3_seed0_v2.npz"
)
DEFAULT_SHARP_NPZ = (
    REPO_ROOT
    / "results"
    / "month8_controlled"
    / "rollouts_moment_match"
    / "cheetah_v3_seed0_v2.npz"
)

GRID_STEPS = [0, 5, 15, 30, 50, 100, 150, 199]
N_STEPS = 200
RESOLUTION = 256
GIF_FPS = 15
DIFF_AMPLIFY = 4


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": False,
        "savefig.bbox": "tight",
    })


def load_obs(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=True) as data:
        if key not in data.files:
            raise KeyError(f"{key} not in {path}")
        arr = np.asarray(data[key])
    if arr.shape[0] < N_STEPS or arr.shape[1] != 17:
        raise ValueError(f"{path}:{key} expected (>={N_STEPS}, 17), got {arr.shape}")
    return arr[:N_STEPS].astype(np.float64)


def integrate_rootx(obs: np.ndarray, dt: float) -> np.ndarray:
    """Integrate root_x velocity (obs[:, 8]) to recover horizontal position."""

    rootx_vel = obs[:, 8]
    return np.concatenate([[0.0], np.cumsum(rootx_vel[:-1] * dt)])


def set_state_and_render(
    env,
    obs: np.ndarray,
    rootx: float,
    *,
    resolution: int,
    camera_id: int | str,
) -> np.ndarray:
    physics = env.physics
    qpos_size = physics.data.qpos.shape[0]
    qvel_size = physics.data.qvel.shape[0]

    if qpos_size - 1 + qvel_size != obs.shape[0]:
        raise ValueError(
            f"obs length {obs.shape[0]} does not match qpos[1:] ({qpos_size - 1}) "
            f"+ qvel ({qvel_size}) for this cheetah build."
        )

    new_qpos = np.zeros(qpos_size, dtype=np.float64)
    new_qpos[0] = float(rootx)
    new_qpos[1:] = obs[: qpos_size - 1]

    new_qvel = np.zeros(qvel_size, dtype=np.float64)
    new_qvel[:] = obs[qpos_size - 1 : qpos_size - 1 + qvel_size]

    physics.data.qpos[:] = new_qpos
    physics.data.qvel[:] = new_qvel
    physics.forward()

    return physics.render(height=resolution, width=resolution, camera_id=camera_id)


def render_trajectory(
    env,
    obs: np.ndarray,
    rootx: np.ndarray,
    *,
    resolution: int,
    camera_id: int | str,
    label: str,
) -> np.ndarray:
    frames = np.empty((obs.shape[0], resolution, resolution, 3), dtype=np.uint8)
    for t in range(obs.shape[0]):
        frame = set_state_and_render(
            env, obs[t], rootx[t], resolution=resolution, camera_id=camera_id
        )
        frames[t] = frame
        if t % 50 == 0 or t == obs.shape[0] - 1:
            print(f"  [{label}] rendered step {t + 1}/{obs.shape[0]}", flush=True)
    return frames


def diff_frame(true_frame: np.ndarray, other_frame: np.ndarray, amplify: int) -> np.ndarray:
    diff = np.abs(true_frame.astype(np.int16) - other_frame.astype(np.int16))
    return np.clip(diff * amplify, 0, 255).astype(np.uint8)


def figure_grid(
    *,
    rows: Sequence[tuple[str, np.ndarray]],
    steps: Sequence[int],
    title: str,
    out: Path,
) -> None:
    n_rows = len(rows)
    n_cols = len(steps)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(1.7 * n_cols, 1.7 * n_rows + 0.6),
        gridspec_kw={"wspace": 0.04, "hspace": 0.06},
    )
    if n_rows == 1:
        axes = np.array([axes])

    for col, step in enumerate(steps):
        for row_idx, (_label, frames) in enumerate(rows):
            ax = axes[row_idx, col]
            ax.imshow(frames[step])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[0, col].set_title(f"t={step}", fontsize=10)

    for row_idx, (label, _frames) in enumerate(rows):
        axes[row_idx, 0].set_ylabel(
            label, rotation=0, ha="right", va="center", fontsize=11, labelpad=24
        )

    fig.suptitle(title, y=0.99)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


def _label_strip(width: int, label: str, *, height: int = 18) -> np.ndarray:
    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.text(0.5, 0.5, label, ha="center", va="center", fontsize=10, family="serif")
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return rgba[..., :3].copy()


def write_gif(
    *,
    true_frames: np.ndarray,
    base_frames: np.ndarray,
    sharp_frames: np.ndarray,
    out: Path,
    fps: int,
) -> None:
    panel_w = true_frames.shape[2]
    label_h = 18
    header = np.concatenate(
        [
            _label_strip(panel_w, "Ground Truth", height=label_h),
            _label_strip(panel_w, "Baseline", height=label_h),
            _label_strip(panel_w, "SHARP", height=label_h),
        ],
        axis=1,
    )

    images: list[Image.Image] = []
    for t in range(true_frames.shape[0]):
        body = np.concatenate([true_frames[t], base_frames[t], sharp_frames[t]], axis=1)
        if header.shape[1] != body.shape[1]:
            header_resized = np.array(
                Image.fromarray(header).resize((body.shape[1], header.shape[0]), Image.BILINEAR)
            )
        else:
            header_resized = header
        composite = np.concatenate([header_resized, body], axis=0)
        images.append(Image.fromarray(composite))

    duration_ms = max(1, int(round(1000.0 / fps)))
    images[0].save(
        out,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )
    print(f"wrote {out} ({len(images)} frames @ {fps} fps)")


def sanity_check(env, *, resolution: int, camera_id: int | str, out: Path) -> None:
    physics = env.physics
    print(f"qpos shape: {physics.data.qpos.shape}")
    print(f"qvel shape: {physics.data.qvel.shape}")
    print(f"qpos: {physics.data.qpos.copy()}")
    print(f"control_timestep: {env.control_timestep()}")
    frame = physics.render(height=resolution, width=resolution, camera_id=camera_id)
    print(f"frame shape: {frame.shape}, dtype: {frame.dtype}")
    Image.fromarray(frame).save(out)
    print(f"wrote {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_npz", default=str(DEFAULT_BASELINE_NPZ))
    parser.add_argument("--sharp_npz", default=str(DEFAULT_SHARP_NPZ))
    parser.add_argument("--resolution", type=int, default=RESOLUTION)
    parser.add_argument("--camera_id", default="0",
                        help="Integer index or string camera name; default 0.")
    parser.add_argument("--fps", type=int, default=GIF_FPS)
    parser.add_argument("--sanity_only", action="store_true",
                        help="Render the default cheetah pose to test_render.png and exit.")
    parser.add_argument("--no_gif", action="store_true",
                        help="Skip the 200-frame GIF (faster iteration).")
    return parser


def parse_camera_id(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def main() -> None:
    args = build_parser().parse_args()
    configure_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    camera_id = parse_camera_id(args.camera_id)

    from dm_control import suite  # imported here so MUJOCO_GL takes effect first

    env = suite.load("cheetah", "run")

    if args.sanity_only:
        sanity_check(
            env,
            resolution=args.resolution,
            camera_id=camera_id,
            out=FIG_DIR / "test_render.png",
        )
        return

    qpos_size = env.physics.data.qpos.shape[0]
    qvel_size = env.physics.data.qvel.shape[0]
    print(f"cheetah physics: qpos={qpos_size}, qvel={qvel_size}")
    expected_obs = (qpos_size - 1) + qvel_size
    if expected_obs != 17:
        print(f"warning: this cheetah build expects {expected_obs}-dim obs, not 17.")

    dt = float(env.control_timestep())
    print(f"control_timestep dt={dt}")

    true_obs = load_obs(Path(args.baseline_npz), "true_obs")
    base_imag = load_obs(Path(args.baseline_npz), "imagined_obs")
    sharp_true = load_obs(Path(args.sharp_npz), "true_obs")
    sharp_imag = load_obs(Path(args.sharp_npz), "imagined_obs")

    if not np.allclose(true_obs, sharp_true, atol=1e-4):
        diff = float(np.abs(true_obs - sharp_true).max())
        print(
            f"warning: baseline and SHARP true_obs differ (max abs {diff:.4g}); "
            "using sharp.true_obs as the reference."
        )
    true_obs = sharp_true

    rootx_true = integrate_rootx(true_obs, dt)
    rootx_base = integrate_rootx(base_imag, dt)
    rootx_sharp = integrate_rootx(sharp_imag, dt)

    print("rendering ground-truth trajectory")
    true_frames = render_trajectory(
        env, true_obs, rootx_true,
        resolution=args.resolution, camera_id=camera_id, label="true",
    )
    print("rendering baseline imagination")
    base_frames = render_trajectory(
        env, base_imag, rootx_base,
        resolution=args.resolution, camera_id=camera_id, label="baseline",
    )
    print("rendering SHARP imagination")
    sharp_frames = render_trajectory(
        env, sharp_imag, rootx_sharp,
        resolution=args.resolution, camera_id=camera_id, label="sharp",
    )

    figure_grid(
        rows=[
            ("Ground Truth", true_frames),
            ("Baseline Imagination", base_frames),
            ("SHARP Imagination", sharp_frames),
        ],
        steps=GRID_STEPS,
        title="Cheetah 3D Visualization: 200-Step Imagination Drift",
        out=FIG_DIR / "fig9_3d_render_comparison.png",
    )

    diff_frames = np.stack(
        [diff_frame(true_frames[t], sharp_frames[t], DIFF_AMPLIFY) for t in range(N_STEPS)]
    )
    figure_grid(
        rows=[
            ("Ground Truth", true_frames),
            ("Baseline Imagination", base_frames),
            ("SHARP Imagination", sharp_frames),
            (f"|Truth - SHARP| (x{DIFF_AMPLIFY})", diff_frames),
        ],
        steps=GRID_STEPS,
        title="Cheetah 3D Visualization with Drift Highlighting",
        out=FIG_DIR / "fig9_3d_render_comparison_diff.png",
    )

    if not args.no_gif:
        write_gif(
            true_frames=true_frames,
            base_frames=base_frames,
            sharp_frames=sharp_frames,
            out=FIG_DIR / "fig9_3d_render_comparison.gif",
            fps=args.fps,
        )


if __name__ == "__main__":
    main()
