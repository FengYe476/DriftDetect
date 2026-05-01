"""Run V4 frequency decomposition for latent and image-space rollouts."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.diagnostics.freq_decompose import (  # noqa: E402
    DEFAULT_BANDS,
    PCAResult,
    decompose_pair,
    fit_pca,
)


ROLLOUT_DIR = REPO_ROOT / "results" / "rollouts_v4"
MANIFEST = ROLLOUT_DIR / "manifest_v4.json"
OUTPUT_PATH = REPO_ROOT / "results" / "tables" / "v4_band_errors.json"

N_SEEDS = 20
HORIZON = 200
TRIM_START = 25
TRIM_END = 175
VAR_THRESHOLD = 0.95
IMAGE_PCA_CAP = 50
IMAGE_PCA_RANDOM_SEED = 17

BAND_ORDER = list(DEFAULT_BANDS.keys())
V3_KNOWN_SHARES = {
    "dc_trend": 0.467,
    "very_low": 0.104,
    "low": 0.284,
    "mid": 0.103,
    "high": 0.042,
}


@dataclass(frozen=True)
class Projection:
    mean: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray
    total_variance: float
    n_components: int


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    paths = collect_v4_rollout_paths()
    manifest = load_manifest()

    print("=== V4 FREQUENCY DECOMPOSITION ===")
    print(f"Rollouts: {len(paths)} files from {ROLLOUT_DIR.relative_to(REPO_ROOT)}")
    print(f"Trim window: [{TRIM_START}, {TRIM_END}] inclusive")
    print()

    print("[1/3] Latent-space decomposition (primary)")
    latent_true, latent_imag = load_latent_rollouts(paths)
    latent_pca, _ = fit_pca(np.concatenate(latent_true, axis=0), var_threshold=VAR_THRESHOLD)
    latent_result = aggregate_band_errors(
        latent_true,
        latent_imag,
        pca=latent_pca,
        representation="latent-space",
        notes=(
            "DreamerV4 packed tokenizer latents flattened from (8,64) to 512; "
            "shared PCA fitted on pooled true latents."
        ),
    )
    print_section("V4 latent-space", latent_result)
    print()

    print("[2/3] Image-space channel-mean decomposition (secondary, lossy)")
    channel_true, channel_imag = load_channel_mean_rollouts(paths)
    channel_result = aggregate_band_errors(
        channel_true,
        channel_imag,
        pca=None,
        representation="image-space-channel-mean",
        notes="Spatial mean RGB per frame, shape (T,3); lossy architecture-neutral sanity check.",
    )
    print_section("V4 image-space channel mean", channel_result)
    print()

    print("[3/3] Image-space PCA decomposition (secondary, richer)")
    image_projection = fit_streaming_image_pca(paths)
    image_true, image_imag = project_image_rollouts(paths, image_projection)
    image_pca_result = aggregate_band_errors(
        image_true,
        image_imag,
        pca=None,
        representation="image-space-PCA",
        notes=(
            "Flattened RGB frames projected through randomized PCA fitted on pooled true frames; "
            f"95% variance target capped at {IMAGE_PCA_CAP} PCs."
        ),
        projection_info={
            "n_components": image_projection.n_components,
            "var_explained": float(np.sum(image_projection.explained_variance_ratio)),
            "total_variance": float(image_projection.total_variance),
            "cap": IMAGE_PCA_CAP,
            "random_seed": IMAGE_PCA_RANDOM_SEED,
        },
    )
    print_section("V4 image-space PCA", image_pca_result)
    print()

    results = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "rollout_dir": str(ROLLOUT_DIR.relative_to(REPO_ROOT)),
            "manifest": str(MANIFEST.relative_to(REPO_ROOT)),
            "manifest_task": manifest.get("task"),
            "n_seeds": len(paths),
            "horizon": HORIZON,
            "trim_window": [TRIM_START, TRIM_END],
            "bands": {band: list(DEFAULT_BANDS[band]) for band in BAND_ORDER},
            "filter": "Butterworth order 4, zero-phase sosfiltfilt",
            "var_threshold": VAR_THRESHOLD,
        },
        "latent-space": latent_result,
        "image-space-channel-mean": channel_result,
        "image-space-PCA": image_pca_result,
        "comparison_table": build_comparison_table(latent_result, image_pca_result),
    }
    OUTPUT_PATH.write_text(json.dumps(to_jsonable(results), indent=2) + "\n")

    print_comparison_table(results["comparison_table"])
    print()
    print(f"Saved {OUTPUT_PATH.relative_to(REPO_ROOT)}")


def collect_v4_rollout_paths() -> list[Path]:
    paths = [ROLLOUT_DIR / f"cheetah-run_v4_seed{seed}_v1.npz" for seed in range(N_SEEDS)]
    missing = [path for path in paths if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing V4 rollout files: {missing_text}")
    return paths


def load_manifest() -> dict[str, Any]:
    if not MANIFEST.exists():
        return {}
    return json.loads(MANIFEST.read_text())


def load_latent_rollouts(paths: list[Path]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    true_signals: list[np.ndarray] = []
    imagined_signals: list[np.ndarray] = []
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            true_latent = as_2d_float(data["true_latent"], name=f"{path.name}:true_latent")
            imagined_latent = as_2d_float(data["imagined_latent"], name=f"{path.name}:imagined_latent")
        validate_shape(true_latent, (HORIZON, 512), f"{path.name}:true_latent")
        validate_shape(imagined_latent, (HORIZON, 512), f"{path.name}:imagined_latent")
        true_signals.append(true_latent)
        imagined_signals.append(imagined_latent)
    return true_signals, imagined_signals


def load_channel_mean_rollouts(paths: list[Path]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    true_signals: list[np.ndarray] = []
    imagined_signals: list[np.ndarray] = []
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            true_obs = as_obs_float(data["true_obs"], name=f"{path.name}:true_obs")
            imagined_obs = as_obs_float(data["imagined_obs"], name=f"{path.name}:imagined_obs")
        true_signals.append(true_obs.mean(axis=(2, 3), dtype=np.float64).astype(np.float32))
        imagined_signals.append(imagined_obs.mean(axis=(2, 3), dtype=np.float64).astype(np.float32))
    return true_signals, imagined_signals


def aggregate_band_errors(
    true_signals: list[np.ndarray],
    imagined_signals: list[np.ndarray],
    *,
    pca: PCAResult | None,
    representation: str,
    notes: str,
    projection_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if len(true_signals) != len(imagined_signals):
        raise ValueError("true_signals and imagined_signals length mismatch.")

    per_band_curves: dict[str, list[np.ndarray]] = {band: [] for band in BAND_ORDER}
    decompose_info: dict[str, Any] | None = None
    for true_signal, imagined_signal in zip(true_signals, imagined_signals):
        true_dec, imagined_dec = decompose_pair(
            true_signal,
            imagined_signal,
            pca=pca,
            var_threshold=VAR_THRESHOLD,
        )
        decompose_info = true_dec["_info"]
        for band in BAND_ORDER:
            diff = true_dec[band] - imagined_dec[band]
            per_band_curves[band].append(np.mean(diff**2, axis=1).astype(np.float64))

    bands: dict[str, Any] = {}
    window_means: dict[str, float] = {}
    for band in BAND_ORDER:
        curves = np.stack(per_band_curves[band], axis=0)
        mean_curve = curves.mean(axis=0)
        std_curve = curves.std(axis=0, ddof=1)
        sem_curve = std_curve / np.sqrt(curves.shape[0])
        ci_low = mean_curve - 1.96 * sem_curve
        ci_high = mean_curve + 1.96 * sem_curve

        window = curves[:, TRIM_START : TRIM_END + 1].mean(axis=1)
        window_mean = float(window.mean())
        window_std = float(window.std(ddof=1))
        window_sem = window_std / np.sqrt(window.shape[0])
        bands[band] = {
            "curve_mean": mean_curve,
            "curve_std": std_curve,
            "curve_ci95_low": ci_low,
            "curve_ci95_high": ci_high,
            "window_mean_mse": window_mean,
            "window_std_mse": window_std,
            "window_ci95_low": float(window_mean - 1.96 * window_sem),
            "window_ci95_high": float(window_mean + 1.96 * window_sem),
            "per_seed_window_mse": window,
        }
        window_means[band] = window_mean

    total = float(sum(window_means.values()))
    for band in BAND_ORDER:
        bands[band]["share"] = window_means[band] / total if total > 0 else 0.0

    ranking = sorted(BAND_ORDER, key=lambda band: window_means[band], reverse=True)
    return {
        "representation": representation,
        "notes": notes,
        "decompose_info": decompose_info or {},
        "projection_info": projection_info,
        "trim_window": [TRIM_START, TRIM_END],
        "n_seeds": len(true_signals),
        "bands": bands,
        "ranking": ranking,
        "ranking_text": " > ".join(ranking),
    }


def fit_streaming_image_pca(paths: list[Path]) -> Projection:
    n_frames = len(paths) * HORIZON
    n_features = 3 * 128 * 128
    rank = IMAGE_PCA_CAP
    oversample = 10
    sketch_dim = min(rank + oversample, n_frames)
    rng = np.random.default_rng(IMAGE_PCA_RANDOM_SEED)
    omega = rng.standard_normal((n_features, sketch_dim), dtype=np.float32)

    mean = np.zeros(n_features, dtype=np.float64)
    row_offset = 0
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            frames = as_obs_float(data["true_obs"], name=f"{path.name}:true_obs")
        flat = frames.reshape(frames.shape[0], -1).astype(np.float32, copy=False)
        mean += flat.sum(axis=0, dtype=np.float64)
        row_offset += flat.shape[0]
    if row_offset != n_frames:
        raise ValueError(f"Expected {n_frames} frames for image PCA, saw {row_offset}.")
    mean /= float(n_frames)
    mean32 = mean.astype(np.float32)

    y = np.empty((n_frames, sketch_dim), dtype=np.float32)
    total_ss = 0.0
    row_offset = 0
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            frames = as_obs_float(data["true_obs"], name=f"{path.name}:true_obs")
        centered = frames.reshape(frames.shape[0], -1).astype(np.float32, copy=True)
        centered -= mean32
        y[row_offset : row_offset + centered.shape[0]] = centered @ omega
        total_ss += float(np.sum(centered * centered, dtype=np.float64))
        row_offset += centered.shape[0]

    q, _ = np.linalg.qr(y, mode="reduced")
    del y

    b = np.zeros((q.shape[1], n_features), dtype=np.float32)
    row_offset = 0
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            frames = as_obs_float(data["true_obs"], name=f"{path.name}:true_obs")
        centered = frames.reshape(frames.shape[0], -1).astype(np.float32, copy=True)
        centered -= mean32
        q_block = q[row_offset : row_offset + centered.shape[0]].astype(np.float32, copy=False)
        b += q_block.T @ centered
        row_offset += centered.shape[0]

    _, singular_values, vt = np.linalg.svd(b.astype(np.float64), full_matrices=False)
    explained = singular_values**2 / max(n_frames - 1, 1)
    total_variance = total_ss / max(n_frames - 1, 1)
    ratios = explained / total_variance if total_variance > 0 else np.zeros_like(explained)
    cumulative = np.cumsum(ratios)
    needed = int(np.searchsorted(cumulative, VAR_THRESHOLD, side="left") + 1)
    n_components = min(IMAGE_PCA_CAP, max(1, needed), vt.shape[0])

    return Projection(
        mean=mean32,
        components=vt[:n_components].astype(np.float32),
        explained_variance_ratio=ratios[:n_components].astype(np.float64),
        total_variance=float(total_variance),
        n_components=n_components,
    )


def project_image_rollouts(
    paths: list[Path],
    projection: Projection,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    true_signals: list[np.ndarray] = []
    imagined_signals: list[np.ndarray] = []
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            true_obs = as_obs_float(data["true_obs"], name=f"{path.name}:true_obs")
            imagined_obs = as_obs_float(data["imagined_obs"], name=f"{path.name}:imagined_obs")
        true_signals.append(project_frames(true_obs, projection))
        imagined_signals.append(project_frames(imagined_obs, projection))
    return true_signals, imagined_signals


def project_frames(frames: np.ndarray, projection: Projection) -> np.ndarray:
    flat = frames.reshape(frames.shape[0], -1).astype(np.float32, copy=True)
    flat -= projection.mean
    return (flat @ projection.components.T).astype(np.float32)


def build_comparison_table(
    latent_result: dict[str, Any],
    image_pca_result: dict[str, Any],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for band in BAND_ORDER:
        rows.append(
            {
                "band": band,
                "v3_latent_share": V3_KNOWN_SHARES[band],
                "v4_latent_share": float(latent_result["bands"][band]["share"]),
                "v4_img_pca_share": float(image_pca_result["bands"][band]["share"]),
            }
        )
    return rows


def print_section(label: str, result: dict[str, Any]) -> None:
    projection_info = result.get("projection_info") or {}
    info = result["decompose_info"]
    pcs = projection_info.get("n_components", info.get("n_pcs", info.get("output_dim")))
    var = projection_info.get("var_explained", info.get("var_explained", 1.0))
    print(f"{label}: n_pcs={pcs}, var_explained={float(var) * 100:.1f}%")
    print("  Band          Mean MSE [25:175]        Std        95% CI             Share")
    for band in BAND_ORDER:
        stats = result["bands"][band]
        print(
            f"  {band:<12}"
            f"{stats['window_mean_mse']:>14.6g}"
            f"{stats['window_std_mse']:>12.6g}"
            f"   [{stats['window_ci95_low']:.6g}, {stats['window_ci95_high']:.6g}]"
            f"{stats['share'] * 100:>10.1f}%"
        )
    print(f"  Ranking: {result['ranking_text']}")


def print_comparison_table(rows: list[dict[str, float | str]]) -> None:
    print("=== CROSS-ARCHITECTURE SHARE TABLE ===")
    print("Band        | V3 latent share | V4 latent share | V4 img-PCA share")
    for row in rows:
        print(
            f"{row['band']:<11} | "
            f"{float(row['v3_latent_share']) * 100:>7.1f}%         | "
            f"{float(row['v4_latent_share']) * 100:>7.1f}%         | "
            f"{float(row['v4_img_pca_share']) * 100:>7.1f}%"
        )


def as_2d_float(array: np.ndarray, *, name: str) -> np.ndarray:
    out = np.asarray(array, dtype=np.float32)
    if out.ndim != 2:
        raise ValueError(f"{name} must be 2D, got {out.shape}.")
    return out


def as_obs_float(array: np.ndarray, *, name: str) -> np.ndarray:
    out = np.asarray(array, dtype=np.float32)
    if out.shape != (HORIZON, 3, 128, 128):
        raise ValueError(f"{name} must have shape {(HORIZON, 3, 128, 128)}, got {out.shape}.")
    if out.min() < -1e-5 or out.max() > 1.0 + 1e-5:
        raise ValueError(f"{name} expected values in [0,1], got min={out.min()} max={out.max()}.")
    return out


def validate_shape(array: np.ndarray, expected: tuple[int, int], name: str) -> None:
    if array.shape != expected:
        raise ValueError(f"{name} expected shape {expected}, got {array.shape}.")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "driftdetect-matplotlib-cache"),
    )
    main()
