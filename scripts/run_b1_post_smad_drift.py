#!/usr/bin/env python3
"""B1 check: post-SMAD drift-basis staleness vs model adaptation."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXISTING_ROLLOUT_DIR = (
    REPO_ROOT / "results" / "smad_phase2" / "smad_eta020" / "rollouts"
)
DEFAULT_COLLECTED_ROLLOUT_DIR = REPO_ROOT / "results" / "b1_post_smad_rollouts_eta0"
OUTPUT_PATH = REPO_ROOT / "results" / "tables" / "b1_staleness_check.json"
CORRECTED_OUTPUT_PATH = REPO_ROOT / "results" / "tables" / "b1_staleness_check_corrected.json"
BASELINE_U_PATH = REPO_ROOT / "results" / "smad" / "U_drift_cheetah_r10.npy"

N_ROLLOUTS = 20
HORIZON = 200
TOTAL_STEPS = 1000
IMAGINATION_START = 200
LATENT_DIM = 1536
DETER_START = 1024
DETER_DIM = 512
RANK = 10
RANKS = (3, 5, 10)
ETA = 0.20
TRIM_START = 25
TRIM_END_INCLUSIVE = 175
VAR_THRESHOLD = 0.95
TASK = "dmc_cheetah_run"
APPROX_EQUAL_PCT_POINTS = 2.0

BANDS = {
    "dc_trend": (0.00, 0.02),
    "very_low": (0.02, 0.05),
    "low": (0.05, 0.10),
    "mid": (0.10, 0.20),
    "high": (0.20, 0.50),
}


@dataclass(frozen=True)
class PCAFit:
    mean: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray
    singular_values: np.ndarray


def main() -> None:
    args = parse_args()
    checkpoint_path, checkpoint_status = resolve_checkpoint(args.checkpoint)
    if args.mode == "collect":
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"Cannot collect rollouts because checkpoint is unavailable: "
                f"{checkpoint_status}\n{checkpoint_instructions()}"
            )
        collect_post_smad_rollouts(
            checkpoint_path=checkpoint_path,
            output_dir=args.collected_rollout_dir,
            task=args.task,
            device=args.device,
            n_rollouts=args.n_rollouts,
            total_steps=args.total_steps,
            imagination_start=args.imagination_start,
            horizon=args.horizon,
            inference_eta=args.eta_inference,
            baseline_u_path=args.baseline_u_path,
            adaptive_u_from_checkpoint=args.adaptive_u_from_checkpoint,
            rank=args.rank,
            force=args.force_recollect,
        )
        print(
            f"Collected {args.n_rollouts} rollouts at eta_inference="
            f"{args.eta_inference:.3f} into {relative_path(args.collected_rollout_dir)}."
        )
        return

    rollout_dir, collection_status = prepare_rollouts(
        args=args,
        checkpoint_path=checkpoint_path,
        checkpoint_status=checkpoint_status,
    )

    rollout_paths = discover_rollouts(rollout_dir, min_count=args.n_rollouts)
    true_deter, imag_deter = load_deter_rollouts(rollout_paths)

    post_smad_u, post_smad_pca = estimate_u_drift(
        true_deter,
        imag_deter,
        rank=args.rank,
        trim_start=args.trim_start,
        trim_end_inclusive=args.trim_end_inclusive,
    )
    baseline_u = load_basis(args.baseline_u_path, rank=args.rank)
    overlaps = basis_overlaps(baseline_u, post_smad_u, ranks=args.ranks)

    old_basis_eval = evaluate_posthoc_damping(
        true_deter,
        imag_deter,
        u_drift=baseline_u,
        eta=args.eta,
        trim_start=args.trim_start,
        trim_end_inclusive=args.trim_end_inclusive,
        var_threshold=args.var_threshold,
    )
    new_basis_eval = evaluate_posthoc_damping(
        true_deter,
        imag_deter,
        u_drift=post_smad_u,
        eta=args.eta,
        trim_start=args.trim_start,
        trim_end_inclusive=args.trim_end_inclusive,
        var_threshold=args.var_threshold,
    )
    original_comparison = load_original_comparison(args.original_result_path)
    side_by_side = build_side_by_side_comparison(
        original=original_comparison,
        ranks=args.ranks,
        overlaps=overlaps,
        old_basis_eval=old_basis_eval,
        new_basis_eval=new_basis_eval,
    )
    verdict = interpret_result(
        rank10_overlap=overlaps["10"],
        new_reduction=new_basis_eval["dc_trend_reduction_pct"],
        old_reduction=old_basis_eval["dc_trend_reduction_pct"],
        approx_equal_pct_points=args.approx_equal_pct_points,
    )
    rollout_eta = infer_rollout_eta(rollout_paths, fallback=args.eta_inference)

    result = {
        "analysis": "b1_post_smad_drift",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task": TASK,
        "rollout_eta": float(rollout_eta),
        "purpose": (
            "Test whether the Phase 2 post-hoc/training-time effectiveness gap "
            "is better explained by U_drift basis staleness or model adaptation."
        ),
        "checkpoint": {
            "path": relative_path(checkpoint_path) if checkpoint_path else None,
            "status": checkpoint_status,
            "instructions_if_missing": checkpoint_instructions(),
        },
        "rollouts": {
            "mode": args.mode,
            "collection_status": collection_status,
            "rollout_dir": relative_path(rollout_dir),
            "rollout_files": [relative_path(path) for path in rollout_paths],
            "n_rollouts": int(true_deter.shape[0]),
            "horizon": int(true_deter.shape[1]),
            "total_steps": int(args.total_steps),
            "imagination_start": int(args.imagination_start),
            "rollout_eta": float(rollout_eta),
            "inference_eta": float(rollout_eta),
        },
        "latent": {
            "latent_dim": LATENT_DIM,
            "deter_dim": DETER_DIM,
            "deter_slice": [DETER_START, LATENT_DIM],
        },
        "drift_estimation": {
            "rank": int(args.rank),
            "trim_window_inclusive": [args.trim_start, args.trim_end_inclusive],
            "post_smad_drift_pool_shape": [
                int(true_deter.shape[0] * (args.trim_end_inclusive - args.trim_start + 1)),
                DETER_DIM,
            ],
            "post_smad_explained_variance_ratio_top10": post_smad_pca.explained_variance_ratio[
                : args.rank
            ].tolist(),
            "post_smad_cumulative_explained_variance_top10": float(
                np.sum(post_smad_pca.explained_variance_ratio[: args.rank])
            ),
            "baseline_u_path": relative_path(args.baseline_u_path),
        },
        "overlaps": {
            str(rank): {
                "baseline_vs_post_smad": float(overlaps[str(rank)]),
            }
            for rank in args.ranks
        },
        "posthoc_damping": {
            "eta": float(args.eta),
            "frequency_bands": {band: list(bounds) for band, bounds in BANDS.items()},
            "old_basis": old_basis_eval,
            "new_basis": new_basis_eval,
        },
        "diagnostic_table": build_diagnostic_table(
            ranks=args.ranks,
            overlaps=overlaps,
            old_basis_eval=old_basis_eval,
            new_basis_eval=new_basis_eval,
        ),
        "original_results_path": relative_path(args.original_result_path),
        "original_overlaps": original_comparison.get("overlaps"),
        "original_posthoc_damping": original_comparison.get("posthoc_damping"),
        "side_by_side_comparison": side_by_side,
        "interpretation": verdict,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    print_summary(result)
    print(f"\nSaved: {relative_path(args.output)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the B1 post-SMAD drift-basis staleness check."
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "collect", "analyze", "analyze-existing"),
        default="auto",
        help=(
            "auto collects from --checkpoint when available and then analyzes; "
            "collect only collects rollout NPZs; analyze/analyze-existing analyze "
            "an existing rollout directory."
        ),
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--rollout-dir",
        "--rollout_dir",
        dest="rollout_dir",
        type=Path,
        default=DEFAULT_EXISTING_ROLLOUT_DIR,
        help="Directory of existing post-SMAD rollout NPZs for analysis fallback.",
    )
    parser.add_argument(
        "--collected-rollout-dir",
        "--output_dir",
        dest="collected_rollout_dir",
        type=Path,
        default=DEFAULT_COLLECTED_ROLLOUT_DIR,
        help="Output directory for freshly collected eta=0 post-SMAD rollouts.",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--baseline-u-path",
        "--baseline_U",
        dest="baseline_u_path",
        type=Path,
        default=BASELINE_U_PATH,
    )
    parser.add_argument(
        "--original-result-path",
        type=Path,
        default=OUTPUT_PATH,
        help="Original B1 result JSON used for side-by-side comparison.",
    )
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--n-rollouts",
        "--n_seeds",
        dest="n_rollouts",
        type=int,
        default=N_ROLLOUTS,
    )
    parser.add_argument("--total-steps", type=int, default=TOTAL_STEPS)
    parser.add_argument("--imagination-start", type=int, default=IMAGINATION_START)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--rank", type=int, default=RANK)
    parser.add_argument("--ranks", type=int, nargs="+", default=list(RANKS))
    parser.add_argument("--eta", type=float, default=ETA)
    parser.add_argument(
        "--eta-inference",
        "--eta_inference",
        dest="eta_inference",
        type=float,
        default=0.0,
        help=(
            "SMAD damping eta applied during rollout collection. Use 0.0 to "
            "load SMAD-trained weights but run vanilla img_step at inference."
        ),
    )
    parser.add_argument(
        "--adaptive-u-from-checkpoint",
        "--adaptive_U_from_checkpoint",
        dest="adaptive_u_from_checkpoint",
        action="store_true",
        help=(
            "When eta_inference > 0, load adaptive_smad_scheduler_state.current_U "
            "from --checkpoint instead of --baseline_U."
        ),
    )
    parser.add_argument("--trim-start", type=int, default=TRIM_START)
    parser.add_argument(
        "--trim-end-inclusive",
        type=int,
        default=TRIM_END_INCLUSIVE,
    )
    parser.add_argument("--var-threshold", type=float, default=VAR_THRESHOLD)
    parser.add_argument(
        "--approx-equal-pct-points",
        type=float,
        default=APPROX_EQUAL_PCT_POINTS,
    )
    parser.add_argument(
        "--force-recollect",
        action="store_true",
        help="Recollect rollout NPZs even when collected outputs already exist.",
    )
    args = parser.parse_args()
    args.rollout_dir = resolve_path(args.rollout_dir)
    args.collected_rollout_dir = resolve_path(args.collected_rollout_dir)
    args.output = resolve_path(args.output)
    args.baseline_u_path = resolve_path(args.baseline_u_path)
    args.original_result_path = resolve_path(args.original_result_path)
    if args.checkpoint is not None:
        args.checkpoint = resolve_path(args.checkpoint)
    args.ranks = tuple(sorted(set(args.ranks)))

    if args.horizon != HORIZON:
        raise ValueError(f"--horizon must be {HORIZON} for this analysis.")
    if args.n_rollouts < 1:
        raise ValueError("--n-rollouts must be positive.")
    if args.total_steps < args.imagination_start + args.horizon:
        raise ValueError("--total-steps must cover the imagination window.")
    if not 1 <= args.rank <= DETER_DIM:
        raise ValueError(f"--rank must be in [1, {DETER_DIM}].")
    if any(rank <= 0 or rank > args.rank for rank in args.ranks):
        raise ValueError(f"--ranks must be in [1, {args.rank}].")
    if args.eta < 0:
        raise ValueError("--eta must be non-negative.")
    if args.eta_inference < 0:
        raise ValueError("--eta-inference must be non-negative.")
    if not 0 < args.var_threshold <= 1:
        raise ValueError("--var-threshold must be in (0, 1].")
    if not 0 <= args.trim_start <= args.trim_end_inclusive < HORIZON:
        raise ValueError("--trim-start/--trim-end-inclusive must be inside horizon.")
    return args


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def resolve_checkpoint(requested: Path | None) -> tuple[Path | None, str]:
    if requested is not None:
        if not requested.exists():
            raise FileNotFoundError(f"Specified checkpoint does not exist: {requested}")
        return requested, "specified via --checkpoint"

    root = REPO_ROOT / "results" / "smad_phase2" / "smad_eta020"
    ordered_candidates = [
        root / "checkpoints" / "latest.pt",
        root / "checkpoints" / "final.pt",
        root / "latest.pt",
        root / "final.pt",
    ]
    checkpoint_dir = root / "checkpoints"
    if checkpoint_dir.exists():
        ordered_candidates.extend(sorted(checkpoint_dir.glob("*.pt")))
        ordered_candidates.extend(sorted(checkpoint_dir.glob("*.pth")))
    ordered_candidates.extend(sorted(root.glob("*.pt")))
    ordered_candidates.extend(sorted(root.glob("*.pth")))
    candidates = []
    seen = set()
    for path in ordered_candidates:
        if path in seen or not path.is_file():
            continue
        candidates.append(path)
        seen.add(path)
    if len(candidates) == 1:
        return candidates[0], "auto-located unique SMAD Phase 2 checkpoint"
    if len(candidates) > 1:
        return (
            None,
            "multiple candidate SMAD checkpoints found; specify one with --checkpoint: "
            + ", ".join(relative_path(path) for path in candidates),
        )
    return None, "no SMAD Phase 2 checkpoint found under results/smad_phase2/smad_eta020"


def prepare_rollouts(
    *,
    args: argparse.Namespace,
    checkpoint_path: Path | None,
    checkpoint_status: str,
) -> tuple[Path, str]:
    if args.mode in {"analyze", "analyze-existing"}:
        if not args.rollout_dir.exists():
            raise FileNotFoundError(f"Rollout directory not found: {args.rollout_dir}")
        return args.rollout_dir, "analyzed explicit rollout directory"

    if args.mode in {"auto", "collect"} and checkpoint_path is not None:
        collect_post_smad_rollouts(
            checkpoint_path=checkpoint_path,
            output_dir=args.collected_rollout_dir,
            task=args.task,
            device=args.device,
            n_rollouts=args.n_rollouts,
            total_steps=args.total_steps,
            imagination_start=args.imagination_start,
            horizon=args.horizon,
            inference_eta=args.eta_inference,
            baseline_u_path=args.baseline_u_path,
            adaptive_u_from_checkpoint=args.adaptive_u_from_checkpoint,
            rank=args.rank,
            force=args.force_recollect,
        )
        return (
            args.collected_rollout_dir,
            f"freshly collected from SMAD checkpoint with eta={args.eta_inference} inference",
        )

    if args.mode == "collect":
        raise FileNotFoundError(
            f"Cannot collect rollouts because checkpoint is unavailable: "
            f"{checkpoint_status}\n{checkpoint_instructions()}"
        )

    if args.rollout_dir.exists():
        print(
            "Warning: SMAD checkpoint was not available; analyzing existing "
            f"rollouts in {relative_path(args.rollout_dir)}."
        )
        print(checkpoint_instructions())
        return (
            args.rollout_dir,
            (
                "analyzed existing rollout directory because checkpoint was "
                f"unavailable ({checkpoint_status})"
            ),
        )

    raise FileNotFoundError(
        f"No checkpoint and no existing rollout directory found.\n"
        f"Checkpoint status: {checkpoint_status}\n"
        f"Expected existing rollouts at: {args.rollout_dir}\n"
        f"{checkpoint_instructions()}"
    )


def checkpoint_instructions() -> str:
    return (
        "Run on RunPod with the Phase 2 SMAD checkpoint, for example:\n"
        "  python scripts/run_b1_post_smad_drift.py --mode collect "
        "--checkpoint /path/to/smad_eta020/latest.pt --device cuda "
        "--eta_inference 0.0\n"
        "The script will collect eta=0 inference rollouts into "
        "results/b1_post_smad_rollouts_eta0/ for a separate analyze pass."
    )


def collect_post_smad_rollouts(
    *,
    checkpoint_path: Path,
    output_dir: Path,
    task: str,
    device: str | None,
    n_rollouts: int,
    total_steps: int,
    imagination_start: int,
    horizon: int,
    inference_eta: float,
    baseline_u_path: Path,
    adaptive_u_from_checkpoint: bool,
    rank: int,
    force: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = [output_dir / f"cheetah_v3_seed{seed}_v2.npz" for seed in range(n_rollouts)]
    if not force and all(path.exists() for path in expected):
        print(f"Using existing collected rollouts in {relative_path(output_dir)}.")
        return

    from src.models.dreamerv3_adapter import DreamerV3Adapter
    import torch

    adapter = DreamerV3Adapter(task=task, device=device)
    original_build = adapter._build
    projector = None
    patch_description = "not applied; vanilla img_step"
    if inference_eta > 0.0:
        from src.smad.img_step_patch import MutableImgStepPatch

        if adaptive_u_from_checkpoint:
            basis = load_adaptive_basis_from_checkpoint(
                checkpoint_path,
                rank=rank,
                torch_module=torch,
            )
            basis_source = "adaptive checkpoint current_U"
        else:
            basis = load_basis(baseline_u_path, rank=rank)
            basis_source = f"basis file {relative_path(baseline_u_path)}"
        projector = torch.as_tensor(basis @ basis.T, dtype=torch.float32)
        patch_description = (
            "src.smad.img_step_patch.MutableImgStepPatch with eta="
            f"{inference_eta} using {basis_source}"
        )

    def build_with_inference_eta(seed: int) -> None:
        original_build(seed)
        if inference_eta == 0.0:
            return
        assert adapter.agent is not None
        assert projector is not None
        MutableImgStepPatch(adapter.agent._wm.dynamics, projector, eta=inference_eta)

    adapter._build = build_with_inference_eta  # type: ignore[method-assign]
    adapter.load_checkpoint(str(checkpoint_path))

    for seed, output_path in enumerate(expected):
        if output_path.exists() and not force:
            print(f"Skipping existing rollout {relative_path(output_path)}.")
            continue
        print(f"Collecting post-SMAD eta={inference_eta} rollout seed {seed}...")
        rollout = adapter.extract_rollout(
            seed=seed,
            total_steps=total_steps,
            imagination_start=imagination_start,
            horizon=horizon,
            include_latent=True,
        )
        np.savez(
            output_path,
            true_obs=rollout["true_obs"].astype(np.float32),
            imagined_obs=rollout["imagined_obs"].astype(np.float32),
            actions=rollout["actions"].astype(np.float32),
            rewards=rollout["rewards"].astype(np.float32),
            true_latent=rollout["true_latent"].astype(np.float32),
            imagined_latent=rollout["imagined_latent"].astype(np.float32),
            metadata=np.array(
                {
                    "analysis": "b1_post_smad_drift",
                    "checkpoint_path": str(checkpoint_path),
                    "seed": seed,
                    "task": task,
                    "total_steps": total_steps,
                    "imagination_start": imagination_start,
                    "horizon": horizon,
                    "inference_eta": float(inference_eta),
                    "rollout_eta": float(inference_eta),
                    "smad_patch": patch_description,
                },
                dtype=object,
            ),
        )
        print(f"Saved {relative_path(output_path)}.")


def load_adaptive_basis_from_checkpoint(
    checkpoint_path: Path,
    *,
    rank: int,
    torch_module,
) -> np.ndarray:
    """Load the final Adaptive-SMAD basis from a training checkpoint."""

    try:
        checkpoint = torch_module.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch_module.load(checkpoint_path, map_location="cpu")

    scheduler_state = checkpoint.get("adaptive_smad_scheduler_state")
    if not isinstance(scheduler_state, dict):
        raise KeyError(
            f"{relative_path(checkpoint_path)} does not contain "
            "adaptive_smad_scheduler_state."
        )
    if "current_U" not in scheduler_state:
        raise KeyError(
            f"{relative_path(checkpoint_path)} adaptive_smad_scheduler_state "
            "does not contain current_U."
        )

    basis = np.asarray(scheduler_state["current_U"], dtype=np.float64)
    if basis.ndim != 2:
        raise ValueError(f"current_U must be 2D, got shape {basis.shape}.")
    if basis.shape[0] != DETER_DIM:
        raise ValueError(
            f"current_U leading dimension must be {DETER_DIM}, got {basis.shape[0]}."
        )
    if rank > basis.shape[1]:
        raise ValueError(
            f"Requested rank={rank}, but checkpoint current_U only has "
            f"{basis.shape[1]} columns."
        )
    return basis[:, :rank].copy()


def discover_rollouts(rollout_dir: Path, *, min_count: int) -> list[Path]:
    paths = sorted(rollout_dir.glob("*.npz"))
    if len(paths) < min_count:
        raise FileNotFoundError(
            f"Expected at least {min_count} rollout NPZs in {rollout_dir}, "
            f"found {len(paths)}."
        )
    if len(paths) > min_count:
        print(
            f"Warning: found {len(paths)} rollout NPZs in {relative_path(rollout_dir)}; "
            "analyzing all of them."
        )
    return paths


def load_deter_rollouts(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    true_deters = []
    imag_deters = []
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            true_latent = require_array(data, "true_latent", path)
            imag_latent = require_array(data, "imagined_latent", path)
        validate_latent(true_latent, "true_latent", path)
        validate_latent(imag_latent, "imagined_latent", path)
        true_deters.append(true_latent[:, DETER_START:].astype(np.float64, copy=False))
        imag_deters.append(imag_latent[:, DETER_START:].astype(np.float64, copy=False))
    return np.stack(true_deters, axis=0), np.stack(imag_deters, axis=0)


def require_array(data: np.lib.npyio.NpzFile, key: str, path: Path) -> np.ndarray:
    if key not in data:
        raise KeyError(f"{path} does not contain {key!r}.")
    return np.asarray(data[key])


def infer_rollout_eta(paths: list[Path], *, fallback: float) -> float:
    values = []
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            if "metadata" not in data:
                continue
            try:
                metadata = data["metadata"].item()
            except ValueError:
                continue
            if not isinstance(metadata, dict):
                continue
            if "rollout_eta" in metadata:
                values.append(float(metadata["rollout_eta"]))
            elif "inference_eta" in metadata:
                values.append(float(metadata["inference_eta"]))
    if values:
        unique = sorted(set(round(value, 12) for value in values))
        if len(unique) > 1:
            raise ValueError(f"Rollout metadata contains mixed inference eta values: {unique}")
        return float(values[0])
    if "smad_eta020" in str(paths[0].parent):
        return 0.20
    return float(fallback)


def validate_latent(array: np.ndarray, key: str, path: Path) -> None:
    expected = (HORIZON, LATENT_DIM)
    if array.shape != expected:
        raise ValueError(f"{path}:{key} expected {expected}, got {array.shape}.")


def load_basis(path: Path, *, rank: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Baseline U_drift basis not found: {path}")
    basis = np.load(path).astype(np.float64, copy=False)
    if basis.ndim != 2 or basis.shape[0] != DETER_DIM or basis.shape[1] < rank:
        raise ValueError(
            f"{path} expected shape ({DETER_DIM}, >= {rank}), got {basis.shape}."
        )
    basis = basis[:, :rank]
    assert_orthonormal(basis, "baseline U_drift")
    return basis


def estimate_u_drift(
    true_deter: np.ndarray,
    imag_deter: np.ndarray,
    *,
    rank: int,
    trim_start: int,
    trim_end_inclusive: int,
) -> tuple[np.ndarray, PCAFit]:
    drift = imag_deter - true_deter
    drift_pool = drift[:, trim_start : trim_end_inclusive + 1, :].reshape(
        -1,
        DETER_DIM,
    )
    drift_pca = fit_pca(drift_pool)
    basis = drift_pca.components[:rank].T
    assert_orthonormal(basis, "post-SMAD U_drift")
    return basis, drift_pca


def basis_overlaps(
    baseline_u: np.ndarray,
    post_smad_u: np.ndarray,
    *,
    ranks: tuple[int, ...],
) -> dict[str, float]:
    return {
        str(rank): subspace_overlap(baseline_u[:, :rank], post_smad_u[:, :rank])
        for rank in ranks
    }


def evaluate_posthoc_damping(
    true_deter: np.ndarray,
    imag_deter: np.ndarray,
    *,
    u_drift: np.ndarray,
    eta: float,
    trim_start: int,
    trim_end_inclusive: int,
    var_threshold: float,
) -> dict[str, Any]:
    eval_pca = fit_pca(true_deter.reshape(-1, DETER_DIM))
    eval_k = n_components_for_variance(
        eval_pca.explained_variance_ratio,
        threshold=var_threshold,
    )
    eval_components = eval_pca.components[:eval_k]
    true_projected = project_rollouts(true_deter, eval_pca.mean, eval_components)
    imag_projected = project_rollouts(imag_deter, eval_pca.mean, eval_components)
    damped_deter = apply_posthoc_damping(true_deter, imag_deter, u_drift, eta)
    damped_projected = project_rollouts(damped_deter, eval_pca.mean, eval_components)

    true_bands = filter_all_bands(true_projected)
    imag_bands = filter_all_bands(imag_projected)
    damped_bands = filter_all_bands(damped_projected)
    original_band_mse = band_mse_over_window(
        true_bands,
        imag_bands,
        trim_start=trim_start,
        trim_end_inclusive=trim_end_inclusive,
    )
    damped_band_mse = band_mse_over_window(
        true_bands,
        damped_bands,
        trim_start=trim_start,
        trim_end_inclusive=trim_end_inclusive,
    )
    original_total = float(sum(original_band_mse.values()))
    damped_total = float(sum(damped_band_mse.values()))
    raw_original = raw_mse_over_window(
        true_deter,
        imag_deter,
        trim_start=trim_start,
        trim_end_inclusive=trim_end_inclusive,
    )
    raw_damped = raw_mse_over_window(
        true_deter,
        damped_deter,
        trim_start=trim_start,
        trim_end_inclusive=trim_end_inclusive,
    )
    return {
        "frequency_pca": {
            "retained_components": int(eval_k),
            "retained_variance_explained": float(
                np.sum(eval_pca.explained_variance_ratio[:eval_k])
            ),
            "explained_variance_ratio_top20": eval_pca.explained_variance_ratio[
                : min(20, eval_pca.explained_variance_ratio.size)
            ].tolist(),
        },
        "band_mse_original": original_band_mse,
        "band_mse_damped": damped_band_mse,
        "dc_trend_mse_original": float(original_band_mse["dc_trend"]),
        "dc_trend_mse_damped": float(damped_band_mse["dc_trend"]),
        "dc_trend_reduction_pct": percent_reduction(
            original_band_mse["dc_trend"],
            damped_band_mse["dc_trend"],
        ),
        "total_mse_original": original_total,
        "total_mse_damped": damped_total,
        "total_reduction_pct": percent_reduction(original_total, damped_total),
        "raw_deter_mse_original": float(raw_original),
        "raw_deter_mse_damped": float(raw_damped),
        "raw_deter_reduction_pct": percent_reduction(raw_original, raw_damped),
    }


def fit_pca(matrix: np.ndarray) -> PCAFit:
    matrix = np.asarray(matrix, dtype=np.float64)
    mean = matrix.mean(axis=0, keepdims=True)
    centered = matrix - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    explained_variance = singular_values**2 / max(matrix.shape[0] - 1, 1)
    total = explained_variance.sum()
    if total <= 0:
        explained_ratio = np.zeros_like(explained_variance)
        explained_ratio[0] = 1.0
    else:
        explained_ratio = explained_variance / total
    return PCAFit(
        mean=mean.reshape(-1),
        components=vt,
        explained_variance_ratio=explained_ratio,
        singular_values=singular_values,
    )


def n_components_for_variance(ratios: np.ndarray, threshold: float) -> int:
    cumulative = np.cumsum(ratios)
    return int(np.searchsorted(cumulative, threshold, side="left") + 1)


def project_rollouts(
    rollouts: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
) -> np.ndarray:
    return (rollouts - mean.reshape(1, 1, -1)) @ components.T


def apply_posthoc_damping(
    true_deter: np.ndarray,
    imag_deter: np.ndarray,
    u_drift: np.ndarray,
    eta: float,
) -> np.ndarray:
    drift = imag_deter - true_deter
    projected_drift = (drift @ u_drift) @ u_drift.T
    return imag_deter - float(eta) * projected_drift


def filter_all_bands(projected: np.ndarray) -> dict[str, np.ndarray]:
    return {
        band: filter_projected(projected, low=low, high=high)
        for band, (low, high) in BANDS.items()
    }


def filter_projected(projected: np.ndarray, low: float, high: float) -> np.ndarray:
    out = np.empty_like(projected, dtype=np.float64)
    for seed in range(projected.shape[0]):
        for dim in range(projected.shape[2]):
            out[seed, :, dim] = bandpass(projected[seed, :, dim], low=low, high=high)
    return out


def bandpass(signal_1d: np.ndarray, low: float, high: float) -> np.ndarray:
    signal_1d = np.asarray(signal_1d, dtype=np.float64)
    nyquist = 0.5
    if low == 0 and high >= nyquist:
        return signal_1d.copy()
    if high <= 0:
        return np.full_like(signal_1d, signal_1d.mean())
    high = min(high, nyquist)
    if low == 0:
        sos = butter(4, high, btype="lowpass", output="sos", fs=1.0)
    elif high >= nyquist:
        sos = butter(4, low, btype="highpass", output="sos", fs=1.0)
    else:
        sos = butter(4, (low, high), btype="bandpass", output="sos", fs=1.0)
    padlen = min(signal_1d.size - 1, 3 * (2 * sos.shape[0] + 1))
    return sosfiltfilt(sos, signal_1d, padlen=padlen)


def band_mse_over_window(
    true_bands: dict[str, np.ndarray],
    imag_bands: dict[str, np.ndarray],
    *,
    trim_start: int,
    trim_end_inclusive: int,
) -> dict[str, float]:
    window = slice(trim_start, trim_end_inclusive + 1)
    return {
        band: float(
            np.mean(
                (
                    imag_bands[band][:, window, :]
                    - true_bands[band][:, window, :]
                )
                ** 2
            )
        )
        for band in BANDS
    }


def raw_mse_over_window(
    true_deter: np.ndarray,
    imag_deter: np.ndarray,
    *,
    trim_start: int,
    trim_end_inclusive: int,
) -> float:
    diff = imag_deter[:, trim_start : trim_end_inclusive + 1, :] - true_deter[
        :, trim_start : trim_end_inclusive + 1, :
    ]
    return float(np.mean(diff**2))


def percent_reduction(original: float, damped: float) -> float:
    if original == 0:
        return 0.0 if damped == 0 else float("-inf")
    return float((original - damped) / original * 100.0)


def subspace_overlap(u_a: np.ndarray, u_b: np.ndarray) -> float:
    if u_a.shape != u_b.shape:
        raise ValueError(f"Basis shape mismatch: {u_a.shape} vs {u_b.shape}.")
    rank = u_a.shape[1]
    overlap = np.trace((u_a @ u_a.T) @ (u_b @ u_b.T)) / rank
    return float(np.clip(overlap, 0.0, 1.0))


def assert_orthonormal(matrix: np.ndarray, name: str, *, atol: float = 1e-10) -> None:
    gram = matrix.T @ matrix
    error = float(np.max(np.abs(gram - np.eye(gram.shape[0]))))
    if error > atol:
        raise RuntimeError(f"{name} is not orthonormal; max_abs error={error:.3e}.")


def build_diagnostic_table(
    *,
    ranks: tuple[int, ...],
    overlaps: dict[str, float],
    old_basis_eval: dict[str, Any],
    new_basis_eval: dict[str, Any],
) -> list[dict[str, str | float]]:
    return [
        {
            "metric": f"baseline/post_smad overlap r={rank}",
            "value": float(overlaps[str(rank)]),
        }
        for rank in ranks
    ] + [
        {
            "metric": "Post-hoc damping with NEW basis: dc_trend reduction",
            "value": float(new_basis_eval["dc_trend_reduction_pct"]),
        },
        {
            "metric": "Post-hoc damping with OLD basis: dc_trend reduction",
            "value": float(old_basis_eval["dc_trend_reduction_pct"]),
        },
    ]


def load_original_comparison(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "available": False,
            "path": relative_path(path),
            "reason": "original result file not found",
        }
    try:
        data = json.loads(path.read_text())
        overlaps = {
            str(rank): float(data["overlaps"][str(rank)]["baseline_vs_post_smad"])
            for rank in RANKS
            if str(rank) in data.get("overlaps", {})
        }
        old_basis = data.get("posthoc_damping", {}).get("old_basis", {})
        new_basis = data.get("posthoc_damping", {}).get("new_basis", {})
        rollouts = data.get("rollouts", {})
        return {
            "available": True,
            "path": relative_path(path),
            "rollout_eta": rollouts.get("rollout_eta", rollouts.get("inference_eta")),
            "overlaps": overlaps,
            "posthoc_damping": {
                "old_basis_dc_trend_reduction_pct": old_basis.get(
                    "dc_trend_reduction_pct"
                ),
                "new_basis_dc_trend_reduction_pct": new_basis.get(
                    "dc_trend_reduction_pct"
                ),
            },
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {
            "available": False,
            "path": relative_path(path),
            "reason": f"could not parse original result: {exc}",
        }


def build_side_by_side_comparison(
    *,
    original: dict[str, Any],
    ranks: tuple[int, ...],
    overlaps: dict[str, float],
    old_basis_eval: dict[str, Any],
    new_basis_eval: dict[str, Any],
) -> list[dict[str, Any]]:
    original_overlaps = original.get("overlaps") or {}
    original_posthoc = original.get("posthoc_damping") or {}
    rows = [
        {
            "metric": f"overlap r={rank}",
            "original_eta020_rollouts": maybe_float(original_overlaps.get(str(rank))),
            "corrected_eta0_rollouts": float(overlaps[str(rank)]),
        }
        for rank in ranks
    ]
    rows.extend(
        [
            {
                "metric": "New basis post-hoc dc_trend reduction",
                "original_eta020_rollouts": maybe_float(
                    original_posthoc.get("new_basis_dc_trend_reduction_pct")
                ),
                "corrected_eta0_rollouts": float(
                    new_basis_eval["dc_trend_reduction_pct"]
                ),
            },
            {
                "metric": "Old basis post-hoc dc_trend reduction",
                "original_eta020_rollouts": maybe_float(
                    original_posthoc.get("old_basis_dc_trend_reduction_pct")
                ),
                "corrected_eta0_rollouts": float(
                    old_basis_eval["dc_trend_reduction_pct"]
                ),
            },
        ]
    )
    return rows


def maybe_float(value: Any) -> float | None:
    return None if value is None else float(value)


def interpret_result(
    *,
    rank10_overlap: float,
    new_reduction: float,
    old_reduction: float,
    approx_equal_pct_points: float,
) -> dict[str, Any]:
    delta = float(new_reduction - old_reduction)
    if rank10_overlap < 0.3:
        verdict = "STALENESS confirmed; Adaptive-SMAD should work."
    elif rank10_overlap <= 0.6:
        verdict = "MIXED: staleness plus adaptation both plausible."
    elif rank10_overlap > 0.7:
        verdict = "ADAPTATION primary; staleness in damped rollouts was likely an artifact."
    else:
        verdict = "MIXED: overlap falls between the pre-registered decision bands."
    return {
        "rank10_overlap": float(rank10_overlap),
        "new_minus_old_dc_reduction_pct_points": delta,
        "approx_equal_pct_points": float(approx_equal_pct_points),
        "rule": (
            "corrected r10 overlap < 0.3 => staleness confirmed; "
            "0.3-0.6 => mixed staleness + adaptation; > 0.7 => adaptation "
            "primary; 0.6-0.7 remains mixed."
        ),
        "verdict": verdict,
    }


def print_summary(result: dict[str, Any]) -> None:
    print("B1: Post-SMAD U_drift staleness check")
    print(f"Checkpoint: {result['checkpoint']['path'] or 'not available'}")
    print(f"Checkpoint status: {result['checkpoint']['status']}")
    print(f"Rollout source: {result['rollouts']['collection_status']}")
    print(f"Rollout eta: {result['rollouts']['rollout_eta']}")
    print()
    if result.get("side_by_side_comparison"):
        rollout_eta = float(result["rollouts"]["rollout_eta"])
        current_label = (
            "Corrected (eta=0 rollouts)"
            if abs(rollout_eta) < 1e-12
            else f"Current analyzed rollouts (eta={rollout_eta:g})"
        )
        print(f"| Metric | Original (eta=0.20 rollouts) | {current_label} |")
        print("|---|---:|---:|")
        for row in result["side_by_side_comparison"]:
            print(
                f"| {row['metric']} | "
                f"{format_optional_float(row['original_eta020_rollouts'])} | "
                f"{format_optional_float(row['corrected_eta0_rollouts'])} |"
            )
    else:
        print("| Metric | Value |")
        print("|---|---:|")
        for row in result["diagnostic_table"]:
            value = row["value"]
            if isinstance(value, float):
                print(f"| {row['metric']} | {value:.4f} |")
            else:
                print(f"| {row['metric']} | {value} |")
    print()
    print(f"Verdict: {result['interpretation']['verdict']}")


def format_optional_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    main()
