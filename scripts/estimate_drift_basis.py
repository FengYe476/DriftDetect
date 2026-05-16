#!/usr/bin/env python3
"""Estimate a fixed low-rank deterministic drift basis for SP-DPR.

The saved basis has shape ``(deter_dim, n_dims)`` with orthonormal columns.
All RSSM posterior tensors are batch-major:

    post["deter"] shape is [B, T, D]
    state_t = {k: v[:, t].detach() for k, v in post.items()}
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DREAMERV3_ROOT = PROJECT_ROOT / "external" / "dreamerv3-torch"

if "DRIFTDETECT_MUJOCO_GL" not in os.environ and sys.platform == "darwin":
    os.environ["DRIFTDETECT_MUJOCO_GL"] = "disable"
if "DRIFTDETECT_MUJOCO_GL" in os.environ:
    os.environ["MUJOCO_GL"] = os.environ["DRIFTDETECT_MUJOCO_GL"]


class NullLogger:
    """Minimal logger surface required by NM512's Dreamer constructor."""

    step = 0

    def scalar(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def video(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def write(self, *_args: Any, **_kwargs: Any) -> None:
        pass


def main() -> None:
    args = parse_args()
    ensure_sklearn_available()
    for path in (DREAMERV3_ROOT, PROJECT_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    import dreamer
    import torch

    checkpoint = resolve_path(args.checkpoint)
    config_path = resolve_path(args.config)
    output_path = resolve_path(args.output)
    replay_dir = resolve_path(args.replay_dir) if args.replay_dir else checkpoint.parent / "train_eps"
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    config = load_run_config(
        dreamer_module=dreamer,
        config_path=config_path,
        checkpoint=checkpoint,
        replay_dir=replay_dir,
        device=device,
    )
    dreamer.tools.set_seed_everywhere(config.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print_header(checkpoint, config_path, replay_dir, output_path, config, args)
    agent, env = build_agent(dreamer, torch, config, checkpoint)
    try:
        basis_inputs = collect_basis_inputs(
            dreamer_module=dreamer,
            torch_module=torch,
            world_model=agent._wm,
            config=config,
            replay_dir=replay_dir,
            num_batches=args.num_batches,
            horizon=args.horizon,
            dataset_limit=args.dataset_limit,
        )
    finally:
        try:
            env.close()
        except Exception:
            pass

    probes = fit_probe_bases(
        deter=basis_inputs["deter"],
        velocity=basis_inputs["velocity"],
        position=basis_inputs["position"],
        ridge_alpha=args.ridge_alpha,
        svd_tol=args.svd_tol,
    )
    Q_probed = orthonormal_columns(
        np.concatenate([probes["velocity"]["basis"], probes["position"]["basis"]], axis=1),
        args.svd_tol,
    )

    drift = basis_inputs["drift"]
    residual = remove_subspace(drift, Q_probed)
    Q_residual = residual_pca_basis(
        residual,
        residual_k=args.residual_k,
        seed=args.seed,
    )
    Q_full = orthonormal_columns(np.concatenate([Q_probed, Q_residual], axis=1), args.svd_tol)
    validate_basis(Q_full)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, Q_full.astype(np.float32))

    print_results(
        probes=probes,
        Q_probed=Q_probed,
        Q_residual=Q_residual,
        Q_full=Q_full,
        num_probe_samples=basis_inputs["deter"].shape[0],
        num_drift_vectors=drift.shape[0],
        output_path=output_path,
    )


def collect_basis_inputs(
    *,
    dreamer_module: Any,
    torch_module: Any,
    world_model: Any,
    config: SimpleNamespace,
    replay_dir: Path,
    num_batches: int,
    horizon: int,
    dataset_limit: int | None,
) -> dict[str, np.ndarray]:
    episodes = dreamer_module.tools.load_episodes(
        replay_dir,
        limit=dataset_limit or config.dataset_size,
    )
    if not episodes:
        raise FileNotFoundError(f"No replay episodes found in {replay_dir}")

    dataset = dreamer_module.make_dataset(episodes, config)
    deter_batches: list[np.ndarray] = []
    velocity_batches: list[np.ndarray] = []
    position_batches: list[np.ndarray] = []
    drift_batches: list[np.ndarray] = []

    with torch_module.no_grad():
        for batch_idx in range(num_batches):
            raw_data = next(dataset)
            data = world_model.preprocess(raw_data)
            require_physical_keys(data)
            embed = world_model.encoder(data)
            post, _prior = world_model.dynamics.observe(
                embed,
                data["action"],
                data["is_first"],
            )
            actions = data["action"]
            assert_batch_major(post, actions)
            B, T, D = post["deter"].shape
            check_physical_shape(data, B, T)

            deter_batches.append(post["deter"].detach().cpu().numpy().reshape(B * T, D))
            velocity_batches.append(data["velocity"].detach().cpu().numpy().reshape(B * T, -1))
            position_batches.append(data["position"].detach().cpu().numpy().reshape(B * T, -1))

            drift_batch = collect_drift_vectors(
                dynamics=world_model.dynamics,
                post=post,
                actions=actions,
                horizon=horizon,
            )
            drift_batches.append(drift_batch)
            print(
                f"Collected batch {batch_idx + 1}/{num_batches}: "
                f"B={B}, T={T}, D={D}, drift_vectors={drift_batch.shape[0]}",
                flush=True,
            )

    drift = np.concatenate(drift_batches, axis=0).astype(np.float64, copy=False)
    if drift.shape[0] == 0:
        raise ValueError(f"No drift vectors collected. Check horizon={horizon} and replay length.")
    return {
        "deter": np.concatenate(deter_batches, axis=0).astype(np.float64, copy=False),
        "velocity": np.concatenate(velocity_batches, axis=0).astype(np.float64, copy=False),
        "position": np.concatenate(position_batches, axis=0).astype(np.float64, copy=False),
        "drift": drift,
    }


def collect_drift_vectors(
    *,
    dynamics: Any,
    post: Mapping[str, Any],
    actions: Any,
    horizon: int,
) -> np.ndarray:
    post_deter = post["deter"]
    B, T, D = post_deter.shape
    if T <= horizon:
        return np.zeros((0, D), dtype=np.float32)

    drift_vectors: list[np.ndarray] = []
    for t_start in range(T - horizon):
        state = state_at(post, t_start)
        for step in range(horizon):
            action_step = actions[:, t_start + step].detach()
            state = detach_state(dynamics.img_step(state, action_step, sample=False))
            target_deter = post_deter[:, t_start + step + 1].detach()
            delta = state["deter"] - target_deter
            drift_vectors.append(delta.detach().cpu().numpy().reshape(B, D))
    return np.concatenate(drift_vectors, axis=0).astype(np.float32, copy=False)


def fit_probe_bases(
    *,
    deter: np.ndarray,
    velocity: np.ndarray,
    position: np.ndarray,
    ridge_alpha: float,
    svd_tol: float,
) -> dict[str, dict[str, Any]]:
    from sklearn.linear_model import Ridge

    probes = {}
    for name, target in (("velocity", velocity), ("position", position)):
        model = Ridge(alpha=ridge_alpha)
        model.fit(deter, target)
        coef = np.asarray(model.coef_, dtype=np.float64)
        if coef.ndim == 1:
            coef = coef[None, :]
        row_basis, singular_values, rank = row_space_basis(coef, svd_tol)
        probes[name] = {
            "r2": float(model.score(deter, target)),
            "coef": coef,
            "singular_values": singular_values,
            "rank": int(rank),
            "basis": row_basis.T,
        }
    return probes


def residual_pca_basis(residual: np.ndarray, *, residual_k: int, seed: int) -> np.ndarray:
    from sklearn.decomposition import PCA

    n_components = min(int(residual_k), residual.shape[0], residual.shape[1])
    if n_components <= 0:
        raise ValueError(f"residual_k must be positive, got {residual_k}.")
    if n_components != residual_k:
        print(
            f"Warning: requested residual_k={residual_k}, using n_components={n_components}.",
            flush=True,
        )
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(residual)
    return pca.components_.T.astype(np.float64, copy=False)


def row_space_basis(coef: np.ndarray, svd_tol: float) -> tuple[np.ndarray, np.ndarray, int]:
    _u, singular_values, vh = np.linalg.svd(coef, full_matrices=False)
    if singular_values.size == 0:
        rank = 0
    else:
        rank = int(np.sum(singular_values > svd_tol * max(float(singular_values[0]), 1.0)))
    return vh[:rank].astype(np.float64, copy=False), singular_values, rank


def orthonormal_columns(matrix: np.ndarray, svd_tol: float) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got {matrix.shape}.")
    if matrix.shape[1] == 0:
        return np.zeros((matrix.shape[0], 0), dtype=np.float64)
    u, singular_values, _vh = np.linalg.svd(matrix, full_matrices=False)
    if singular_values.size == 0:
        rank = 0
    else:
        rank = int(np.sum(singular_values > svd_tol * max(float(singular_values[0]), 1.0)))
    return u[:, :rank].astype(np.float64, copy=False)


def remove_subspace(vectors: np.ndarray, basis: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    if basis.shape[1] == 0:
        return vectors
    projected = (vectors @ basis) @ basis.T
    return vectors - projected


def validate_basis(basis: np.ndarray) -> None:
    if basis.ndim != 2:
        raise ValueError(f"Basis must be 2D, got {basis.shape}.")
    if basis.shape[0] <= 0 or basis.shape[1] <= 0:
        raise ValueError(f"Basis must be nonempty, got {basis.shape}.")
    gram = basis.T @ basis
    err = float(np.max(np.abs(gram - np.eye(basis.shape[1]))))
    if err > 1e-5:
        raise ValueError(f"Estimated basis columns are not orthonormal enough: max error {err}.")


def state_at(post: Mapping[str, Any], t: int) -> dict[str, Any]:
    return {key: value[:, t].detach() for key, value in post.items()}


def detach_state(state: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value.detach() for key, value in state.items()}


def require_physical_keys(data: Mapping[str, Any]) -> None:
    missing = [key for key in ("position", "velocity") if key not in data]
    if missing:
        raise KeyError(f"Replay batch is missing required physical keys: {missing}")


def check_physical_shape(data: Mapping[str, Any], batch_size: int, time_steps: int) -> None:
    for key, expected_dim in (("position", 8), ("velocity", 9)):
        value = data[key]
        if value.ndim != 3:
            raise ValueError(f"Expected data['{key}'] [B,T,D], got {value.shape}")
        if value.shape[0] != batch_size or value.shape[1] != time_steps:
            raise ValueError(
                f"data['{key}'] must share post batch/time dimensions, got {value.shape} "
                f"vs B={batch_size}, T={time_steps}"
            )
        if value.shape[-1] != expected_dim:
            print(
                f"Warning: expected {key} dim {expected_dim}, got {value.shape[-1]}",
                flush=True,
            )


def assert_batch_major(post: Mapping[str, Any], actions: Any) -> None:
    post_deter = post["deter"]
    assert post_deter.ndim == 3, f"Expected post['deter'] [B,T,D], got {post_deter.shape}"
    assert actions.ndim == 3, f"Expected actions [B,T,A], got {actions.shape}"
    B, T, D = post_deter.shape
    del D
    assert B == actions.shape[0], f"Batch mismatch: post {B} vs action {actions.shape[0]}"
    assert T == actions.shape[1], f"Time mismatch: post {T} vs action {actions.shape[1]}"


def ensure_sklearn_available() -> None:
    try:
        from sklearn.decomposition import PCA  # noqa: F401
        from sklearn.linear_model import Ridge  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "estimate_drift_basis.py requires scikit-learn for Ridge probes and PCA."
        ) from exc


def load_run_config(
    *,
    dreamer_module: Any,
    config_path: Path,
    checkpoint: Path,
    replay_dir: Path,
    device: str,
) -> SimpleNamespace:
    loader = dreamer_module.yaml.YAML(typ="safe", pure=True)
    run_config = loader.load(config_path.read_text())
    dreamer_configs = loader.load((DREAMERV3_ROOT / "configs.yaml").read_text())

    legacy_dreamerv3 = run_config.get("dreamerv3", {}) or {}
    config_names = run_config.get("dreamer_configs")
    if config_names is None and legacy_dreamerv3:
        config_names = legacy_dreamerv3.get("configs")
    if config_names is None:
        config_names = ["dmc_proprio"]
    if isinstance(config_names, str):
        config_names = [part.strip() for part in config_names.split(",") if part.strip()]

    values: dict[str, Any] = {}
    for name in ["defaults", *list(config_names)]:
        recursive_update(values, dreamer_configs[name])
    if run_config.get("dreamer"):
        recursive_update(values, run_config["dreamer"])
    if legacy_dreamerv3:
        recursive_update(
            values,
            {
                key: value
                for key, value in legacy_dreamerv3.items()
                if key not in {"configs", "task", "steps", "seed"}
            },
        )
    recursive_update(
        values,
        {
            key: run_config[key]
            for key in (
                "batch_size",
                "batch_length",
                "envs",
                "compile",
                "eval_every",
                "log_every",
                "dataset_size",
                "action_repeat",
                "time_limit",
            )
            if key in run_config
        },
    )

    task = run_config.get("task", legacy_dreamerv3.get("task"))
    if task is None:
        raise KeyError("Config must define task or dreamerv3.task.")
    if not task.startswith("dmc_"):
        task = "dmc_" + task

    values["task"] = task
    values["seed"] = int(run_config.get("seed", legacy_dreamerv3.get("seed", 0)))
    values["device"] = device
    values["compile"] = False
    values["logdir"] = str(checkpoint.parent)
    values["traindir"] = replay_dir
    values["evaldir"] = checkpoint.parent / "eval_eps"
    return SimpleNamespace(**values)


def build_agent(dreamer_module: Any, torch_module: Any, config: SimpleNamespace, checkpoint: Path):
    env = dreamer_module.make_env(config, "eval", 0)
    config.num_actions = (
        env.action_space.n if hasattr(env.action_space, "n") else env.action_space.shape[0]
    )
    agent = dreamer_module.Dreamer(
        env.observation_space,
        env.action_space,
        config,
        logger=NullLogger(),
        dataset=None,
    ).to(config.device)

    ckpt = load_checkpoint(torch_module, checkpoint, config.device)
    state_dict = ckpt.get("agent_state_dict", ckpt)
    agent.load_state_dict(normalize_checkpoint_state_dict(state_dict))
    agent.requires_grad_(requires_grad=False)
    agent.eval()
    return agent, env


def print_header(
    checkpoint: Path,
    config_path: Path,
    replay_dir: Path,
    output_path: Path,
    config: SimpleNamespace,
    args: argparse.Namespace,
) -> None:
    print("=" * 78)
    print("DreamerV3 Drift Basis Estimation for SP-DPR")
    print("=" * 78)
    print(f"Task: {config.task}, seed: {config.seed}, device: {config.device}")
    print(f"Checkpoint: {display_path(checkpoint)}")
    print(f"Config: {display_path(config_path)}")
    print(f"Replay: {display_path(replay_dir)}")
    print(f"Batches: {args.num_batches}, horizon: {args.horizon}, residual_k: {args.residual_k}")
    print(f"Output: {display_path(output_path)}")
    print("=" * 78, flush=True)


def print_results(
    *,
    probes: Mapping[str, Mapping[str, Any]],
    Q_probed: np.ndarray,
    Q_residual: np.ndarray,
    Q_full: np.ndarray,
    num_probe_samples: int,
    num_drift_vectors: int,
    output_path: Path,
) -> None:
    print("\nProbe fits")
    print_table(
        ["probe", "R^2", "rank", "coef_shape"],
        [
            [
                name,
                fmt(probe["r2"]),
                str(probe["rank"]),
                str(list(probe["coef"].shape)),
            ]
            for name, probe in probes.items()
        ],
    )
    print("\nBasis summary")
    print_table(
        ["field", "value"],
        [
            ["probe samples", str(num_probe_samples)],
            ["drift vectors", str(num_drift_vectors)],
            ["Q_probed shape", str(list(Q_probed.shape))],
            ["Q_residual shape", str(list(Q_residual.shape))],
            ["Q_full shape", str(list(Q_full.shape))],
            ["orthonormal max error", fmt(float(np.max(np.abs(Q_full.T @ Q_full - np.eye(Q_full.shape[1])))))],
            ["saved", display_path(output_path)],
        ],
    )


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    rows = [[str(cell) for cell in row] for row in rows]
    widths = [
        max(len(str(header)), *(len(row[i]) for row in rows)) if rows else len(str(header))
        for i, header in enumerate(headers)
    ]
    fmt_row = "  ".join(f"{{:<{width}}}" for width in widths)
    print(fmt_row.format(*headers))
    print(fmt_row.format(*["-" * width for width in widths]))
    for row in rows:
        print(fmt_row.format(*row))


def recursive_update(base: dict[str, Any], update: Mapping[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            recursive_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)


def normalize_checkpoint_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    return {key.replace("._orig_mod.", "."): value for key, value in state_dict.items()}


def load_checkpoint(torch_module: Any, checkpoint: Path, device: str) -> Mapping[str, Any]:
    try:
        return torch_module.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        return torch_module.load(checkpoint, map_location=device)


def fmt(value: Any) -> str:
    value = float(value)
    if math.isnan(value):
        return "nan"
    if value == 0.0:
        return "0"
    abs_value = abs(value)
    if abs_value < 1e-3 or abs_value >= 1e4:
        return f"{value:.4e}"
    return f"{value:.6f}"


def resolve_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate SP-DPR drift basis.")
    parser.add_argument("--checkpoint", required=True, help="Path to DreamerV3 latest.pt.")
    parser.add_argument("--config", required=True, help="Training config for model architecture.")
    parser.add_argument("--residual_k", type=int, default=16, help="Residual PCA rank.")
    parser.add_argument("--output", required=True, help="Output .npy basis path.")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda:0 or cpu.")
    parser.add_argument("--replay_dir", default=None, help="Defaults to CHECKPOINT_PARENT/train_eps.")
    parser.add_argument("--num_batches", type=int, default=3, help="Replay batches to collect.")
    parser.add_argument("--horizon", type=int, default=50, help="Open-loop drift horizon.")
    parser.add_argument("--ridge_alpha", type=float, default=1.0, help="sklearn Ridge alpha.")
    parser.add_argument("--svd_tol", type=float, default=1e-8, help="Relative SVD rank tolerance.")
    parser.add_argument("--dataset_limit", type=int, default=None, help="Optional replay step limit.")
    parser.add_argument("--seed", type=int, default=0, help="Diagnostic random seed.")
    args = parser.parse_args()
    if args.residual_k <= 0:
        parser.error("--residual_k must be positive.")
    if args.num_batches <= 0:
        parser.error("--num_batches must be positive.")
    if args.horizon <= 0:
        parser.error("--horizon must be positive.")
    if args.ridge_alpha <= 0:
        parser.error("--ridge_alpha must be positive.")
    if args.svd_tol <= 0:
        parser.error("--svd_tol must be positive.")
    return args


if __name__ == "__main__":
    main()
