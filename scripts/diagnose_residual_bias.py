#!/usr/bin/env python3
"""Diagnose one-step RSSM residual bias in DreamerV3 replay trajectories.

The residual path intentionally mirrors the corrected SHARP indexing:

    post["deter"] shape is [B, T, D]
    state_t = {k: v[:, t].detach() for k, v in post.items()}
    residual_t = dynamics.img_step(state_t, action[:, t])["deter"]
                 - post["deter"][:, t + 1]

All computations run under torch.no_grad(); this script is diagnostic only.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DREAMERV3_ROOT = PROJECT_ROOT / "external" / "dreamerv3-torch"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "tables" / "residual_bias_diagnostics.json"

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
    for path in (DREAMERV3_ROOT, PROJECT_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    import dreamer
    import torch

    checkpoint = resolve_path(args.checkpoint)
    config_path = resolve_path(args.config)
    output_path = resolve_path(args.output)
    replay_dir = resolve_path(args.replay_dir) if args.replay_dir else checkpoint.parent / "train_eps"

    config = load_run_config(
        dreamer_module=dreamer,
        config_path=config_path,
        checkpoint=checkpoint,
        replay_dir=replay_dir,
        device=args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"),
    )
    tools = dreamer.tools
    tools.set_seed_everywhere(config.seed)

    print_header(checkpoint, config_path, replay_dir, output_path, config, args)

    agent, env = build_agent(dreamer, torch, config, checkpoint)
    try:
        residuals = collect_residual_trajectories(
            dreamer_module=dreamer,
            torch_module=torch,
            agent=agent,
            config=config,
            replay_dir=replay_dir,
            num_batches=args.num_batches,
            dataset_limit=args.dataset_limit,
        )
    finally:
        try:
            env.close()
        except Exception:
            pass

    results = run_diagnostics(
        residuals=residuals,
        lags=args.lags,
        shuffle_repeats=args.shuffle_repeats,
        linear_max_samples=args.linear_max_samples,
        seed=args.seed,
    )
    results["metadata"] = {
        "checkpoint": str(checkpoint),
        "config": str(config_path),
        "replay_dir": str(replay_dir),
        "output": str(output_path),
        "device": str(config.device),
        "task": str(config.task),
        "seed": int(config.seed),
        "num_batches": int(args.num_batches),
        "dataset_limit": None if args.dataset_limit is None else int(args.dataset_limit),
        "residual_shape": list(residuals.shape),
        "shape_convention": "residuals [B_total, T_minus_1, D]",
    }

    print_results(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"\nSaved residual bias diagnostics to {display_path(output_path)}")


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


def collect_residual_trajectories(
    *,
    dreamer_module: Any,
    torch_module: Any,
    agent: Any,
    config: SimpleNamespace,
    replay_dir: Path,
    num_batches: int,
    dataset_limit: int | None,
) -> Any:
    tools = dreamer_module.tools
    episodes = tools.load_episodes(replay_dir, limit=dataset_limit or config.dataset_size)
    if not episodes:
        raise FileNotFoundError(f"No replay episodes found in {replay_dir}")

    dataset = dreamer_module.make_dataset(episodes, config)
    residual_batches = []
    with torch_module.no_grad():
        for batch_idx in range(num_batches):
            raw_data = next(dataset)
            data = agent._wm.preprocess(raw_data)
            embed = agent._wm.encoder(data)
            post, _prior = agent._wm.dynamics.observe(
                embed,
                data["action"],
                data["is_first"],
            )
            residuals = compute_residual_batch(
                dynamics=agent._wm.dynamics,
                post=post,
                actions=data["action"],
                torch_module=torch_module,
            )
            residual_batches.append(residuals.detach().cpu())
            print(
                f"Collected batch {batch_idx + 1}/{num_batches}: "
                f"residuals {tuple(residuals.shape)}",
                flush=True,
            )

    return torch_module.cat(residual_batches, dim=0)


def compute_residual_batch(*, dynamics: Any, post: Mapping[str, Any], actions: Any, torch_module: Any):
    post_deter = post["deter"]
    if not isinstance(actions, torch_module.Tensor):
        actions = torch_module.as_tensor(actions, device=post_deter.device)

    assert post_deter.ndim == 3, f"Expected post['deter'] [B,T,D], got {post_deter.shape}"
    assert actions.ndim == 3, f"Expected actions [B,T,A], got {actions.shape}"

    # post["deter"] shape: [B, T, D] where B=batch, T=time, D=latent_dim
    B, T, _D = post_deter.shape
    assert B == actions.shape[0], f"Batch mismatch: post {B} vs action {actions.shape[0]}"
    assert T == actions.shape[1], f"Time mismatch: post {T} vs action {actions.shape[1]}"
    if T < 2:
        raise ValueError("Need at least two time steps to compute one-step residuals.")

    residuals = []
    for t in range(T - 1):
        state_t = {key: value[:, t].detach() for key, value in post.items()}
        action_t = actions[:, t].detach()
        pred_next = dynamics.img_step(state_t, action_t, sample=False)
        residual_t = pred_next["deter"] - post_deter[:, t + 1].detach()
        residuals.append(residual_t)
    return torch_module.stack(residuals, dim=1)


def run_diagnostics(
    *,
    residuals: Any,
    lags: Sequence[int],
    shuffle_repeats: int,
    linear_max_samples: int,
    seed: int,
) -> dict[str, Any]:
    import torch

    residuals = residuals.detach().to(dtype=torch.float64)
    return {
        "per_trajectory_mean": diagnostic_per_trajectory_mean(residuals),
        "autocorrelation": diagnostic_autocorrelation(residuals, lags),
        "low_frequency_energy": diagnostic_low_frequency_energy(residuals),
        "shuffle_test": diagnostic_shuffle_test(
            residuals,
            repeats=shuffle_repeats,
            seed=seed,
        ),
        "conditional_predictor": diagnostic_conditional_predictor(
            residuals,
            max_samples=linear_max_samples,
            seed=seed,
        ),
    }


def diagnostic_per_trajectory_mean(residuals: Any) -> dict[str, Any]:
    import torch

    residual_mean = residuals.mean(dim=1)
    norms = residual_mean.norm(dim=-1)
    mean_energy_time_expanded = residual_mean[:, None, :].pow(2).sum()
    total_energy = residuals.pow(2).sum().clamp_min(1e-30)
    global_bias = residual_mean.mean(dim=0)
    top_dims = top_abs_indices(global_bias, k=min(10, global_bias.numel()))
    return {
        "trajectory_norms": summarize_tensor(norms),
        "global_bias_norm": scalar(global_bias.norm()),
        "global_bias_direction": tensor_list(global_bias),
        "global_bias_top_abs_dims": top_dims,
        "mean_energy_fraction": scalar(mean_energy_time_expanded / total_energy),
        "total_residual_energy": scalar(total_energy),
    }


def diagnostic_autocorrelation(residuals: Any, lags: Sequence[int]) -> dict[str, Any]:
    import torch

    _, T, _ = residuals.shape
    by_lag: dict[str, Any] = {}
    for lag in lags:
        lag = int(lag)
        if lag <= 0 or lag >= T:
            by_lag[str(lag)] = {"valid": False, "mean": 0.0, "reason": "lag outside sequence"}
            continue
        x = residuals[:, :-lag, :].reshape(-1, residuals.shape[-1])
        y = residuals[:, lag:, :].reshape(-1, residuals.shape[-1])
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)
        denom = x.pow(2).mean(dim=0).sqrt() * y.pow(2).mean(dim=0).sqrt()
        corr = (x * y).mean(dim=0) / denom.clamp_min(1e-12)
        by_lag[str(lag)] = {"valid": True, **summarize_tensor(corr)}
    return {"lags": by_lag}


def diagnostic_low_frequency_energy(residuals: Any) -> dict[str, Any]:
    array = residuals.detach().cpu().numpy()
    coeff = dct_time(array)
    energy_by_coeff = np.square(coeff).sum(axis=(0, 2))
    total = float(max(energy_by_coeff.sum(), 1e-30))
    T = int(coeff.shape[1])
    bands = {
        "dc_trend": (0, min(2, T)),
        "very_low": (min(2, T), min(5, T)),
        "low": (min(5, T), min(10, T)),
        "mid": (min(10, T), min(20, T)),
        "high": (min(20, T), T),
    }
    result: dict[str, Any] = {"total_energy": total, "bands": {}}
    for name, (start, stop) in bands.items():
        energy = float(energy_by_coeff[start:stop].sum()) if stop > start else 0.0
        result["bands"][name] = {
            "coeff_range": [int(start), int(stop)],
            "energy": energy,
            "fraction": energy / total,
        }
    return result


def diagnostic_shuffle_test(residuals: Any, *, repeats: int, seed: int) -> dict[str, Any]:
    import torch

    generator = torch.Generator(device=residuals.device)
    generator.manual_seed(seed)

    drift_original = residuals.cumsum(dim=1)
    original_final = drift_original[:, -1, :].pow(2).sum(dim=-1)
    original_path = drift_original.pow(2).sum(dim=-1).mean(dim=1)

    shuffled_final_values = []
    shuffled_path_values = []
    B, T, D = residuals.shape
    for _ in range(repeats):
        perms = torch.stack([torch.randperm(T, generator=generator) for _ in range(B)])
        shuffled = residuals[torch.arange(B)[:, None], perms, :].reshape(B, T, D)
        drift_shuffled = shuffled.cumsum(dim=1)
        shuffled_final_values.append(drift_shuffled[:, -1, :].pow(2).sum(dim=-1).mean())
        shuffled_path_values.append(drift_shuffled.pow(2).sum(dim=-1).mean(dim=1).mean())

    shuffled_final = torch.stack(shuffled_final_values)
    shuffled_path = torch.stack(shuffled_path_values)
    original_final_mean = original_final.mean()
    original_path_mean = original_path.mean()
    return {
        "endpoint_note": (
            "The requested final cumsum MSE is invariant to time permutation when "
            "drift is modeled as a plain cumulative sum; path_mse is included as "
            "the temporal-order-sensitive companion metric."
        ),
        "original_final_mse": scalar(original_final_mean),
        "shuffled_final_mse_mean": scalar(shuffled_final.mean()),
        "shuffled_final_mse_std": scalar(shuffled_final.std(unbiased=False)),
        "final_ratio_original_over_shuffled": scalar(
            original_final_mean / shuffled_final.mean().clamp_min(1e-30)
        ),
        "original_path_mse": scalar(original_path_mean),
        "shuffled_path_mse_mean": scalar(shuffled_path.mean()),
        "shuffled_path_mse_std": scalar(shuffled_path.std(unbiased=False)),
        "path_ratio_original_over_shuffled": scalar(
            original_path_mean / shuffled_path.mean().clamp_min(1e-30)
        ),
        "repeats": int(repeats),
    }


def diagnostic_conditional_predictor(
    residuals: Any,
    *,
    max_samples: int,
    seed: int,
) -> dict[str, Any]:
    import torch

    X = residuals[:, :-1, :].reshape(-1, residuals.shape[-1])
    Y = residuals[:, 1:, :].reshape(-1, residuals.shape[-1])
    if X.shape[0] > max_samples > 0:
        generator = torch.Generator(device=X.device)
        generator.manual_seed(seed)
        idx = torch.randperm(X.shape[0], generator=generator)[:max_samples]
        X = X[idx]
        Y = Y[idx]

    X = X.cpu()
    Y = Y.cpu()
    ones = torch.ones((X.shape[0], 1), dtype=X.dtype)
    X_aug = torch.cat([X, ones], dim=1)
    solution = torch.linalg.lstsq(X_aug, Y).solution
    A = solution[:-1, :]
    b = solution[-1, :]
    pred = X_aug @ solution
    residual_sum_squares = (Y - pred).pow(2).sum()
    total_sum_squares = (Y - Y.mean(dim=0, keepdim=True)).pow(2).sum().clamp_min(1e-30)
    r2 = 1.0 - residual_sum_squares / total_sum_squares

    A_np = A.numpy()
    singular_values = np.linalg.svd(A_np, compute_uv=False)
    eigenvalues = np.linalg.eigvals(A_np)
    top_eigenvalues = sorted(eigenvalues, key=lambda z: abs(z), reverse=True)[:10]
    return {
        "num_samples": int(X.shape[0]),
        "latent_dim": int(X.shape[1]),
        "frobenius_norm": float(np.linalg.norm(A_np, ord="fro")),
        "spectral_norm": float(singular_values[0]) if singular_values.size else 0.0,
        "r2": scalar(r2),
        "bias_norm": scalar(b.norm()),
        "top_eigenvalues": [
            {"real": float(np.real(value)), "imag": float(np.imag(value)), "abs": float(abs(value))}
            for value in top_eigenvalues
        ],
    }


def dct_time(array: np.ndarray) -> np.ndarray:
    try:
        from scipy.fft import dct

        return dct(array, type=2, axis=1, norm="ortho")
    except Exception:
        return dct_time_numpy(array)


def dct_time_numpy(array: np.ndarray) -> np.ndarray:
    """Small DCT-II fallback for environments without scipy."""

    _, T, _ = array.shape
    n = np.arange(T, dtype=np.float64)
    k = np.arange(T, dtype=np.float64)[:, None]
    basis = np.cos((math.pi / T) * (n + 0.5) * k)
    basis[0] *= math.sqrt(1.0 / T)
    if T > 1:
        basis[1:] *= math.sqrt(2.0 / T)
    return np.einsum("kt,btd->bkd", basis, array, optimize=True)


def print_results(results: Mapping[str, Any]) -> None:
    shape = results["metadata"]["residual_shape"]
    print("\nResidual tensor")
    print_table(["field", "value"], [["shape", str(shape)], ["convention", "B_total, T-1, D"]])

    diag1 = results["per_trajectory_mean"]
    norm = diag1["trajectory_norms"]
    print("\nDiagnostic 1: Per-Trajectory Residual Mean")
    print_table(
        ["metric", "value"],
        [
            ["trajectory ||mean_t residual|| mean", fmt(norm["mean"])],
            ["trajectory ||mean_t residual|| std", fmt(norm["std"])],
            ["trajectory ||mean_t residual|| p50", fmt(norm["p50"])],
            ["trajectory ||mean_t residual|| p90", fmt(norm["p90"])],
            ["trajectory ||mean_t residual|| max", fmt(norm["max"])],
            ["global bias norm", fmt(diag1["global_bias_norm"])],
            ["mean energy fraction", fmt(diag1["mean_energy_fraction"])],
        ],
    )
    top_dims = diag1["global_bias_top_abs_dims"][:8]
    print_table(
        ["rank", "dim", "value", "abs"],
        [
            [str(i + 1), str(item["index"]), fmt(item["value"]), fmt(item["abs"])]
            for i, item in enumerate(top_dims)
        ],
    )

    print("\nDiagnostic 2: Residual Autocorrelation")
    rows = []
    for lag, stats in results["autocorrelation"]["lags"].items():
        if not stats["valid"]:
            rows.append([lag, "invalid", "", stats.get("reason", "")])
        else:
            rows.append([lag, fmt(stats["mean"]), fmt(stats["p50"]), fmt(stats["p90"])])
    print_table(["lag", "mean", "p50", "p90"], rows)

    print("\nDiagnostic 3: Residual Low-Frequency Energy")
    rows = []
    for name, band in results["low_frequency_energy"]["bands"].items():
        rows.append([name, str(band["coeff_range"]), fmt(band["energy"]), pct(band["fraction"])])
    print_table(["band", "coeffs", "energy", "fraction"], rows)

    diag4 = results["shuffle_test"]
    print("\nDiagnostic 4: Residual Shuffle Test")
    print_table(
        ["metric", "value"],
        [
            ["original final MSE", fmt(diag4["original_final_mse"])],
            ["shuffled final MSE mean", fmt(diag4["shuffled_final_mse_mean"])],
            ["shuffled final MSE std", fmt(diag4["shuffled_final_mse_std"])],
            ["final original/shuffled", fmt(diag4["final_ratio_original_over_shuffled"])],
            ["original path MSE", fmt(diag4["original_path_mse"])],
            ["shuffled path MSE mean", fmt(diag4["shuffled_path_mse_mean"])],
            ["shuffled path MSE std", fmt(diag4["shuffled_path_mse_std"])],
            ["path original/shuffled", fmt(diag4["path_ratio_original_over_shuffled"])],
        ],
    )
    print(f"Note: {diag4['endpoint_note']}")

    diag5 = results["conditional_predictor"]
    print("\nDiagnostic 5: Conditional Residual Predictor")
    print_table(
        ["metric", "value"],
        [
            ["num samples", str(diag5["num_samples"])],
            ["latent dim", str(diag5["latent_dim"])],
            ["||A||_F", fmt(diag5["frobenius_norm"])],
            ["||A||_2", fmt(diag5["spectral_norm"])],
            ["R^2", fmt(diag5["r2"])],
            ["||b||", fmt(diag5["bias_norm"])],
        ],
    )
    print_table(
        ["rank", "real", "imag", "abs"],
        [
            [str(i + 1), fmt(value["real"]), fmt(value["imag"]), fmt(value["abs"])]
            for i, value in enumerate(diag5["top_eigenvalues"][:8])
        ],
    )


def print_header(
    checkpoint: Path,
    config_path: Path,
    replay_dir: Path,
    output_path: Path,
    config: SimpleNamespace,
    args: argparse.Namespace,
) -> None:
    print("=" * 78)
    print("DreamerV3 Residual Bias Diagnostics")
    print("=" * 78)
    print(f"Task: {config.task}, seed: {config.seed}, device: {config.device}")
    print(f"Checkpoint: {display_path(checkpoint)}")
    print(f"Config: {display_path(config_path)}")
    print(f"Replay: {display_path(replay_dir)}")
    print(f"Batches: {args.num_batches}, shuffle repeats: {args.shuffle_repeats}")
    print(f"Output: {display_path(output_path)}")
    print("=" * 78, flush=True)


def summarize_tensor(x: Any) -> dict[str, float]:
    import torch

    flat = x.detach().reshape(-1).to(dtype=torch.float64)
    if flat.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "p05": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": scalar(flat.mean()),
        "std": scalar(flat.std(unbiased=False)),
        "p05": scalar(torch.quantile(flat, 0.05)),
        "p50": scalar(torch.quantile(flat, 0.50)),
        "p90": scalar(torch.quantile(flat, 0.90)),
        "p95": scalar(torch.quantile(flat, 0.95)),
        "max": scalar(flat.max()),
    }


def top_abs_indices(x: Any, k: int) -> list[dict[str, float | int]]:
    import torch

    flat = x.detach().reshape(-1)
    values, indices = torch.topk(flat.abs(), k=k)
    return [
        {
            "index": int(index.item()),
            "value": scalar(flat[index]),
            "abs": scalar(value),
        }
        for value, index in zip(values, indices)
    ]


def print_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
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


def scalar(x: Any) -> float:
    if hasattr(x, "detach"):
        return float(x.detach().cpu().item())
    return float(x)


def tensor_list(x: Any) -> list[float]:
    return [float(value) for value in x.detach().cpu().reshape(-1).tolist()]


def fmt(value: Any) -> str:
    value = float(value)
    if value == 0.0:
        return "0"
    abs_value = abs(value)
    if abs_value < 1e-3 or abs_value >= 1e4:
        return f"{value:.4e}"
    return f"{value:.6f}"


def pct(value: Any) -> str:
    return f"{100.0 * float(value):.2f}%"


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
    parser = argparse.ArgumentParser(
        description="Compute residual-bias diagnostics from DreamerV3 replay data."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to DreamerV3 checkpoint, e.g. results/baseline_100k/cheetah_seed42/latest.pt.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Training config used to build the DreamerV3 architecture.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device. Defaults to cuda:0 if available, otherwise cpu.",
    )
    parser.add_argument(
        "--replay_dir",
        default=None,
        help="Replay directory. Defaults to CHECKPOINT_PARENT/train_eps.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="JSON output path.",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=8,
        help="Number of replay batches to collect posterior residuals from.",
    )
    parser.add_argument(
        "--dataset_limit",
        type=int,
        default=None,
        help="Optional replay step limit passed to tools.load_episodes.",
    )
    parser.add_argument(
        "--shuffle_repeats",
        type=int,
        default=100,
        help="Number of residual time-shuffle repetitions.",
    )
    parser.add_argument(
        "--linear_max_samples",
        type=int,
        default=20000,
        help="Maximum residual pairs for the linear conditional predictor.",
    )
    parser.add_argument(
        "--lags",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16],
        help="Autocorrelation lags.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for shuffle and linear subsampling diagnostics.",
    )
    args = parser.parse_args()
    if args.num_batches <= 0:
        parser.error("--num_batches must be positive.")
    if args.shuffle_repeats <= 0:
        parser.error("--shuffle_repeats must be positive.")
    return args


if __name__ == "__main__":
    main()
