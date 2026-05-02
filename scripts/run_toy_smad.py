#!/usr/bin/env python3
"""Run the toy SMAD validation experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.toy.dual_oscillator_env import DualOscillatorEnv  # noqa: E402
from src.toy.minimal_rssm import MinimalRSSM  # noqa: E402
from src.toy.smad_intervention import (  # noqa: E402
    analyze_drift,
    apply_damping,
    collect_rollouts,
    sample_episode,
)


DEFAULT_OUTPUT = REPO_ROOT / "results" / "tables" / "toy_smad_results.json"


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)
    env = DualOscillatorEnv(seed=args.seed, episode_length=args.horizon)
    rssm = MinimalRSSM(device=device, lr=args.lr)

    print("Toy SMAD validation", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Training steps: {args.train_steps}", flush=True)
    print(f"Horizon: {args.horizon}", flush=True)
    print(f"Evaluation episodes: {args.eval_episodes}", flush=True)
    for step in range(1, args.train_steps + 1):
        obs, actions, rewards = sample_batch(env, args.batch_size, args.horizon)
        metrics = rssm.train_step(
            obs,
            actions,
            reward_seq=rewards,
            free_bits=args.free_bits,
            kl_scale=args.kl_scale,
            sample=True,
        )
        if step == 1 or step % 100 == 0 or step == args.train_steps:
            print(
                f"  step {step}/{args.train_steps}  "
                f"loss={metrics['loss']:.4f}  "
                f"recon={metrics['recon_loss']:.4f}  "
                f"kl={metrics['kl_loss']:.4f}",
                flush=True,
            )

    baseline_rollouts = collect_rollouts(
        rssm,
        env,
        n_episodes=args.eval_episodes,
        horizon=args.horizon,
    )
    baseline_analysis = analyze_drift(
        baseline_rollouts["true_deter"],
        baseline_rollouts["imagined_deter"],
        trim=args.trim,
    )
    U_drift = baseline_analysis["bases"]["U_drift"][:, : args.rank]

    damping_results = []
    for eta in args.etas:
        result = apply_damping(
            rssm,
            U_drift,
            eta=float(eta),
            env=env,
            n_episodes=args.eval_episodes,
            horizon=args.horizon,
            baseline_rollouts=baseline_rollouts,
            trim=args.trim,
        )
        damping_results.append(result)

    output = {
        "analysis": "toy_smad_validation",
        "seed": int(args.seed),
        "device": str(device),
        "train_steps": int(args.train_steps),
        "batch_size": int(args.batch_size),
        "horizon": int(args.horizon),
        "eval_episodes": int(args.eval_episodes),
        "trim_window_inclusive": [int(args.trim[0]), int(args.trim[1])],
        "rank": int(args.rank),
        "environment": {
            "f_slow": env.f_slow,
            "f_fast": env.f_fast,
            "A_slow": env.A_slow,
            "A_fast": env.A_fast,
            "dt": env.dt,
            "noise_std": env.noise_std,
        },
        "baseline_analysis": baseline_analysis,
        "damping": damping_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(to_jsonable(output), indent=2) + "\n")
    print_summary(baseline_analysis, damping_results)
    print(f"Saved: {relative_path(args.output)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run toy SMAD validation.")
    parser.add_argument("--train_steps", type=int, default=2_000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--eval_episodes", "--n_episodes", dest="eval_episodes", type=int, default=10)
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--etas", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    parser.add_argument("--trim", type=int, nargs=2, default=(5, 45))
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--free_bits", type=float, default=0.0)
    parser.add_argument("--kl_scale", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output = resolve_path(args.output)
    args.trim = (int(args.trim[0]), int(args.trim[1]))
    if args.train_steps <= 0:
        raise ValueError("--train_steps must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.eval_episodes <= 0:
        raise ValueError("--eval_episodes/--n_episodes must be positive.")
    if args.horizon <= 0:
        raise ValueError("--horizon must be positive.")
    if not 0 <= args.trim[0] <= args.trim[1] < args.horizon:
        raise ValueError("--trim must be inside the horizon.")
    return args


def sample_batch(
    env: DualOscillatorEnv,
    batch_size: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs = []
    actions = []
    rewards = []
    for _ in range(batch_size):
        episode_obs, episode_actions, episode_rewards = sample_episode(env, horizon)
        obs.append(episode_obs)
        actions.append(episode_actions)
        rewards.append(episode_rewards)
    return (
        np.asarray(obs, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
        np.asarray(rewards, dtype=np.float32),
    )


def select_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_summary(
    baseline_analysis: dict[str, Any],
    damping_results: list[dict[str, Any]],
) -> None:
    freq = baseline_analysis["frequency"]
    print("\nToy SMAD summary", flush=True)
    print("----------------", flush=True)
    print("Subspace overlaps:", flush=True)
    for rank, values in baseline_analysis["overlaps"].items():
        print(
            f"  r={rank}: "
            f"SFA/post={values['sfa_posterior']:.4f} "
            f"SFA/drift={values['sfa_drift']:.4f} "
            f"post/drift={values['posterior_drift']:.4f}",
            flush=True,
        )
    print("Baseline frequency MSE:", flush=True)
    for band, value in freq["band_mse"].items():
        share = 100.0 * freq["band_share"][band]
        print(f"  {band:<10} {value:.6f} ({share:.1f}%)", flush=True)
    print("Damping:", flush=True)
    for result in damping_results:
        print(
            f"  eta={result['eta']:.2f}: "
            f"J_slow reduction={result['J_slow_reduction_pct']:.1f}% "
            f"J_total reduction={result['J_total_reduction_pct']:.1f}%",
            flush=True,
        )


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return relative_path(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    main()
