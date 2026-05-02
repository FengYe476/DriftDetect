#!/usr/bin/env python3
"""Toy complexity sweep for SMAD frequency-drift diagnostics.

The sweep keeps RSSM capacity fixed while increasing oscillator count and
observation dimensionality. It tests whether low-frequency drift share grows as
the environment/model capacity gap widens.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


sys.stdout.reconfigure(line_buffering=True)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.toy.minimal_rssm import MinimalRSSM  # noqa: E402
from src.toy.smad_intervention import (  # noqa: E402
    analyze_drift,
    apply_damping,
    collect_rollouts,
    sample_episode,
)


DEFAULT_OUTPUT = REPO_ROOT / "results" / "tables" / "toy_complexity_sweep.json"
BANDS = ("dc_trend", "very_low", "low", "mid", "high")


@dataclass(frozen=True)
class ComplexitySpec:
    name: str
    frequencies: tuple[float, ...]
    amplitudes: tuple[float, ...]
    coupling_strength: float
    seed_offset: int


class MultiOscillatorEnv:
    """N-oscillator toy environment with one shared action input."""

    def __init__(
        self,
        frequencies: list[float] | tuple[float, ...],
        amplitudes: list[float] | tuple[float, ...] | None = None,
        dt: float = 0.01,
        noise_std: float = 0.01,
        episode_length: int = 50,
        coupling_strength: float = 0.1,
        seed: int | None = None,
    ) -> None:
        self.frequencies = np.asarray(frequencies, dtype=np.float64)
        if self.frequencies.ndim != 1 or self.frequencies.size < 1:
            raise ValueError("frequencies must be a non-empty 1D sequence.")
        if amplitudes is None:
            amplitudes = np.ones_like(self.frequencies)
        self.amplitudes = np.asarray(amplitudes, dtype=np.float64)
        if self.amplitudes.shape != self.frequencies.shape:
            raise ValueError("amplitudes must match frequencies.")

        self.n_osc = int(self.frequencies.size)
        self.obs_dim = 2 * self.n_osc
        self.dt = float(dt)
        self.noise_std = float(noise_std)
        self.episode_length = int(episode_length)
        self.rng = np.random.default_rng(seed)
        self.coupling = self.rng.normal(
            loc=0.0,
            scale=float(coupling_strength),
            size=(self.n_osc,),
        )
        self.t = 0.0
        self.step_count = 0
        self.phases = np.zeros(self.n_osc, dtype=np.float64)
        self.x = np.zeros(self.n_osc, dtype=np.float64)
        self.dx = np.zeros(self.n_osc, dtype=np.float64)

    def reset(self) -> np.ndarray:
        self.t = 0.0
        self.step_count = 0
        self.phases = self.rng.uniform(0.0, 2.0 * math.pi, size=(self.n_osc,))
        self.x = self.amplitudes * np.sin(self.phases)
        self.dx = self._base_derivatives(action=0.0)
        return self._obs()

    def step(self, action: np.ndarray | list[float] | float) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        action_scalar = float(np.asarray(action, dtype=np.float64).reshape(-1)[0])
        action_scalar = float(np.clip(action_scalar, -1.0, 1.0))
        noise = self.rng.normal(0.0, self.noise_std, size=(self.n_osc,))
        self.dx = self._base_derivatives(action=action_scalar) + noise
        self.x = self.x + self.dt * self.dx
        self.t += self.dt
        self.step_count += 1
        obs = self._obs()
        reward = -float(np.mean(obs**2))
        done = self.step_count >= self.episode_length
        info = {"t": self.t, "step": self.step_count}
        return obs, reward, done, info

    def _base_derivatives(self, action: float) -> np.ndarray:
        omega = 2.0 * math.pi * self.frequencies
        phase = omega * self.t + self.phases
        return omega * self.amplitudes * np.cos(phase) + self.coupling * float(action)

    def _obs(self) -> np.ndarray:
        return np.concatenate([self.x, self.dx], axis=0).astype(np.float32)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    specs = complexity_specs()
    results = []

    print("Toy complexity sweep", flush=True)
    print(f"Device: cpu", flush=True)
    print(f"Train steps per level: {args.train_steps}", flush=True)
    print(f"Horizon: {args.horizon}", flush=True)
    print(f"Eval episodes: {args.eval_episodes}", flush=True)
    print(f"SMAD eta: {args.smad_eta}", flush=True)
    print("", flush=True)

    for spec in specs:
        result = run_level(spec, args)
        results.append(result)

    output = {
        "analysis": "toy_complexity_sweep",
        "hypothesis": (
            "dc_trend share increases as environment complexity grows while "
            "RSSM capacity stays fixed"
        ),
        "seed": int(args.seed),
        "train_steps": int(args.train_steps),
        "batch_size": int(args.batch_size),
        "horizon": int(args.horizon),
        "eval_episodes": int(args.eval_episodes),
        "trim_window_inclusive": [int(args.trim[0]), int(args.trim[1])],
        "smad_eta": float(args.smad_eta),
        "rssm": {
            "deter_dim": 64,
            "stoch_dim": 8,
            "hidden_dim": 64,
        },
        "levels": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(to_jsonable(output), indent=2) + "\n")
    print_comparison_table(results)
    print(f"\nSaved: {relative_path(args.output)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed-capacity toy complexity sweep.")
    parser.add_argument("--train_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--eval_episodes", "--n_episodes", dest="eval_episodes", type=int, default=10)
    parser.add_argument("--trim", type=int, nargs=2, default=(5, 45))
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--free_bits", type=float, default=0.0)
    parser.add_argument("--kl_scale", type=float, default=0.1)
    parser.add_argument("--smad_eta", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output = resolve_path(args.output)
    args.trim = (int(args.trim[0]), int(args.trim[1]))
    if args.train_steps <= 0:
        raise ValueError("--train_steps must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.eval_episodes <= 0:
        raise ValueError("--eval_episodes must be positive.")
    if not 0 <= args.trim[0] <= args.trim[1] < args.horizon:
        raise ValueError("--trim must be inside horizon.")
    if args.smad_eta < 0:
        raise ValueError("--smad_eta must be non-negative.")
    return args


def complexity_specs() -> list[ComplexitySpec]:
    return [
        ComplexitySpec(
            name="simple",
            frequencies=(0.5, 5.0),
            amplitudes=(1.0, 0.5),
            coupling_strength=0.10,
            seed_offset=0,
        ),
        ComplexitySpec(
            name="medium",
            frequencies=(0.3, 0.7, 2.0, 5.0, 12.0),
            amplitudes=(1.0, 0.9, 0.75, 0.55, 0.40),
            coupling_strength=0.10,
            seed_offset=100,
        ),
        ComplexitySpec(
            name="hard",
            frequencies=(0.1, 0.3, 0.5, 1.0, 2.0, 4.0, 7.0, 10.0, 15.0, 20.0),
            amplitudes=(1.0, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50, 0.42, 0.34, 0.28),
            coupling_strength=0.10,
            seed_offset=200,
        ),
    ]


def run_level(spec: ComplexitySpec, args: argparse.Namespace) -> dict[str, Any]:
    level_seed = args.seed + spec.seed_offset
    set_seed(level_seed)
    env = MultiOscillatorEnv(
        frequencies=spec.frequencies,
        amplitudes=spec.amplitudes,
        episode_length=args.horizon,
        coupling_strength=spec.coupling_strength,
        seed=level_seed,
    )
    rssm = MinimalRSSM(
        obs_dim=env.obs_dim,
        action_dim=1,
        deter_dim=64,
        stoch_dim=8,
        hidden_dim=64,
        lr=args.lr,
        device="cpu",
    )

    print(
        f"[{spec.name}] obs_dim={env.obs_dim} n_osc={env.n_osc} "
        f"frequencies={list(spec.frequencies)}",
        flush=True,
    )
    last_metrics = {}
    for step in range(1, args.train_steps + 1):
        obs, actions, rewards = sample_batch(env, args.batch_size, args.horizon)
        last_metrics = rssm.train_step(
            obs,
            actions,
            reward_seq=rewards,
            free_bits=args.free_bits,
            kl_scale=args.kl_scale,
            sample=True,
        )
        if step == 1 or step % 200 == 0 or step == args.train_steps:
            print(
                f"  [{spec.name}] step {step}/{args.train_steps} "
                f"loss={last_metrics['loss']:.4f} "
                f"recon={last_metrics['recon_loss']:.4f} "
                f"kl={last_metrics['kl_loss']:.4f}",
                flush=True,
            )

    rollouts = collect_rollouts(
        rssm,
        env,
        n_episodes=args.eval_episodes,
        horizon=args.horizon,
    )
    analysis = analyze_drift(
        rollouts["true_deter"],
        rollouts["imagined_deter"],
        trim=args.trim,
    )
    frequency = analysis["frequency"]
    rank = min(10, rollouts["true_deter"].shape[-1])
    U_drift = analysis["bases"]["U_drift"][:, :rank]
    damping = apply_damping(
        rssm,
        U_drift,
        eta=args.smad_eta,
        env=env,
        n_episodes=args.eval_episodes,
        horizon=args.horizon,
        baseline_rollouts=rollouts,
        trim=args.trim,
    )
    result = {
        "complexity": spec.name,
        "obs_dim": int(env.obs_dim),
        "n_osc": int(env.n_osc),
        "frequencies": list(spec.frequencies),
        "amplitudes": list(spec.amplitudes),
        "coupling": env.coupling.tolist(),
        "final_train_metrics": last_metrics,
        "frequency": {
            "band_mse": frequency["band_mse"],
            "band_share": frequency["band_share"],
            "J_slow": frequency["J_slow"],
            "J_total": frequency["J_total"],
            "pca_components": frequency["pca_components"],
            "pca_var_explained": frequency["pca_var_explained"],
        },
        "overlaps": analysis["overlaps"],
        "drift_pca": analysis["drift_pca"],
        "smad": {
            "eta": float(args.smad_eta),
            "rank": int(rank),
            "J_slow_reduction_pct": damping["J_slow_reduction_pct"],
            "J_total_reduction_pct": damping["J_total_reduction_pct"],
            "per_band_reduction_pct": damping["per_band_reduction_pct"],
            "damped_frequency": damping["damped_frequency"],
        },
    }
    print(
        f"  [{spec.name}] dc_trend share="
        f"{100.0 * frequency['band_share']['dc_trend']:.1f}% "
        f"J_total={frequency['J_total']:.6f} "
        f"SMAD J_slow reduction={damping['J_slow_reduction_pct']:.1f}% "
        f"J_total reduction={damping['J_total_reduction_pct']:.1f}%",
        flush=True,
    )
    print("", flush=True)
    return result


def sample_batch(
    env: MultiOscillatorEnv,
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


def print_comparison_table(results: list[dict[str, Any]]) -> None:
    print("Complexity sweep: dc_trend share and SMAD effect vs environment complexity", flush=True)
    print("", flush=True)
    print(
        "Complexity   obs_dim   dc_trend%   recon_loss   "
        "J_slow_reduction%   J_total_reduction%",
        flush=True,
    )
    for result in results:
        shares = result["frequency"]["band_share"]
        recon = result["final_train_metrics"]["recon_loss"]
        smad = result["smad"]
        print(
            f"{result['complexity']:<12}"
            f"{result['obs_dim']:>7}   "
            f"{100.0 * shares['dc_trend']:>8.1f}%   "
            f"{recon:>10.4f}   "
            f"{smad['J_slow_reduction_pct']:>17.1f}%   "
            f"{smad['J_total_reduction_pct']:>17.1f}%",
            flush=True,
        )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


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
