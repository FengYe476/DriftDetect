#!/usr/bin/env python3
"""Extract R2-Dreamer posterior/imagination rollouts and compare drift.

The script evaluates two completed R2-Dreamer checkpoints, typically baseline
and SHARP, by collecting real policy rollouts, encoding posterior latents, and
running a same-action open-loop RSSM imagination from a fixed start step.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
R2_ROOT = PROJECT_ROOT / "external" / "r2dreamer"


def add_import_paths() -> None:
    for path in (R2_ROOT, PROJECT_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def patch_r2_dmc_size() -> None:
    """Coerce Hydra list sizes to tuples before R2's DMC env stores them."""

    try:
        import envs.dmc as dmc
    except Exception:
        return

    if getattr(dmc.DeepMindControl, "_driftdetect_size_patch", False):
        return

    original_init = dmc.DeepMindControl.__init__

    def patched_init(self, name, action_repeat=1, size=(64, 64), camera=None, seed=0):
        return original_init(
            self,
            name,
            action_repeat=action_repeat,
            size=tuple(size),
            camera=camera,
            seed=seed,
        )

    dmc.DeepMindControl.__init__ = patched_init
    dmc.DeepMindControl._driftdetect_size_patch = True


def load_r2_config(args: argparse.Namespace, *, seed: int = 0):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(version_base=None, config_dir=str(R2_ROOT / "configs")):
        config = compose(
            config_name="configs",
            overrides=[f"env={args.env_config}", f"model={args.model_config}"],
        )

    config.seed = int(seed)
    config.device = str(args.device)
    config.deterministic_run = bool(args.deterministic)

    config.env.task = str(args.task)
    config.env.seed = int(seed)
    config.env.env_num = 1
    config.env.eval_episode_num = 1
    config.env.action_repeat = int(args.action_repeat)
    config.env.time_limit = int(args.time_limit)

    config.model.rep_loss = "r2dreamer"
    config.model.compile = False
    config.model.device = str(args.device)
    config.model.rssm.device = str(args.device)

    OmegaConf.resolve(config)
    return config


def resolve_path(path: str | pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def make_single_env(config):
    from envs import make_env

    return make_env(config.env, 0)


def build_agent(args: argparse.Namespace, ckpt_path: pathlib.Path):
    import torch
    from dreamer import Dreamer

    config = load_r2_config(args, seed=0)
    env = make_single_env(config)
    try:
        agent = Dreamer(config.model, env.observation_space, env.action_space).to(args.device)
    finally:
        env.close()

    checkpoint = torch.load(ckpt_path, map_location=args.device)
    state_dict = checkpoint.get("agent_state_dict", checkpoint)
    try:
        agent.load_state_dict(state_dict)
    except RuntimeError as exc:
        print(f"Strict checkpoint load failed for {ckpt_path}: {exc}", flush=True)
        missing, unexpected = agent.load_state_dict(state_dict, strict=False)
        print(f"Loaded with strict=False. Missing={missing}, unexpected={unexpected}", flush=True)

    # Ensure policy inference uses frozen copies that match the loaded weights.
    agent.clone_and_freeze()
    agent.eval()
    return agent


def record_from_obs(obs: dict[str, Any], action: np.ndarray, reward: float) -> dict[str, np.ndarray]:
    record = {k: np.asarray(v) for k, v in obs.items()}
    record["reward"] = np.asarray(reward, dtype=np.float32)
    record["action"] = np.asarray(action, dtype=np.float32)
    return record


def batch_obs_for_policy(obs: dict[str, Any], reward: float, device: torch.device):
    import torch
    from tensordict import TensorDict

    values = {k: np.asarray(v) for k, v in obs.items()}
    values["reward"] = np.asarray(reward, dtype=np.float32)

    tensors = {}
    for key, value in values.items():
        if value.ndim == 0:
            value = value.reshape(1, 1)
        else:
            value = value[None]
        tensors[key] = torch.as_tensor(value, device=device)
    return TensorDict(tensors, batch_size=(1,), device=device)


def records_to_sequence(records: list[dict[str, np.ndarray]], device: torch.device):
    import torch
    from tensordict import TensorDict

    keys = records[0].keys()
    tensors = {}
    for key in keys:
        values = []
        for record in records:
            value = np.asarray(record[key])
            if value.ndim == 0:
                value = value.reshape(1)
            values.append(value)
        stacked = np.stack(values, axis=0)
        stacked = stacked[None]  # (B=1, T, ...)
        tensors[key] = torch.as_tensor(stacked, device=device)
    return TensorDict(tensors, batch_size=(1, len(records)), device=device)


def collect_episode_records(
    *,
    agent,
    env,
    seed: int,
    required_steps: int,
    device: torch.device,
) -> list[dict[str, np.ndarray]]:
    import torch
    import tools

    with torch.no_grad():
        tools.set_seed_everywhere(seed)
        obs = env.reset()
        action = np.zeros(agent.act_dim, dtype=np.float32)
        records = [record_from_obs(obs, action, reward=0.0)]

        state = agent.get_initial_state(1)
        policy_obs = batch_obs_for_policy(obs, reward=0.0, device=device)
        action_tensor, state = agent.act(policy_obs, state, eval=True)

        for _ in range(required_steps):
            action_np = action_tensor[0].detach().cpu().numpy().astype(np.float32)
            next_obs, reward, done, _info = env.step(action_np)
            records.append(record_from_obs(next_obs, action_np, reward=float(reward)))

            if len(records) >= required_steps + 1:
                break
            if done:
                raise RuntimeError(
                    f"Episode ended before collecting {required_steps} steps "
                    f"(got {len(records) - 1}). Increase --time_limit or use another seed."
                )

            policy_obs = batch_obs_for_policy(next_obs, reward=float(reward), device=device)
            action_tensor, state = agent.act(policy_obs, state, eval=True)

        return records


def encode_and_imagine(
    *,
    agent,
    records: list[dict[str, np.ndarray]],
    imagination_start: int,
    horizon: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    import torch

    with torch.no_grad():
        data = records_to_sequence(records, device)
        p_data = agent.preprocess(data)

        embed = agent.encoder(p_data)
        initial = agent.rssm.initial(1)
        post_stoch, post_deter, _post_logit = agent.rssm.observe(
            embed,
            p_data["action"],
            initial,
            p_data["is_first"],
        )

        start = int(imagination_start)
        end = start + int(horizon)
        if end >= post_deter.shape[1]:
            raise ValueError(
                f"Need posterior through index {end}; collected sequence has "
                f"{post_deter.shape[1]} states."
            )

        stoch = post_stoch[:, start]
        deter = post_deter[:, start]
        imag_stochs = []
        imag_deters = []
        future_actions = p_data["action"][:, start + 1 : end + 1]
        for t in range(horizon):
            stoch, deter = agent.rssm.img_step(stoch, deter, future_actions[:, t])
            imag_stochs.append(stoch)
            imag_deters.append(deter)

        imagined_stoch = torch.stack(imag_stochs, dim=1)
        imagined_deter = torch.stack(imag_deters, dim=1)
        true_stoch = post_stoch[:, start + 1 : end + 1]
        true_deter = post_deter[:, start + 1 : end + 1]

        return {
            "true_deter": true_deter[0].detach().cpu().numpy().astype(np.float32),
            "imagined_deter": imagined_deter[0].detach().cpu().numpy().astype(np.float32),
            "true_stoch": true_stoch[0].detach().cpu().numpy().astype(np.float32),
            "imagined_stoch": imagined_stoch[0].detach().cpu().numpy().astype(np.float32),
            "actions": future_actions[0].detach().cpu().numpy().astype(np.float32),
            "posterior_context_deter": post_deter[0, : start + 1].detach().cpu().numpy().astype(np.float32),
        }


def compute_rollout_metrics(
    true_deter: np.ndarray,
    imagined_deter: np.ndarray,
    *,
    trim_start: int,
    trim_end: int,
) -> dict[str, float]:
    mse_per_step = np.mean((imagined_deter - true_deter) ** 2, axis=-1)

    def band(start: int, end: int) -> float:
        return float(np.mean(mse_per_step[start:end]))

    if trim_start < 0 or trim_end > len(mse_per_step) or trim_start >= trim_end:
        raise ValueError(
            f"Invalid trim window [{trim_start}, {trim_end}) for horizon {len(mse_per_step)}"
        )
    band_width = (trim_end - trim_start) // 3
    early_start, early_end = trim_start, trim_start + band_width
    mid_start, mid_end = early_end, early_end + band_width
    late_start, late_end = mid_end, trim_end

    return {
        "J_total": band(trim_start, trim_end),
        "early": band(early_start, early_end),
        "mid": band(mid_start, mid_end),
        "late": band(late_start, late_end),
        "step0_1": float(mse_per_step[0]),
        "full_horizon": float(np.mean(mse_per_step)),
    }


def summarize_metrics(metrics: list[dict[str, float]]) -> dict[str, Any]:
    keys = metrics[0].keys()
    summary: dict[str, Any] = {"n": len(metrics)}
    for key in keys:
        values = np.asarray([m[key] for m in metrics], dtype=np.float64)
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=0)),
        }
    return summary


def extract_checkpoint(
    *,
    label: str,
    ckpt_path: pathlib.Path,
    outdir: pathlib.Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import torch

    device = torch.device(args.device)
    agent = build_agent(args, ckpt_path)
    rollout_dir = outdir / f"rollouts_{label}"
    rollout_dir.mkdir(parents=True, exist_ok=True)

    metrics = []
    required_steps = int(args.imagination_start) + int(args.horizon)

    for seed in range(args.num_seeds):
        print(f"[{label}] extracting seed {seed}", flush=True)
        config = load_r2_config(args, seed=seed)
        env = make_single_env(config)
        try:
            records = collect_episode_records(
                agent=agent,
                env=env,
                seed=seed,
                required_steps=required_steps,
                device=device,
            )
        finally:
            env.close()

        rollout = encode_and_imagine(
            agent=agent,
            records=records,
            imagination_start=args.imagination_start,
            horizon=args.horizon,
            device=device,
        )
        rollout_metrics = compute_rollout_metrics(
            rollout["true_deter"],
            rollout["imagined_deter"],
            trim_start=args.trim_start,
            trim_end=args.trim_end,
        )
        metrics.append(rollout_metrics)

        np.savez_compressed(
            rollout_dir / f"seed_{seed:02d}.npz",
            seed=np.asarray(seed, dtype=np.int32),
            task=np.asarray(args.task),
            checkpoint=np.asarray(str(ckpt_path)),
            imagination_start=np.asarray(args.imagination_start, dtype=np.int32),
            horizon=np.asarray(args.horizon, dtype=np.int32),
            **rollout,
            mse_per_step=np.mean(
                (rollout["imagined_deter"] - rollout["true_deter"]) ** 2,
                axis=-1,
            ).astype(np.float32),
            **{f"metric_{k}": np.asarray(v, dtype=np.float32) for k, v in rollout_metrics.items()},
        )

    return {
        "checkpoint": str(ckpt_path),
        "rollout_dir": str(rollout_dir),
        "summary": summarize_metrics(metrics),
        "per_seed": metrics,
    }


def pct_change(new: float, old: float) -> float:
    if old == 0:
        return float("nan")
    return 100.0 * (new - old) / old


def build_comparison(baseline: dict[str, Any], sharp: dict[str, Any]) -> dict[str, Any]:
    comparison = {}
    for key in ("J_total", "early", "mid", "late", "step0_1", "full_horizon"):
        base = float(baseline["summary"][key]["mean"])
        sh = float(sharp["summary"][key]["mean"])
        comparison[key] = {
            "baseline": base,
            "sharp": sh,
            "change_pct": pct_change(sh, base),
        }
    return comparison


def print_table(comparison: dict[str, Any], *, trim_start: int, trim_end: int) -> None:
    band_width = (trim_end - trim_start) // 3
    early = (trim_start, trim_start + band_width)
    mid = (early[1], early[1] + band_width)
    late = (mid[1], trim_end)
    rows = [
        (f"J_total ({trim_start}:{trim_end})", "J_total"),
        (f"Early ({early[0]}:{early[1]})", "early"),
        (f"Mid ({mid[0]}:{mid[1]})", "mid"),
        (f"Late ({late[0]}:{late[1]})", "late"),
        ("Step 0->1", "step0_1"),
        ("Full horizon", "full_horizon"),
    ]
    print("\nR2-Dreamer Drift Comparison")
    print("=" * 76)
    print(f"{'Metric':<22} {'Baseline':>14} {'SHARP':>14} {'Change':>14}")
    print("-" * 76)
    for label, key in rows:
        item = comparison[key]
        print(
            f"{label:<22} "
            f"{item['baseline']:>14.6f} "
            f"{item['sharp']:>14.6f} "
            f"{item['change_pct']:>13.1f}%"
        )
    print("=" * 76)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_ckpt", default="results/r2_baseline_cheetah_v2/latest.pt")
    parser.add_argument("--sharp_ckpt", default="results/r2_sharp_cheetah/latest.pt")
    parser.add_argument("--task", default="dmc_cheetah_run")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--outdir", default="results/r2_cross_model")
    parser.add_argument("--num_seeds", type=int, default=20)
    parser.add_argument("--imagination_start", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--trim_start", type=int, default=25)
    parser.add_argument("--trim_end", type=int, default=175)
    parser.add_argument("--action_repeat", type=int, default=2)
    parser.add_argument("--time_limit", type=int, default=1000)
    parser.add_argument("--env_config", default="dmc_proprio")
    parser.add_argument("--model_config", default="size12M")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not R2_ROOT.exists():
        raise FileNotFoundError(f"Missing R2-Dreamer repo at {R2_ROOT}")

    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    add_import_paths()
    patch_r2_dmc_size()

    baseline_ckpt = resolve_path(args.baseline_ckpt)
    sharp_ckpt = resolve_path(args.sharp_ckpt)
    if not baseline_ckpt.exists():
        raise FileNotFoundError(f"Missing baseline checkpoint: {baseline_ckpt}")
    if not sharp_ckpt.exists():
        raise FileNotFoundError(f"Missing SHARP checkpoint: {sharp_ckpt}")

    outdir = resolve_path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    baseline = extract_checkpoint(
        label="baseline",
        ckpt_path=baseline_ckpt,
        outdir=outdir,
        args=args,
    )
    sharp = extract_checkpoint(
        label="sharp",
        ckpt_path=sharp_ckpt,
        outdir=outdir,
        args=args,
    )
    comparison = build_comparison(baseline, sharp)
    result = {
        "task": args.task,
        "num_seeds": args.num_seeds,
        "imagination_start": args.imagination_start,
        "horizon": args.horizon,
        "trim": [args.trim_start, args.trim_end],
        "baseline": baseline,
        "sharp": sharp,
        "comparison": comparison,
    }

    output_path = outdir / "r2_drift_comparison.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print_table(comparison, trim_start=args.trim_start, trim_end=args.trim_end)
    print(f"\nSaved summary: {output_path}")


if __name__ == "__main__":
    main()
