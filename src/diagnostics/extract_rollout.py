"""Extract a validation rollout from a randomly initialized NM512 DreamerV3.

This script is for Week 2 pipeline validation only. The agent is not trained,
so the imagined observations are expected to be poor; the useful signal here is
whether env stepping, RSSM imagination, decoding, and NPZ storage all connect.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import ruamel.yaml as yaml
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DREAMER_ROOT = REPO_ROOT / "external" / "dreamerv3-torch"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "rollouts" / "cheetah_random_seed0.npz"

sys.path.insert(0, str(DREAMER_ROOT))

from dreamer import Dreamer, make_env  # noqa: E402
import tools  # noqa: E402


# dreamer.py sets MUJOCO_GL=osmesa on import. On macOS, let the caller override
# it, otherwise default to glfw before dm_control is imported by the env wrapper.
if "DRIFTDETECT_MUJOCO_GL" in os.environ:
    os.environ["MUJOCO_GL"] = os.environ["DRIFTDETECT_MUJOCO_GL"]
elif sys.platform == "darwin":
    os.environ["MUJOCO_GL"] = "glfw"


class NullLogger:
    """Minimal logger surface required by NM512's Dreamer constructor."""

    step = 0

    def scalar(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def video(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def write(self, *_args: Any, **_kwargs: Any) -> None:
        pass


def recursive_update(base: dict[str, Any], update: Mapping[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            recursive_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)


def default_device() -> str:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def load_config(task: str, seed: int, device: str | None = None) -> SimpleNamespace:
    configs = yaml.safe_load((DREAMER_ROOT / "configs.yaml").read_text())
    values: dict[str, Any] = {}
    recursive_update(values, configs["defaults"])
    recursive_update(values, configs["dmc_proprio"])

    values.update(
        {
            "task": task,
            "seed": seed,
            "envs": 1,
            "parallel": False,
            "compile": False,
            "device": device or default_device(),
            "logdir": str(REPO_ROOT / "results" / "rollouts"),
            "video_pred_log": False,
        }
    )
    return SimpleNamespace(**values)


def batch_obs(obs: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {
        key: np.stack([np.asarray(value)])
        for key, value in obs.items()
        if not key.startswith("log_")
    }


def clone_latent(latent: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in latent.items()}


def flatten_obs(obs: Mapping[str, Any], keys: Sequence[str]) -> np.ndarray:
    parts = [np.asarray(obs[key], dtype=np.float32).reshape(-1) for key in keys]
    return np.concatenate(parts, axis=0)


def flatten_decoded_obs(
    decoded: Mapping[str, Any], keys: Sequence[str]
) -> np.ndarray:
    parts = []
    for key in keys:
        value = decoded[key].mode().detach().cpu().numpy()
        parts.append(value.reshape(value.shape[0], -1)[0].astype(np.float32))
    return np.concatenate(parts, axis=0)


def action_for_env(
    policy_output: Mapping[str, torch.Tensor],
) -> tuple[dict[str, np.ndarray], torch.Tensor]:
    action = policy_output["action"].detach()
    action_np = action[0].cpu().numpy().astype(np.float32)
    return {"action": action_np}, action


def resolve_output_path(output_path: str | Path | None) -> Path:
    path = Path(output_path) if output_path is not None else DEFAULT_OUTPUT
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def extract_rollout(
    task: str = "dmc_cheetah_run",
    seed: int = 0,
    imagination_start: int = 200,
    horizon: int = 200,
    total_steps: int = 1000,
    output_path: str | Path | None = None,
    device: str | None = None,
) -> Path:
    if imagination_start < 0:
        raise ValueError("imagination_start must be non-negative.")
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if total_steps < imagination_start + horizon:
        raise ValueError("total_steps must cover the full imagination window.")

    config = load_config(task=task, seed=seed, device=device)
    tools.set_seed_everywhere(seed)

    env = make_env(config, "eval", 0)
    config.num_actions = (
        env.action_space.n
        if hasattr(env.action_space, "n")
        else env.action_space.shape[0]
    )

    # Random initialization by construction: no checkpoint is loaded here.
    agent = Dreamer(
        env.observation_space,
        env.action_space,
        config,
        logger=NullLogger(),
        dataset=None,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    agent.eval()

    decoder = agent._wm.heads["decoder"]
    obs_keys = tuple(getattr(decoder, "mlp_shapes", {}).keys())
    if not obs_keys:
        raise RuntimeError(
            "No vector observation decoder keys found. This script expects "
            "the dmc_proprio config so true_obs and imagined_obs are flat arrays."
        )

    true_obs_list: list[np.ndarray] = []
    imagined_obs_list: list[np.ndarray] = []
    actions_list: list[np.ndarray] = []
    rewards_list: list[float] = []

    obs = env.reset()
    done = np.ones(1, dtype=bool)
    agent_state = None
    imag_latent: dict[str, torch.Tensor] | None = None

    try:
        for step in range(total_steps):
            with torch.no_grad():
                policy_output, agent_state = agent(
                    batch_obs(obs), done, agent_state, training=False
                )
            env_action, action_tensor = action_for_env(policy_output)

            in_imagination_window = (
                imagination_start <= step < imagination_start + horizon
            )
            if step == imagination_start:
                latent, _prev_action = agent_state
                imag_latent = clone_latent(latent)

            if in_imagination_window:
                if imag_latent is None:
                    raise RuntimeError("Imagination latent was not initialized.")

                # Open-loop branch: advance the RSSM with the real action, but
                # do not feed future environment observations back into it.
                with torch.no_grad():
                    imag_latent = agent._wm.dynamics.img_step(
                        imag_latent, action_tensor, sample=False
                    )
                    feat = agent._wm.dynamics.get_feat(imag_latent)
                    imagined_obs = flatten_decoded_obs(decoder(feat), obs_keys)

            next_obs, reward, done_bool, _info = env.step(env_action)

            if in_imagination_window:
                # These are the true next observations caused by the same action
                # used for the corresponding open-loop model prediction.
                true_obs_list.append(flatten_obs(next_obs, obs_keys))
                imagined_obs_list.append(imagined_obs)
                actions_list.append(env_action["action"])
                rewards_list.append(float(reward))

            obs = next_obs
            done = np.array([done_bool], dtype=bool)
            if done_bool and step + 1 < total_steps:
                obs = env.reset()
                done = np.ones(1, dtype=bool)
                agent_state = None
    finally:
        try:
            env.close()
        except Exception:
            pass

    metadata = {
        "task": task,
        "imagination_start": imagination_start,
        "horizon": horizon,
        "seed": seed,
        "agent_type": "dreamerv3_random_init",
        "obs_keys": obs_keys,
        "obs_alignment": (
            "true and imagined arrays contain next observations for actions "
            f"at steps {imagination_start}-{imagination_start + horizon - 1}"
        ),
    }

    output = resolve_output_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        true_obs=np.asarray(true_obs_list, dtype=np.float32),
        imagined_obs=np.asarray(imagined_obs_list, dtype=np.float32),
        actions=np.asarray(actions_list, dtype=np.float32),
        rewards=np.asarray(rewards_list, dtype=np.float32),
        metadata=np.array(metadata, dtype=object),
    )

    size_mb = output.stat().st_size / (1024 * 1024)
    print("Rollout extracted:")
    print(f"  Task: {task}")
    print(f"  Total steps: {total_steps}")
    print(
        f"  Imagination: steps {imagination_start}-"
        f"{imagination_start + horizon} (H={horizon})"
    )
    print(f"  Saved to: {display_path(output)}")
    print(f"  File size: {size_mb:.1f} MB")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="dmc_cheetah_run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--imagination_start", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--total_steps", type=int, default=1000)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT.relative_to(REPO_ROOT)))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    extract_rollout(
        task=args.task,
        seed=args.seed,
        imagination_start=args.imagination_start,
        horizon=args.horizon,
        total_steps=args.total_steps,
        output_path=args.output,
        device=args.device,
    )
