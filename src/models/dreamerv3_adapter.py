"""DreamerV3 adapter for architecture-agnostic rollout extraction."""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import ruamel.yaml as yaml
import torch

from src.models.adapter import WorldModelAdapter


REPO_ROOT = Path(__file__).resolve().parents[2]
DREAMER_ROOT = REPO_ROOT / "external" / "dreamerv3-torch"

# Proprio rollouts do not need pixels. Disabling MuJoCo rendering avoids GLFW
# hangs in non-interactive macOS shells while preserving vector observations.
if "DRIFTDETECT_MUJOCO_GL" not in os.environ and sys.platform == "darwin":
    os.environ["DRIFTDETECT_MUJOCO_GL"] = "disable"
if "DRIFTDETECT_MUJOCO_GL" in os.environ:
    os.environ["MUJOCO_GL"] = os.environ["DRIFTDETECT_MUJOCO_GL"]

sys.path.insert(0, str(DREAMER_ROOT))

from dreamer import Dreamer, make_env  # noqa: E402
import tools  # noqa: E402


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


def normalize_checkpoint_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {key.replace("._orig_mod.", "."): value for key, value in state_dict.items()}


def clone_latent(latent: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in latent.items()}


def install_dummy_image_renderer(env: Any, size: Sequence[int]) -> None:
    image_shape = tuple(size) + (3,)

    def render_dummy(*_args: Any, **_kwargs: Any) -> np.ndarray:
        return np.zeros(image_shape, dtype=np.uint8)

    current = env
    while current is not None:
        if hasattr(current, "render"):
            current.render = render_dummy
        current = getattr(current, "env", None)


def batch_obs(obs: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {
        key: np.stack([np.asarray(value)])
        for key, value in obs.items()
        if not key.startswith("log_")
    }


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


class DreamerV3Adapter(WorldModelAdapter):
    """Adapter around NM512's DreamerV3 RSSM implementation."""

    def __init__(
        self,
        task: str = "dmc_cheetah_run",
        device: str | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        self.task = task
        self.device = device
        self.config_path = Path(config_path) if config_path else DREAMER_ROOT / "configs.yaml"
        self.checkpoint_path: Path | None = None

        self.config: SimpleNamespace | None = None
        self.env: Any | None = None
        self.agent: Dreamer | None = None
        self.obs_keys: tuple[str, ...] = ()
        self._cached_imag_latent: dict[str, torch.Tensor] | None = None

    def load_checkpoint(self, path: str) -> None:
        self.checkpoint_path = self._resolve_path(path)
        self._build(seed=0)

    def reset(self, seed: int = 0) -> np.ndarray:
        self._build(seed=seed)
        assert self.env is not None
        obs = self.env.reset()
        return flatten_obs(obs, self.obs_keys)

    def encode(self, obs: np.ndarray) -> dict:
        self._ensure_ready()
        assert self.agent is not None
        obs_dict = self._flat_obs_to_mapping(obs)
        with torch.no_grad():
            processed = self.agent._wm.preprocess(batch_obs(obs_dict))
            embed = self.agent._wm.encoder(processed)
            latent, _ = self.agent._wm.dynamics.obs_step(
                None, None, embed, processed["is_first"]
            )
        return clone_latent(latent)

    def imagine(
        self,
        init_obs: np.ndarray,
        actions: np.ndarray,
        horizon: int,
    ) -> np.ndarray:
        del init_obs
        self._ensure_ready()
        if self._cached_imag_latent is None:
            raise RuntimeError(
                "No cached DreamerV3 latent found. Call collect_true_rollout() "
                "before imagine(), or use extract_rollout()."
            )
        if len(actions) < horizon:
            raise ValueError("actions must contain at least horizon steps.")

        assert self.agent is not None
        decoder = self.agent._wm.heads["decoder"]
        imag_latent = clone_latent(self._cached_imag_latent)
        imagined_obs = []

        for step in range(horizon):
            action_tensor = torch.as_tensor(
                actions[step : step + 1],
                device=self.agent._config.device,
                dtype=torch.float32,
            )
            with torch.no_grad():
                imag_latent = self.agent._wm.dynamics.img_step(
                    imag_latent, action_tensor, sample=False
                )
                feat = self.agent._wm.dynamics.get_feat(imag_latent)
                imagined_obs.append(flatten_decoded_obs(decoder(feat), self.obs_keys))

        return np.asarray(imagined_obs, dtype=np.float32)

    def collect_true_rollout(
        self,
        seed: int,
        total_steps: int,
        imagination_start: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if imagination_start < 0:
            raise ValueError("imagination_start must be non-negative.")
        if total_steps <= imagination_start:
            raise ValueError("total_steps must exceed imagination_start.")

        self._build(seed=seed)
        assert self.env is not None
        assert self.agent is not None

        true_obs_list: list[np.ndarray] = []
        actions_list: list[np.ndarray] = []
        rewards_list: list[float] = []

        obs = self.env.reset()
        done = np.ones(1, dtype=bool)
        agent_state = None
        self._cached_imag_latent = None

        for step in range(total_steps):
            with torch.no_grad():
                policy_output, agent_state = self.agent(
                    batch_obs(obs), done, agent_state, training=False
                )
            env_action, _action_tensor = action_for_env(policy_output)

            if step == imagination_start:
                latent, _prev_action = agent_state
                self._cached_imag_latent = clone_latent(latent)

            next_obs, reward, done_bool, _info = self.env.step(env_action)

            true_obs_list.append(flatten_obs(next_obs, self.obs_keys))
            actions_list.append(env_action["action"])
            rewards_list.append(float(reward))

            obs = next_obs
            done = np.array([done_bool], dtype=bool)
            if done_bool and step + 1 < total_steps:
                obs = self.env.reset()
                done = np.ones(1, dtype=bool)
                agent_state = None

        if self._cached_imag_latent is None:
            raise RuntimeError("Imagination latent was not initialized.")

        return (
            np.asarray(true_obs_list, dtype=np.float32),
            np.asarray(actions_list, dtype=np.float32),
            np.asarray(rewards_list, dtype=np.float32),
        )

    def _build(self, seed: int) -> None:
        self._close_env()
        self.config = self._load_config(seed)
        tools.set_seed_everywhere(seed)

        self.env = make_env(self.config, "eval", 0)
        install_dummy_image_renderer(self.env, self.config.size)
        self.config.num_actions = (
            self.env.action_space.n
            if hasattr(self.env.action_space, "n")
            else self.env.action_space.shape[0]
        )

        self.agent = Dreamer(
            self.env.observation_space,
            self.env.action_space,
            self.config,
            logger=NullLogger(),
            dataset=None,
        ).to(self.config.device)
        if self.checkpoint_path is not None:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.config.device)
            self.agent.load_state_dict(
                normalize_checkpoint_state_dict(checkpoint["agent_state_dict"])
            )
        self.agent.requires_grad_(requires_grad=False)
        self.agent.eval()

        decoder = self.agent._wm.heads["decoder"]
        self.obs_keys = tuple(getattr(decoder, "mlp_shapes", {}).keys())
        if not self.obs_keys:
            raise RuntimeError(
                "No vector observation decoder keys found. DreamerV3Adapter "
                "expects a proprioceptive DreamerV3 configuration."
            )

    def _load_config(self, seed: int) -> SimpleNamespace:
        configs = yaml.safe_load(self.config_path.read_text())
        values: dict[str, Any] = {}
        recursive_update(values, configs["defaults"])
        recursive_update(values, configs["dmc_proprio"])
        values.update(
            {
                "task": self.task,
                "seed": seed,
                "envs": 1,
                "parallel": False,
                "compile": False,
                "device": self.device or default_device(),
                "logdir": str(REPO_ROOT / "results" / "rollouts"),
                "video_pred_log": False,
            }
        )
        return SimpleNamespace(**values)

    def _flat_obs_to_mapping(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        self._ensure_ready()
        assert self.agent is not None
        decoder = self.agent._wm.heads["decoder"]
        shapes = getattr(decoder, "mlp_shapes", {})
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)

        offset = 0
        obs_dict: dict[str, np.ndarray] = {}
        for key in self.obs_keys:
            shape = tuple(shapes[key])
            size = int(np.prod(shape))
            obs_dict[key] = obs[offset : offset + size].reshape(shape)
            offset += size

        if offset != obs.size:
            raise ValueError(
                f"Expected flat obs with {offset} values, got {obs.size}."
            )

        assert self.config is not None
        obs_dict["image"] = np.zeros(tuple(self.config.size) + (3,), dtype=np.uint8)
        obs_dict["is_terminal"] = False
        obs_dict["is_first"] = True
        return obs_dict

    def _ensure_ready(self) -> None:
        if self.agent is None or self.env is None or self.config is None:
            self._build(seed=0)

    def _resolve_path(self, path: str | Path) -> Path:
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = REPO_ROOT / resolved
        return resolved

    def _close_env(self) -> None:
        if self.env is None:
            return
        try:
            self.env.close()
        except Exception:
            pass
        self.env = None
