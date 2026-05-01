"""DreamerV4 adapter for architecture-agnostic rollout extraction.

This adapter is CUDA-only. It is intended to be written and imported locally,
but checkpoint loading and inference are expected to run on RunPod.
"""

from __future__ import annotations

import gc
import importlib.util
import json
import logging
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

try:
    from src.models.adapter import WorldModelAdapter
except Exception:  # pragma: no cover - avoids importing V3 deps through src.models.__init__.
    _adapter_path = Path(__file__).resolve().with_name("adapter.py")
    _adapter_spec = importlib.util.spec_from_file_location(
        "_driftdetect_world_model_adapter",
        _adapter_path,
    )
    if _adapter_spec is None or _adapter_spec.loader is None:
        raise
    _adapter_module = importlib.util.module_from_spec(_adapter_spec)
    _adapter_spec.loader.exec_module(_adapter_module)
    WorldModelAdapter = _adapter_module.WorldModelAdapter


REPO_ROOT = Path(__file__).resolve().parents[2]
DREAMER4_ROOT = REPO_ROOT / "external" / "dreamer4"
DREAMER4_PKG_ROOT = DREAMER4_ROOT / "dreamer4"

# Upstream Dreamer4 modules use top-level imports such as `from model import ...`.
# Keep both the repo root requested by the design and the package directory on
# sys.path so those imports resolve in the same way they do from upstream scripts.
for _path in (DREAMER4_PKG_ROOT, DREAMER4_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


LOGGER = logging.getLogger(__name__)


def _import_v4_modules() -> SimpleNamespace:
    """Import upstream Dreamer4 modules lazily so local no-CUDA collection still works."""

    try:
        import interactive as interactive_module
        from model import pack_bottleneck_to_spatial, temporal_patchify
        from train_dynamics import (
            decode_packed_to_frames,
            sample_autoregressive_packed_sequence,
        )
        from wm_dataset import WMDataset
    except Exception as exc:  # pragma: no cover - exercised only on incomplete envs.
        raise ImportError(
            "Could not import Dreamer4 modules. Run inside the RunPod "
            "env-dreamerv4 environment with external/dreamer4 present."
        ) from exc

    return SimpleNamespace(
        interactive=interactive_module,
        pack_bottleneck_to_spatial=pack_bottleneck_to_spatial,
        temporal_patchify=temporal_patchify,
        decode_packed_to_frames=decode_packed_to_frames,
        sample_autoregressive_packed_sequence=sample_autoregressive_packed_sequence,
        WMDataset=WMDataset,
    )


def _torch_load_cpu(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if torch.is_tensor(value):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    return str(value)


def _sample_tensor_shapes(value: Any, *, prefix: str = "", limit: int = 16) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    if torch.is_tensor(value):
        shapes[prefix or "tensor"] = list(value.shape)
        return shapes
    if not isinstance(value, dict):
        return shapes

    for key, child in value.items():
        if len(shapes) >= limit:
            break
        child_prefix = f"{prefix}.{key}" if prefix else str(key)
        if torch.is_tensor(child):
            shapes[child_prefix] = list(child.shape)
        elif isinstance(child, dict):
            shapes.update(_sample_tensor_shapes(child, prefix=child_prefix, limit=limit - len(shapes)))
    return shapes


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved


def _find_checkpoint_file(root: Path, names: tuple[str, ...]) -> Path:
    if root.is_file():
        return root
    for name in names:
        candidate = root / name
        if candidate.exists():
            return candidate
    expected = ", ".join(names)
    raise FileNotFoundError(f"Expected one of {expected} under {root}.")


def save_rollout_npz(path: str | Path, rollout: dict[str, Any]) -> None:
    """Save a V4 rollout dict with object metadata preserved."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    for key, value in rollout.items():
        if key == "metadata":
            payload[key] = np.array(value, dtype=object)
        else:
            payload[key] = value
    np.savez(output, **payload)


class DreamerV4Adapter(WorldModelAdapter):
    """Adapter around Nicklas Hansen's Dreamer4 PyTorch implementation."""

    def __init__(
        self,
        task: str = "cheetah-run",
        device: str = "cuda",
        *,
        packing_factor: int = 2,
        schedule: str = "shortcut",
        schedule_d: float | None = 0.25,
        use_amp: bool = True,
        dataset_path: str | Path | None = None,
    ) -> None:
        self.task = task
        self.device = torch.device(device)
        self.packing_factor = int(packing_factor)
        self.schedule = schedule
        self.schedule_d = schedule_d
        self.use_amp = bool(use_amp)
        self.dataset_path = None if dataset_path is None else _resolve_path(dataset_path)

        if self.device.type != "cuda":
            raise RuntimeError("DreamerV4Adapter is CUDA-only; use device='cuda'.")
        if not torch.cuda.is_available():
            raise RuntimeError("DreamerV4Adapter requires CUDA, but torch.cuda is unavailable.")

        self.checkpoint_dir: Path | None = None
        self.tokenizer_path: Path | None = None
        self.dynamics_path: Path | None = None
        self.checkpoint_info: dict[str, Any] = {}

        self.tokenizer: Any | None = None
        self.encoder: Any | None = None
        self.decoder: Any | None = None
        self.dynamics: Any | None = None
        self.tau_schedule: dict[str, Any] | None = None

        self.k_max: int | None = None
        self.n_latents: int | None = None
        self.d_bottleneck: int | None = None
        self.n_spatial: int | None = None
        self.d_spatial: int | None = None
        self.H: int | None = None
        self.W: int | None = None
        self.C: int | None = None
        self.patch: int | None = None

        self._dataset_cache: dict[tuple[str, str, int], Any] = {}

    def load_checkpoint(
        self,
        path: str,
        *,
        tokenizer_path: str | Path | None = None,
        dynamics_path: str | Path | None = None,
    ) -> None:
        modules = _import_v4_modules()

        root = _resolve_path(path)
        tok_path = (
            _resolve_path(tokenizer_path)
            if tokenizer_path is not None
            else _find_checkpoint_file(root, ("tokenizer.pt", "tokenizer_ckpt.pt"))
        )
        dyn_path = (
            _resolve_path(dynamics_path)
            if dynamics_path is not None
            else _find_checkpoint_file(root, ("dynamics.pt", "dynamics_ckpt.pt"))
        )

        tok_args = self._log_checkpoint_summary(tok_path, "tokenizer")
        dyn_args = self._log_checkpoint_summary(dyn_path, "dynamics")

        LOGGER.info("Loading DreamerV4 tokenizer from %s", tok_path)
        self.tokenizer, tok_info = modules.interactive.load_tokenizer_from_ckpt(
            str(tok_path),
            self.device,
        )
        self.encoder = self.tokenizer.encoder
        self.decoder = self.tokenizer.decoder

        self.H = int(tok_info["H"])
        self.W = int(tok_info["W"])
        self.C = int(tok_info["C"])
        self.patch = int(tok_info["patch"])
        self.n_latents = int(tok_info["n_latents"])
        self.d_bottleneck = int(tok_info["d_bottleneck"])

        LOGGER.info(
            "Loading DreamerV4 dynamics from %s with packing_factor=%d",
            dyn_path,
            self.packing_factor,
        )
        self.dynamics, dyn_info = modules.interactive.load_dynamics_from_ckpt(
            str(dyn_path),
            device=self.device,
            d_bottleneck=self.d_bottleneck,
            n_latents=self.n_latents,
            packing_factor=self.packing_factor,
        )

        self.k_max = int(dyn_info["k_max"])
        self.n_spatial = int(dyn_info["n_spatial"])
        self.d_spatial = int(dyn_info["d_spatial"])
        self.tau_schedule = modules.interactive.make_tau_schedule(
            k_max=self.k_max,
            schedule=self.schedule,
            d=(self.schedule_d if self.schedule == "shortcut" else None),
        )

        for model in (self.tokenizer, self.dynamics):
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)

        self.checkpoint_dir = root if root.is_dir() else root.parent
        self.tokenizer_path = tok_path
        self.dynamics_path = dyn_path
        self.checkpoint_info = {
            "checkpoint_dir": str(self.checkpoint_dir),
            "tokenizer_path": str(tok_path),
            "dynamics_path": str(dyn_path),
            "tokenizer_args": _json_safe(tok_args),
            "dynamics_args": _json_safe(dyn_args),
            "config": self.config_dict(),
            "schedule": _json_safe(self.tau_schedule),
        }

        LOGGER.info("DreamerV4 config: %s", json.dumps(self.config_dict(), sort_keys=True))

    def reset(self, seed: int = 0) -> np.ndarray:
        if self.dataset_path is None:
            raise RuntimeError("reset() requires dataset_path to be provided at construction.")
        self._ensure_loaded()
        sample, _meta = self._load_dataset_window(
            dataset_path=self.dataset_path,
            task_name=self.task,
            seed=seed,
            context_length=1,
            horizon=1,
        )
        frame0 = sample["obs"][0]
        return self._frames_to_numpy(frame0[None])[0]

    def encode(self, obs: np.ndarray | torch.Tensor) -> dict[str, np.ndarray]:
        z_btLd, z_packed = self._encode_to_tensors(obs)
        return {
            "z_btLd": z_btLd[0].detach().float().cpu().numpy().astype(np.float32),
            "z_packed": z_packed[0].detach().float().cpu().numpy().astype(np.float32),
            "flat": z_packed[0].reshape(z_packed.shape[1], -1)
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float32),
        }

    def imagine(
        self,
        context_frames: np.ndarray | torch.Tensor,
        actions: np.ndarray | torch.Tensor,
        action_mask: np.ndarray | torch.Tensor | None = None,
        horizon: int | None = None,
    ) -> dict[str, np.ndarray]:
        self._ensure_loaded()
        if horizon is None:
            raise ValueError("horizon is required for DreamerV4 imagination.")
        if horizon <= 0:
            raise ValueError("horizon must be positive.")

        modules = _import_v4_modules()
        assert self.dynamics is not None
        assert self.tau_schedule is not None
        assert self.k_max is not None
        assert self.n_spatial is not None
        assert self.d_spatial is not None

        _z_btLd, context_packed = self._encode_to_tensors(context_frames)
        ctx = int(context_packed.shape[1])
        total = ctx + int(horizon)

        actions_t = self._actions_to_tensor(actions, total=total, name="actions")
        mask_t = self._action_mask_to_tensor(action_mask, total=total)

        z_seed = torch.zeros(
            (1, total, self.n_spatial, self.d_spatial),
            device=self.device,
            dtype=context_packed.dtype,
        )
        z_seed[:, :ctx] = context_packed

        with torch.no_grad(), self._autocast_context():
            z_pred = modules.sample_autoregressive_packed_sequence(
                self.dynamics,
                z_gt_packed=z_seed,
                ctx_length=ctx,
                horizon=int(horizon),
                k_max=self.k_max,
                sched=self.tau_schedule,
                actions=actions_t,
                act_mask=mask_t,
            )

        imagined_packed = z_pred[:, ctx : ctx + int(horizon)]
        if imagined_packed.shape[1] != int(horizon):
            raise RuntimeError(
                f"DreamerV4 generated {imagined_packed.shape[1]} steps, expected {horizon}."
            )
        flat = imagined_packed[0].reshape(int(horizon), -1)

        return {
            "imagined_latent": flat.detach().float().cpu().numpy().astype(np.float32),
            "imagined_packed": imagined_packed[0]
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float32),
        }

    def decode(self, z_packed: np.ndarray | torch.Tensor) -> np.ndarray:
        self._ensure_loaded()
        modules = _import_v4_modules()
        assert self.decoder is not None
        assert self.H is not None
        assert self.W is not None
        assert self.C is not None
        assert self.patch is not None

        z = torch.as_tensor(z_packed)
        if z.dim() == 3:
            z = z.unsqueeze(0)
        if z.dim() != 4:
            raise ValueError("z_packed must have shape (T,S,D) or (B,T,S,D).")
        z = z.to(device=self.device, dtype=torch.float32)

        with torch.no_grad(), self._autocast_context():
            frames = modules.decode_packed_to_frames(
                self.decoder,
                z_packed=z,
                H=self.H,
                W=self.W,
                C=self.C,
                patch=self.patch,
                packing_factor=self.packing_factor,
            )
        if frames.shape[0] != 1:
            raise ValueError("decode() expects a single batch item.")
        return frames[0].detach().float().cpu().numpy().astype(np.float32)

    def collect_true_rollout(
        self,
        seed: int,
        total_steps: int,
        imagination_start: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.dataset_path is None:
            raise RuntimeError("collect_true_rollout() requires dataset_path.")
        if total_steps <= 0:
            raise ValueError("total_steps must be positive.")
        if not 0 <= imagination_start < total_steps:
            raise ValueError("imagination_start must be in [0, total_steps).")

        sample, _meta = self._load_dataset_window(
            dataset_path=self.dataset_path,
            task_name=self.task,
            seed=seed,
            context_length=imagination_start,
            horizon=total_steps - imagination_start,
        )
        frames = self._frames_to_numpy(sample["obs"][:-1])
        actions, _mask = self._align_actions(sample["act"], sample["act_mask"])
        rewards = sample["rew"].detach().cpu().numpy().astype(np.float32)
        return frames, actions, rewards

    def extract_rollout(
        self,
        dataset_path: str | Path | None = None,
        task_name: str | None = None,
        seed: int = 0,
        context_length: int = 8,
        horizon: int = 200,
        include_latent: bool = True,
        output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        self._ensure_loaded()
        if context_length <= 0:
            raise ValueError("context_length must be positive.")
        if horizon <= 0:
            raise ValueError("horizon must be positive.")

        data_root = _resolve_path(dataset_path) if dataset_path is not None else self.dataset_path
        if data_root is None:
            raise RuntimeError("dataset_path is required for V4 rollout extraction.")
        task = task_name or self.task

        sample, dataset_meta = self._load_dataset_window(
            dataset_path=data_root,
            task_name=task,
            seed=seed,
            context_length=context_length,
            horizon=horizon,
        )

        true_frames = self._frames_to_numpy(sample["obs"][:-1])
        actions, action_mask = self._align_actions(sample["act"], sample["act_mask"])
        rewards = self._frame_aligned_rewards(sample["rew"], context_length, horizon)

        encoded = self.encode(true_frames)
        imagined = self.imagine(
            true_frames[:context_length],
            actions,
            action_mask,
            horizon=horizon,
        )
        imagined_obs = self.decode(imagined["imagined_packed"])

        true_slice = slice(context_length, context_length + horizon)
        metadata = {
            "version": "v1",
            "task": task,
            "seed": int(seed),
            "context_length": int(context_length),
            "horizon": int(horizon),
            "dataset_path": str(data_root),
            "dataset": dataset_meta,
            "checkpoint_info": self.checkpoint_info,
            "action_alignment": (
                "actions[0] and action_mask[0] are zero; actions[t>=1] is the "
                "dataset action that produced frame t."
            ),
            "latent_definition": (
                "DreamerV4 packed tokenizer latent flattened from "
                f"({self.n_spatial}, {self.d_spatial}) to 512 scalars per frame."
            ),
        }
        rollout: dict[str, Any] = {
            "true_obs": true_frames[true_slice].astype(np.float32),
            "imagined_obs": imagined_obs.astype(np.float32),
            "true_latent": encoded["flat"][true_slice].astype(np.float32),
            "imagined_latent": imagined["imagined_latent"].astype(np.float32),
            "actions": actions.astype(np.float32),
            "action_mask": action_mask.astype(np.float32),
            "rewards": rewards.astype(np.float32),
            "metadata": metadata,
        }
        if not include_latent:
            rollout.pop("true_latent")
            rollout.pop("imagined_latent")

        if output_path is not None:
            save_rollout_npz(output_path, rollout)
        return rollout

    def config_dict(self) -> dict[str, Any]:
        return {
            "k_max": self.k_max,
            "packing_factor": self.packing_factor,
            "n_latents": self.n_latents,
            "d_bottleneck": self.d_bottleneck,
            "n_spatial": self.n_spatial,
            "d_spatial": self.d_spatial,
            "H": self.H,
            "W": self.W,
            "C": self.C,
            "patch": self.patch,
            "schedule": self.schedule,
            "schedule_d": self.schedule_d,
            "use_amp": self.use_amp,
        }

    def _ensure_loaded(self) -> None:
        if self.tokenizer is None or self.dynamics is None:
            raise RuntimeError("Call load_checkpoint() before using DreamerV4Adapter.")

    def _autocast_context(self):
        enabled = self.use_amp and self.device.type == "cuda"
        if enabled:
            return torch.autocast(device_type="cuda", enabled=True)
        return nullcontext()

    def _log_checkpoint_summary(self, path: Path, label: str) -> dict[str, Any]:
        ckpt = _torch_load_cpu(path)
        if isinstance(ckpt, dict):
            keys = list(ckpt.keys())
            args = dict(ckpt.get("args", {}) or {})
        else:
            keys = [type(ckpt).__name__]
            args = {}
        shapes = _sample_tensor_shapes(ckpt)
        LOGGER.info(
            "DreamerV4 %s checkpoint=%s keys=%s args=%s tensor_shapes=%s",
            label,
            path,
            keys,
            _json_safe(args),
            shapes,
        )
        del ckpt
        gc.collect()
        return args

    def _frames_to_tensor(self, obs: np.ndarray | torch.Tensor) -> torch.Tensor:
        assert self.H is not None
        assert self.W is not None
        assert self.C is not None

        x = torch.as_tensor(obs)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.dim() != 4:
            raise ValueError("obs must have shape (3,H,W) or (T,3,H,W).")
        if int(x.shape[1]) != self.C or int(x.shape[2]) != self.H or int(x.shape[3]) != self.W:
            raise ValueError(
                f"Expected obs shape (T,{self.C},{self.H},{self.W}), got {tuple(x.shape)}."
            )

        if x.dtype == torch.uint8:
            x = x.to(torch.float32) / 255.0
        else:
            x = x.to(torch.float32)
            if x.numel() > 0 and float(x.detach().max().item()) > 1.5:
                x = x / 255.0
        x = x.clamp(0.0, 1.0)
        return x.unsqueeze(0).to(device=self.device, non_blocking=True)

    def _frames_to_numpy(self, frames: torch.Tensor | np.ndarray) -> np.ndarray:
        x = torch.as_tensor(frames)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.dtype == torch.uint8:
            x = x.to(torch.float32) / 255.0
        else:
            x = x.to(torch.float32)
            if x.numel() > 0 and float(x.max().item()) > 1.5:
                x = x / 255.0
        return x.clamp(0.0, 1.0).cpu().numpy().astype(np.float32)

    def _encode_to_tensors(
        self,
        obs: np.ndarray | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_loaded()
        modules = _import_v4_modules()
        assert self.encoder is not None
        assert self.patch is not None
        assert self.n_spatial is not None

        frames = self._frames_to_tensor(obs)
        with torch.no_grad(), self._autocast_context():
            patches = modules.temporal_patchify(frames, self.patch)
            z_btLd, _ = self.encoder(patches)
            z_packed = modules.pack_bottleneck_to_spatial(
                z_btLd,
                n_spatial=self.n_spatial,
                k=self.packing_factor,
            )
        return z_btLd, z_packed

    def _actions_to_tensor(
        self,
        actions: np.ndarray | torch.Tensor,
        *,
        total: int,
        name: str,
    ) -> torch.Tensor:
        x = torch.as_tensor(actions, dtype=torch.float32)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.dim() != 3 or x.shape[-1] != 16:
            raise ValueError(f"{name} must have shape (T,16) or (1,T,16).")
        if int(x.shape[1]) < total:
            raise ValueError(f"{name} must contain at least {total} timesteps.")
        x = x[:, :total].clone()
        x[:, 0] = 0.0
        return x.to(device=self.device, non_blocking=True)

    def _action_mask_to_tensor(
        self,
        action_mask: np.ndarray | torch.Tensor | None,
        *,
        total: int,
    ) -> torch.Tensor | None:
        if action_mask is None:
            mask = torch.ones((1, total, 16), dtype=torch.float32)
            mask[:, 0] = 0.0
            return mask.to(device=self.device, non_blocking=True)

        mask = torch.as_tensor(action_mask, dtype=torch.float32)
        if mask.dim() == 1:
            if mask.shape[0] != 16:
                raise ValueError("1D action_mask must have 16 entries.")
            mask = mask.view(1, 1, 16).expand(1, total, 16).clone()
        elif mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if mask.dim() != 3 or mask.shape[-1] != 16:
            raise ValueError("action_mask must have shape (16,), (T,16), or (1,T,16).")
        if int(mask.shape[1]) < total:
            raise ValueError(f"action_mask must contain at least {total} timesteps.")
        mask = mask[:, :total].clone()
        mask[:, 0] = 0.0
        return mask.to(device=self.device, non_blocking=True)

    def _align_actions(
        self,
        raw_actions: torch.Tensor | np.ndarray,
        raw_mask: torch.Tensor | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        act = torch.as_tensor(raw_actions, dtype=torch.float32)
        mask = torch.as_tensor(raw_mask, dtype=torch.float32)
        if act.dim() != 2 or act.shape[-1] != 16:
            raise ValueError(f"Expected raw action shape (T,16), got {tuple(act.shape)}.")
        if mask.shape != act.shape:
            raise ValueError("raw action mask must match raw action shape.")

        aligned = torch.zeros_like(act)
        aligned_mask = torch.zeros_like(mask)
        if act.shape[0] > 1:
            aligned[1:] = act[:-1].clamp(-1, 1) * mask[:-1]
            aligned_mask[1:] = mask[:-1]
        return (
            aligned.cpu().numpy().astype(np.float32),
            aligned_mask.cpu().numpy().astype(np.float32),
        )

    def _frame_aligned_rewards(
        self,
        raw_rewards: torch.Tensor | np.ndarray,
        context_length: int,
        horizon: int,
    ) -> np.ndarray:
        rewards = torch.as_tensor(raw_rewards, dtype=torch.float32)
        if rewards.dim() != 1:
            rewards = rewards.reshape(-1)
        start = max(0, int(context_length) - 1)
        end = start + int(horizon)
        if end > rewards.shape[0]:
            raise ValueError("Reward window exceeds available rewards.")
        return rewards[start:end].cpu().numpy().astype(np.float32)

    def _load_dataset_window(
        self,
        *,
        dataset_path: str | Path,
        task_name: str,
        seed: int,
        context_length: int,
        horizon: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        modules = _import_v4_modules()
        assert self.H is not None
        window_steps = int(context_length) + int(horizon)
        data_dirs, frame_dirs = self._resolve_dataset_sources(
            _resolve_path(dataset_path),
            task_name,
        )
        cache_key = (";".join(map(str, data_dirs + frame_dirs)), task_name, window_steps)
        if cache_key not in self._dataset_cache:
            self._dataset_cache[cache_key] = modules.WMDataset(
                data_dir=[str(path) for path in data_dirs],
                frames_dir=[str(path) for path in frame_dirs],
                seq_len=window_steps,
                img_size=self.H,
                action_dim=16,
                tasks_json=str(DREAMER4_ROOT / "tasks.json"),
                tasks=[task_name],
                verbose=True,
                strict_tasks=True,
            )

        dataset = self._dataset_cache[cache_key]
        if len(dataset) <= 0:
            raise RuntimeError(f"No valid V4 dataset windows for task={task_name}.")
        dataset_index = (int(seed) * 300) % len(dataset)
        sample = dataset[dataset_index]
        meta = {
            "index": int(dataset_index),
            "num_windows": int(len(dataset)),
            "source_data_dirs": [str(path) for path in data_dirs],
            "source_frame_dirs": [str(path) for path in frame_dirs],
        }
        return sample, meta

    def _resolve_dataset_sources(
        self,
        dataset_path: Path,
        task_name: str,
    ) -> tuple[list[Path], list[Path]]:
        sources: list[tuple[Path, Path]] = []
        seen: set[tuple[Path, Path]] = set()

        def add(raw_dir: Path, frame_dir: Path) -> None:
            raw_file = raw_dir / f"{task_name}.pt"
            shard_dir = frame_dir / task_name
            key = (raw_dir.resolve(), frame_dir.resolve())
            if key in seen:
                return
            if raw_file.exists() and shard_dir.exists() and any(shard_dir.glob("*shard*.pt")):
                sources.append((raw_dir, frame_dir))
                seen.add(key)

        add(dataset_path, dataset_path)
        add(dataset_path / "data", dataset_path / "frames")
        add(dataset_path / "raw", dataset_path / "frames")
        add(dataset_path / "raw", dataset_path / "shards")

        for name in ("expert", "mixed-small", "mixed-large"):
            add(dataset_path / name, dataset_path / f"{name}-shards")
            add(dataset_path / name, dataset_path / name / "shards")

        if dataset_path.name.endswith("-shards"):
            raw_name = dataset_path.name[: -len("-shards")]
            add(dataset_path.parent / raw_name, dataset_path)

        if not sources:
            raise FileNotFoundError(
                "Could not find V4 raw demos and frame shards. Expected either "
                f"{dataset_path}/{task_name}.pt with {dataset_path}/{task_name}/*shard*.pt, "
                "or paired source directories such as expert/ and expert-shards/."
            )

        data_dirs, frame_dirs = zip(*sources)
        return list(data_dirs), list(frame_dirs)
